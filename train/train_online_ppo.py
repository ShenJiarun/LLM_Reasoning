from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import json
import re
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import sympy as sp
from sympy import simplify, sympify
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


@dataclass
class Samples:
    prompts:       Optional[str]
    seqs:          torch.Tensor  # [B, max_length + max_new_tokens] [batch_size, total number of tokens]
    attention_mask:torch.Tensor  # [B, L] 1=token is real
    action_mask:   torch.Tensor  # [B, L_resp] 1=token counts toward policy loss
    num_actions:   int           # scalar
    packed_seq_lens: Optional[torch.Tensor]
    response_length:torch.Tensor # [B]
    total_length:  torch.Tensor  # [B]
    gt_list: Optional[str]  # [B]


@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


class PromptDataset(Dataset):
    def __init__(self, prompt_list, actor_tokenizer, apply_chat_template):
        super().__init__()
        self.prompt_list = prompt_list
        self.actor_tokenizer = actor_tokenizer
        self.apply_chat_template = apply_chat_template

        self.post_prompt_list = []

        for prompt in self.prompt_list:
            if self.apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                post_prompt = self.actor_tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                post_prompt = self.tokenizer.bos_token + prompt
                
            self.post_prompt_list.append(post_prompt)
    
    def __len__(self):
        return len(self.post_prompt_list)
    
    def __getitem__(self, index):
        return self.post_prompt_list[index]


class ExperienceBuffer:
    """
    To store the history information which is also called Experience
    """
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
        "seqs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "num_actions"
    )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
          
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]


def generate_samples(prompts, gt_list, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    """
    main function used to generate n_samples_per_prompt when doing actor roll-out
    micro_rollout_batch_size: avoid oom
    
    return:
    len(all_prompts) // micro_rollout_batch_size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_list = []
    model.eval()
    all_prompts = []
    all_gts = []
    for index in range(len(prompts)):
        prompt = prompts[index]
        gt = gt_list[index]
        temp_prompt = [prompt] * n_samples_per_prompt
        temp_gt = [gt] * n_samples_per_prompt
        all_prompts.extend(temp_prompt)
        all_gts.extend(temp_gt)
    
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        batch_prompts = all_prompts[i: i+micro_rollout_batch_size]
        inputs = actor_tokenizer.batch_encode_plus(batch_prompts,
                                                    padding="max_length",
                                                    max_length=max_length,
                                                    truncation=True,
                                                    return_tensors="pt",
                                                ).to(model.device)
        input_ids = inputs["input_ids"]
        seqs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            padding_tokens = torch.full(size=(seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=actor_tokenizer.pad_token_id)
            seqs = torch.cat([seqs, padding_tokens], dim=1)
        
        # [B, L]
        attention_mask = (seqs != pad_token_id).long()
        resp_start = input_ids.size(1)
        # response slice
        ans = seqs[:, resp_start:]
        # [B, L_resp]
        action_mask = ((ans != pad_token_id) & (ans != eos_token_id)).long()
        
        samples = Samples(
            prompts=prompts,
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
            gt_list=all_gts
        )
        samples_list.append(samples)

    return samples_list
    

def generate_experiences(samples):
    """
    use history result 'samples' to update the actor model in PPO algorithm
    """
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()
    experiences = []
    
    for sample in samples:
        prompts = sample.prompts
        seqs = sample.seqs
        attention_mask = sample.attention_mask
        action_mask = sample.action_mask
        num_actions = sample.num_actions
        gt_list = sample.gt_list

        with torch.no_grad():
            output = actor_model(seqs.to(actor_model.device), attention_mask=attention_mask.to(actor_model.device))
            logits = output.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            # the log probability of tokens for reference model
            ref_output = ref_model(seqs.to(ref_model.device), attention_mask=attention_mask.to(ref_model.device))
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs.to(ref_model.device)[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # compute the values
            value = critic_model.forward(seqs.to(critic_model.device), attention_mask.to(critic_model.device), num_actions).to(critic_model.device)
            # compute the reward value at the output level
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(reward_model.device)).logits
            r = reward_post_process(r, prompts, seq_texts, gt_list)
            # compute KL value for update
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            rewards = compute_rewards(kl.to(r.device), r, action_mask.to(r.device), kl_ctl=0.1, clip_reward_value=0.2)
            # compute the advantages and returns
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)

        experiences.append(Experience(seqs,
                    action_log_probs.detach(),
                    value.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    sample.response_length,
                    sample.total_length,
                    num_actions,
                    kl.detach(),
        ))

    return experiences


def reward_post_process(reward, batch_prompts, batch_gen, gt_list, alpha=0.8):
    rewards = reward.tolist()
    origin_rewards = []
    final_reward = []

    for r in rewards:
        origin_rewards.append(r[-1])

    # origin_rewards has length batch_size/micro batch_size
    for i, (p, gen, gt) in enumerate(zip(batch_prompts, batch_gen, gt_list)):
        pred = extract_answer(gen)
        correct = float(equiv(gt, pred))
        # scalar
        R = alpha*correct + (1-alpha)*origin_rewards[i]
        # only last token rewarded
        final_reward.append([R])
    
    return torch.tensor(final_reward)
    
            

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):

    log_ratio = log_probs.to(ref_log_probs.device).float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask.to(log_ratio.device)

    return log_ratio


def extract_answer(text: str) -> str:
    """grab the first <answer> ... </answer> pair"""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    return m.group(1).strip() if m else ""


def equiv(gt: str, pred: str) -> bool:
    """symbolic / numeric equivalence test"""
    try:
        return sp.simplify(sp.sympify(gt) - sp.sympify(pred)) == 0
    except Exception:
        return False


def style_score(style_rm, style_tok, prompt, generation, device):
    """small LM that rates CoT clarity, returns float in [0,1]"""
    with torch.no_grad():
        enc = style_tok(prompt + generation,
                        return_tensors="pt",
                        truncation=True,
                        padding=True).to(device)
        s = torch.sigmoid(style_rm(**enc).logits)[0, 0]
    return s.item()


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate

    ends = action_mask.sum(1) + 1

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

    # reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
    reward_clip = r
    batch_size = r.size(0)

    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]

    return rewards


def collate_fn(batch):
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))


def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    clip_eps = torch.tensor(clip_eps)
    advantages = advantages.to(log_probs.device)
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    clipped_surrogate = -torch.min(surr1, surr2)
    if action_mask is None:
        return clipped_surrogate.mean(-1).mean()
    return ((clipped_surrogate * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns.to(values.device)) ** 2
        
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages,action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")


def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            samples = generate_samples(rand_prompts, gt_list, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
                    if steps >= 20:
                        break
                if steps >= 20:
                    break
            if steps >= 20:
                break
            
            buffer.clear()
        
            torch.cuda.empty_cache()


def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask.to(rewards.device) * rewards
    # [batch_size, response_length]
    values = values.to(rewards.device)

    for t in range(response_length - 1, -1, -1):
        nextvalues = values[:, t+1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


class CriticModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        print(f"initialing critic model...")
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, input_ids, attention_mask, num_actions):
        hidden_size = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value_head_output = self.value_head(hidden_size)
        values = value_head_output.squeeze(-1)[:, -num_actions:]
        return values
            

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # num of epoch
    episodes = 1
    # 生成一次经验，训练的轮数
    max_epochs = 2
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    # 记录日志
    writer = SummaryWriter('./runs')
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained('/root/Qwen/Qwen3-0.6B-Base').to('cuda:0')
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/root/Qwen/Qwen3-0.6B-Base').to('cuda:1')
    # 奖励模型
    from transformers import AutoModelForSequenceClassification
    reward_model = AutoModelForSequenceClassification.from_pretrained('/root/reward-model-deberta-v3-large-v2').to('cuda:2')
    actor_tokenizer = AutoTokenizer.from_pretrained('/root/Qwen/Qwen3-0.6B-Base')
    reward_tokenizer = AutoTokenizer.from_pretrained('/root/reward-model-deberta-v3-large-v2')
    # 价值模型
    critic_model = CriticModel(actor_model.base_model).to('cuda:3')
    
    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.000003)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.000005)
    
    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = []
    gt_list = []
    with open('/root/train_gsm8k_origin.json', 'r') as file:
        data = json.load(file)
    
    for index, item in enumerate(data):
        prompt_list.append(item['question'])
        gt_list.append(item['answer'].split('#### ')[-1])

    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
    train()
    actor_model.save_pretrained('./save')