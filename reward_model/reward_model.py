import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.reward_head = nn.Linear(self.transformer.config.hidden_size, 1)
        self._init_weights(self.reward_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use the same initializer_range as the transformer
            module.weight.data.normal_(mean=0.0, std=self.transformer.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        # For example, use the representation of the first token (e.g., [CLS])
        hidden_state = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(hidden_state)
        return reward

model_name = ""
reward_model = RewardModel(model_name)
