from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

# Load your trained reward model (difficulty estimator)
reward_tokenizer = AutoTokenizer.from_pretrained("/root/results/checkpoint-4672/")
reward_model = AutoModelForSequenceClassification.from_pretrained("/root/results/checkpoint-4672/")

# Load the larger LLM you want to prompt
llm_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/Qwen3-0.6B-Base")
llm_model = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/Qwen3-0.6B-Base", 
                                                   device_map="auto", 
                                                   torch_dtype=torch.float16)

# Mapping from difficulty levels to prompt templates
prompt_templates = {
    1: "This is a moderately easy math question:\n{question}\nProvide a brief explanation with key steps.",
    2: "This is a moderately easy math question:\n{question}\nProvide a brief explanation with key steps.",
    3: "This question has medium difficulty:\n{question}\nProvide a step-by-step solution.",
    4: "This question is hard:\n{question}\nProvide a detailed step-by-step explanation.",
    5: "This question is very hard:\n{question}\nProvide a comprehensive reasoning with justification for each step."
}

def get_difficulty(question: str) -> int:
    """Run the reward model to classify difficulty (1-5)."""
    inputs = reward_tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = reward_model(**inputs)
    logits = outputs.logits  # shape: [1, num_labels]
    # Assume labels are 0-4 corresponding to difficulty 1-5
    pred_label = torch.argmax(logits, dim=-1).item()
    return pred_label + 1

def generate_prompt(question: str) -> str:
    """Select an appropriate prompt based on difficulty level."""
    level = get_difficulty(question)
    template = prompt_templates.get(level)
    return template.format(question=question)

def get_llm_response(question: str, max_new_tokens: int = 4096) -> str:
    """Prompt the larger LLM with the generated prompt and return its response."""
    prompt = generate_prompt(question)
    print(f'Warning! The prompt is {prompt}')
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        generated = llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = llm_tokenizer.decode(generated[0], skip_special_tokens=True)
    # Strip the prompt from the output if echoed
    return output[len(prompt):].strip()

if __name__ == "__main__":
    question = "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$"
    print(f'Inferencing: {question}')
    answer = get_llm_response(question)
    print("\nGenerated Reasoning & Answer:\n", answer)
