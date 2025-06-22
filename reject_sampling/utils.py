import os
import re
import regex
import warnings
import multiprocessing
from sympy import simplify, N
from word2number import w2n
from math import isclose
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
from multiprocessing import Process, Queue
from typing import Union

from datasets import interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm


separate_template="""Separate the following multi-step math problem into two questions. The answer to the first question should be a value that is then given as known information in the second question. For each question, provide the ground truth answer.

Example:

Original problem:
Tobias is buying a new pair of shoes that costs \$95. He has been saving up his money each month for the past three months. He gets a \$5 allowance a month. He also mows lawns and shovels driveways. He charges \$15 to mow a lawn and \$7 to shovel. After buying the shoes, he has \$15 in change. If he mows 4 lawns, how many driveways did he shovel?

1. Tobias is buying a new pair of shoes that costs \$95. After buying the shoes, he has \$15 left. How much money did Tobias save in total?
Ground truth answer: \$110

2. Tobias saved a total of \$110 over the past three months. He gets a \$5 allowance per month. He also earned money by mowing lawns (\$15 each) and shoveling driveways (\$7 each). If he mowed 4 lawns, how many driveways did he shovel?
Ground truth answer: 5 driveways

Now, generate a new question pair using a similar structure:
{question}
"""


def blending_datasets(
        datasets,
        probabilities,
        strategy=None,
        seed=42,
        max_count=5000000,
        stopping_strategy="first_exhausted",
        train_split="train"
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    if len(probabilities) != len(datasets):
        raise ValueError(f"Length of probabilities ({len(probabilities)}) must match the length of datasets ({len(datasets)})")

    train_data_list = []
    for _, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        if ext == ".py" or (os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    return train_dataset


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key] if input_key in data.keys() else None
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptGtAnswerDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            strategy,
            input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        self.input_template = input_template

        input_key = self.strategy.args.map_keys.get("prompt", None)
        gt_answer_key = self.strategy.args.map_keys.get("gt_answer", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.gt_answers = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            gt_answer = preprocess_data(data, input_key=gt_answer_key)
            self.prompts.append(prompt)
            self.gt_answers.append(gt_answer)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "gt_answer": self.gt_answers[idx]}


def apply_GenRM_template(task, prompt_text, response_text=None, ground_truth_answer_text=False, language='en'):
    if task == 'reject_sampling_math_difficult':
        if language == "en":
            full_input = f"""
                    You are an expert in evaluating math problems based on their difficulty level. Your task is to analyze the following math question and provide a difficulty score. The difficulty score should be enclosed in <score></score> tags, where:
                    - A score of 1 means the question is very easy (e.g., basic arithmetic).
                    - A score of 2 means the question is relatively simple, but requires some thought (e.g., single-variable algebra).
                    - A score of 3 means the question is moderately difficult, requiring multiple steps or knowledge of intermediate concepts (e.g., basic calculus or linear algebra).
                    - A score of 4 means the question is challenging, requiring advanced knowledge or multiple complex steps (e.g., advanced calculus or multi-variable problems).
                    - A score of 5 means the question is very difficult (e.g., complex calculus or advanced topics).
                    Please consider factors such as the complexity of the problem, required knowledge, and the number of steps needed to solve it.

                    Examples:
                    1. Question: "What is 2 + 2?"
                    Output: <score>1</score>

                    2. Question: "What is the value of x if 2x + 3 = 7?"
                    Output: <score>2</score>

                    3. Question: "What is the derivative of 3x^2 + 2x?"
                    Output: <score>3</score>

                    4. Question: "Evaluate the integral of x^3 from 0 to 1."
                    Output: <score>4</score>

                    5. Question: "Find the critical points and classify them for the function f(x) = x^4 - 4x^3 + 6x^2."
                    Output: <score>5</score>

                    Now, please evaluate the following question:

                    Question: "{prompt_text}"

                    Please provide the difficulty score.
                    """
    if ground_truth_answer_text:
        if language == "zh":
            full_input = f"""
                    你是一名专家，负责评估语言模型助手对数学问题的回答表现。下面给出一个数学问题和对应的回答，你需要根据10分制对该回答进行评分，
                    其中1分为最差，10分为最佳。
                    你应考虑以下方面来评估回答：
                    正确性：解题思路和最终答案是否正确。如果最终答案错误，最高不超过5分。
                    完整性：是否完整展示了解题过程,包括关键步骤和推导过程。
                    清晰性：解题过程的表述是否清晰,公式符号使用是否规范。
                    教学价值：是否解释了重要概念,帮助理解问题。
                    打分标准：
                    [1-2] 低：答案错误,且解题思路存在严重错误,或未给出任何解题过程。
                    [3-4] 中等：答案可能正确但解题思路有误,或解题过程严重不完整。
                    [5-6] 高：答案和主要解题思路正确,但解题过程不够完整或清晰,缺乏必要解释。
                    [7-8] 非常高：答案正确,解题过程完整且清晰,但可能在某些细节上略显不足。
                    [9-10] 优异：答案完全正确,解题过程非常完整清晰,概念解释到位,具有很好的教学价值。
                    请首先从正确性、完整性、清晰性、教学价值这几个方面对回答进行分析，然后罗列出回答的优缺点，最后给出总体评分，注意总体评分应
                    该是一个1到10之间(包括1和10)的整数。

                    问题
                    {prompt_text}

                    回答
                    {response_text}
                    
                    参考答案
                    {ground_truth_answer_text}

                    分析
                    [你的分析内容]

                    总体评分
                    请将分数用<score></score>标记输出，即以<score>分数</score>的格式输出
                """
        elif language == "en":
            full_input = f"""You are an expert responsible for evaluating the performance of a language model assistant in answering math problems. Below is a math problem along with its corresponding answer. You need to rate the answer on a scale of 1 to 10, where 1 is the worst and 10 is the best.\n
                You should consider the following aspects when evaluating the answer:\n
                Proper Tag Pairing: Always check that the `<think>` tag has a corresponding `</think>` tag, and the `<answer>` tag has a corresponding `</answer>` tag, with no overlapping or mixed content between them.\n
                Separation of Reasoning and Answer: Always enclose the complete chain-of-thought in `<think>` and `</think>` tags, and place the final computed answer in `<answer>` and `</answer>` tags immediately after.\n
                Complete Step-by-Step Explanation: Ensure the reasoning includes every important detail, such as intermediate steps, assumptions, substitutions, and simplifications, so that the derivation is clear and understandable.\n
                Clear and Organized Presentation: Write the chain-of-thought in a logical, structured manner using clear language, proper formatting (with LaTeX for equations when needed), and natural breaks to enhance readability.\n
                Logical Consistency and Verification: Each step in the reasoning should logically follow from the previous one. The chain-of-thought must allow a reviewer to replicate the derivation and arrive at the same final answer.\n
                Uniformity Across Problems: Apply the same detailed formatting and level of explanation across all problems to maintain a consistent style and facilitate reliable scoring by evaluation systems.\n
                Conciseness and Readability: While providing detailed reasoning is key, ensure every sentence is purposeful. Avoid unnecessary verbosity, keeping the explanation neat, precise, and free of ambiguities.\n
                [PROMPT]: {prompt_text}
                [RESPONSE]: {response_text}
                Task Objective: Based on the given problem [PROMPT], evaluate the response quality [RESPONSE], considering Proper Tag Pairing, Separation of Reasoning and Answer, Complete Step-by-Step Explanation, Clear and Organized Presentation, Logical Consistency and Verification, Uniformity Across Problems, and Conciseness and Readability. Use concise language to explain your reasons; you do not need to provide an improved answer. Finally, assign a score between 1 and 10, with the score enclosed in <score></score>.
                """
    else:
        if language == "zh":
            full_input = f"""
                    你是一名专家，负责评估语言模型助手对数学问题的回答表现。下面给出一个数学问题和对应的回答，你需要根据10分制对该回答进行评分，
                    其中1分为最差，10分为最佳。
                    你应考虑以下方面来评估回答：
                    正确性：解题思路和最终答案是否正确。如果最终答案错误，最高不超过5分。
                    完整性：是否完整展示了解题过程,包括关键步骤和推导过程。
                    清晰性：解题过程的表述是否清晰,公式符号使用是否规范。
                    教学价值：是否解释了重要概念,帮助理解问题。
                    打分标准：
                    [1-2] 低：答案错误,且解题思路存在严重错误,或未给出任何解题过程。
                    [3-4] 中等：答案可能正确但解题思路有误,或解题过程严重不完整。
                    [5-6] 高：答案和主要解题思路正确,但解题过程不够完整或清晰,缺乏必要解释。
                    [7-8] 非常高：答案正确,解题过程完整且清晰,但可能在某些细节上略显不足。
                    [9-10] 优异：答案完全正确,解题过程非常完整清晰,概念解释到位,具有很好的教学价值。
                    请首先从正确性、完整性、清晰性、教学价值这几个方面对回答进行分析，然后罗列出回答的优缺点，最后给出总体评分，注意总体评分应
                    该是一个1到10之间(包括1和10)的整数。

                    问题
                    {prompt_text}

                    回答
                    {response_text}

                    分析
                    [你的分析内容]

                    总体评分
                    请将分数用<score></score>标记输出，即以<score>分数</score>的格式输出
                    """
        elif language == "en":
            full_input = f"""You are an expert responsible for evaluating the performance of a language model assistant in answering math problems. Below is a math problem along with its corresponding answer. You need to rate the answer on a scale of 1 to 10, where 1 is the worst and 10 is the best.\n
                You should consider the following aspects when evaluating the answer:\n
                Proper Tag Pairing: Always check that the `<think>` tag has a corresponding `</think>` tag, and the `<answer>` tag has a corresponding `</answer>` tag, with no overlapping or mixed content between them.\n
                Separation of Reasoning and Answer: Always enclose the complete chain-of-thought in `<think>` and `</think>` tags, and place the final computed answer in `<answer>` and `</answer>` tags immediately after.\n
                Complete Step-by-Step Explanation: Ensure the reasoning includes every important detail, such as intermediate steps, assumptions, substitutions, and simplifications, so that the derivation is clear and understandable.\n
                Clear and Organized Presentation: Write the chain-of-thought in a logical, structured manner using clear language, proper formatting (with LaTeX for equations when needed), and natural breaks to enhance readability.\n
                Logical Consistency and Verification: Each step in the reasoning should logically follow from the previous one. The chain-of-thought must allow a reviewer to replicate the derivation and arrive at the same final answer.\n
                Uniformity Across Problems: Apply the same detailed formatting and level of explanation across all problems to maintain a consistent style and facilitate reliable scoring by evaluation systems.\n
                Conciseness and Readability: While providing detailed reasoning is key, ensure every sentence is purposeful. Avoid unnecessary verbosity, keeping the explanation neat, precise, and free of ambiguities.\n
                [PROMPT]: {prompt_text}
                [RESPONSE]: {response_text}
                [REFERENCE]: {ground_truth_answer_text}
                Task Objective: Based on the given problem [PROMPT] and reference answer [REFERENCE], evaluate the response quality [RESPONSE], considering Proper Tag Pairing, Separation of Reasoning and Answer, Complete Step-by-Step Explanation, Clear and Organized Presentation, Logical Consistency and Verification, Uniformity Across Problems, and Conciseness and Readability. Use concise language to explain your reasons; you do not need to provide an improved answer. Finally, assign a score between 1 and 10, with the score enclosed in <score></score>.
                """

    return full_input


def rejection_sampling_processor(objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        prompt = obj["prompt"]
        output = obj["output"]
        reward = float(obj["reward"])

        if reward > 0:
            if prompt not in out:
                out[prompt] = {"output": output, "reward": reward}
            elif reward > out[prompt]["reward"]:
                out[prompt]["reward"] = reward
                out[prompt]["output"] = output

    return [{"prompt": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]


def rejection_sampling_math_difficulty_processor(objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        prompt = obj["input"]
        reward = float(obj["reward"])

        if reward > 0:
            if prompt not in out:
                out[prompt] = {"input": prompt, "reward": reward}
            elif reward > out[prompt]["reward"]:
                out[prompt]["reward"] = reward
                out[prompt]["input"] = prompt

    return [{"prompt": k, "difficulty": v["reward"]} for k, v in out.items()]


def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):

    def worker(q, prediction, reference):
        result = math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    p.join(timeout=timeout_seconds)
    
    if p.is_alive():
        p.terminate()
        p.join()  
        return False
        
    try:
        return q.get_nowait()
    except Exception as e:
        return False   


def preprocess_box_response_for_qwen_prompt(sequences, answers, **kwargs):
    scores = []

    for sequence, answer in zip(sequences, answers):
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL, count=1)
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
        for stop_word in stop_words:
            if stop_word in model_output:
                model_output = model_output.split(stop_word)[0].strip()
        ext_answer = extract_answer_subprocess(model_output=model_output)

        if qwen_math_equal_subprocess(prediction=ext_answer, reference=answer):
            box_match = 1.0
        else:
            box_match = -0.5
            
        if "boxed" not in model_output:
            box_match = -1.0

        scores.append(box_match)
        
    return scores


def extract_answer_subprocess(model_output, timeout_seconds=10):

    def worker(q, model_output):
        result = extract_answer(pred_str=model_output, data_name="math")
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, model_output))
    p.start()
    
    p.join(timeout=timeout_seconds)
    
    if p.is_alive():
        p.terminate()
        p.join()  
        return ""
        
    try:
        return q.get_nowait()
    except Exception as e:
        return ""

def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def is_digit(num):
    return parse_digits(num) is not None


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    string = string.replace("\n", "")

    string = string.rstrip(".")
    string = string.replace("\\!", "")

    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    if not skip_unit:
        for _ in range(2):
            for unit_text in unit_texts:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    string = convert_word_number(string)

    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)

    string = _fix_a_slash_b(string)

    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        if string != "{}/{}".format(a, b):
            raise ValueError(f"String does not match the expected format: {string}")
        new_string = f"\\frac{{{a}}}{{{b}}}"
        return new_string
    except ValueError:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except ValueError:
        pass
    return text


unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    string = string.replace("\n", "")

    string = string.rstrip(".")
    string = string.replace("\\!", "")

    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    if not skip_unit:
        for _ in range(2):
            for unit_text in unit_texts:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    string = convert_word_number(string)

    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)

    string = _fix_a_slash_b(string)

    return string


def extract_multi_choice_answer(pred_str):
    if "Problem:" in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group("ans").upper()
    return "placeholder"


direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            pred = pred[0]
        else:
            pred = pred[-1]

    pred = pred.rstrip(".").rstrip("/")

    return pred


def extract_answer(pred_str, data_name, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]

def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception as e:
                    warnings.warn(f"An exception occurred during comparison: {e}")
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = []
        prediction_processed = prediction[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")]
        for line in prediction_processed.split("\\\\"):
            stripped_line = line.strip()
            if stripped_line:
                pred_lines.append(stripped_line)

        ref_lines = []
        reference_processed = reference[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")]
        for line in reference_processed.split("\\\\"):
            stripped_line = line.strip()
            if stripped_line:
                ref_lines.append(stripped_line)

        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass
    return False


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)
    
    
def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        m_processed = m.replace(",", "\\")
        pmatrix = fr"\begin{{{m_processed}}}\end{{pmatrix}}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)