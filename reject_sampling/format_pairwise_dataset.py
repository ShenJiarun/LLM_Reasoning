import argparse
import gc
import json
import re
import logging

import jsonlines
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def load_json_file(args, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info(f"Successfully loaded JSON file from {file_path}...")
            return data
    except FileNotFoundError:
        logger.info(f"Error: file '{file_path}' is not found.")
        raise
    except json.JSONDecodeError as e:
        logger.info(f"Error: file '{file_path}' is not vaild ")
        logger.info(f"Error information: {str(e)}")
        raise

import json

def read_jsonl_to_list(args, file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data_dict = json.loads(line)
                data_list.append(data_dict)
    logger.info(f"Successfully loaded JSON file from {file_path}...")
    return data_list


def format_sharegpt_dataset(args):
    output_dataset = []
    readin_accept_data = read_jsonl_to_list(args, args.input_accept)
    readin_reject_data = read_jsonl_to_list(args, args.input_reject)
    
    assert len(readin_accept_data) == len(readin_reject_data), "Your input accept data and reject data must be of same length."
    
    for index in range(len(readin_accept_data)):
        assert readin_accept_data[index]["prompt"] == readin_reject_data[index]["prompt"]
        prompt = readin_accept_data[index]["prompt"]
        raw_data = {"conversations": [{"from": "human", "value": prompt}],
                    "chosen": {"from": "gpt", "value": readin_accept_data[index]["output"]},
                    "rejected": {"from": "gpt", "value": readin_reject_data[index]["output"]}
                    }
    
        output_dataset.append(raw_data)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dataset, f, indent=4, ensure_ascii=False)


def format_implicit_prompt_preference_dataset(args):
    output_dataset = []
    readin_accept_data = read_jsonl_to_list(args, args.input_accept)
    readin_reject_data = read_jsonl_to_list(args, args.input_reject)
    
    assert len(readin_accept_data) == len(readin_reject_data), "Your input accept data and reject data must be of same length."
    
    for index in range(len(readin_accept_data)):
        assert readin_accept_data[index]["prompt"] == readin_reject_data[index]["prompt"]
        prompt = readin_accept_data[index]["prompt"]
        raw_data = { "chosen": [{"content": prompt, "role": "user"},
                    {"content": readin_accept_data[index]["output"], "role": "assistant"}
                    ],
                     "rejected": [{"content": prompt, "role": "user"},
                    {"content": readin_reject_data[index]["output"], "role": "assistant"}
                    ]
        }
    
        output_dataset.append(raw_data)

    df = pd.DataFrame(output_dataset)
    df.to_parquet(args.output_path, engine='pyarrow', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-accept', '--accept', help='Your accepted data path.')
    parser.add_argument('--input-reject', '--output', help='Your rejected data path.')
    parser.add_argument('--output-format', type=str, choices=['sharegpt_preference', 'implicit_prompt_preference'],
                        default='sharegpt', help='Your output data format.')
    parser.add_argument('--output-path', type=str, help='Your output data directory.')

    args = parser.parse_args()

    if args.output_format and args.output_format == "sharegpt_preference":
        format_sharegpt_dataset(args)
    elif args.output_format and args.output_format == "implicit_prompt_preference":
        format_implicit_prompt_preference_dataset(args)
