import argparse
import pandas as pd
# https://hf-mirror.com/datasets/open-r1/OpenR1-Math-220k/tree/472472d80032b525579a78b5fb8cf5c548ccccd4/extended
def main():
    parser = argparse.ArgumentParser(description="Process dataset with configurable columns")
    
    parser.add_argument('--problem-col', type=str, default='problem',
                        help='Column name for problem text (default: problem)')
    parser.add_argument('--solution-col', type=str, default='solution',
                        help='Column name for solution steps (default: solution)')
    parser.add_argument('--answer-col', type=str, default='answer',
                        help='Column name for final answer (default: answer)')
    parser.add_argument('--correctness-col', type=str, default='correctness_count',
                        help='Column name for correctness count (default: correctness_count)')
    parser.add_argument('--input', type=str, 
                        default='./OpenR1-Math-220k/extended/train-{:05d}-of-00007.parquet',
                        help='Input template file path')
    parser.add_argument('--num-files', type=int, default=7,
                        help='Total number of input files (default: 7)')
    parser.add_argument('--output', type=str, default='./reason_cot_data.parquet',
                        help='Output file path in parquet format')
    
    args = parser.parse_args()

    dfs = [pd.read_parquet(args.input.format(i)) for i in range(args.num_files)]
    df = pd.concat(dfs)

    reannotated_content, problems = [], []
    for p, s, o, a in zip(
        df[args.problem_col],
        df[args.solution_col],
        df[args.answer_col],
        df[args.correctness_col]
    ):
        if a == 2:
            problems.append(p)
            reannotated_content.append(
                f"<think>{s}</think>\n<answer>\\\\box{{{o}}}</answer>"
            )

    pd.DataFrame({
        'instruction': problems,
        'input': "",
        'output': reannotated_content
    }).to_parquet(args.output)

if __name__ == "__main__":
    main()