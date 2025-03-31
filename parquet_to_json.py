import pandas as pd
import json
from pathlib import Path

def parquet_to_custom_json(input_path, output_path=None, indent=4):
    """
    """
    df = pd.read_parquet(input_path)

    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            "instruction": row.get("instruction", ""),
            "output": row.get("output", "")
        })
    
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.with_suffix('.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=indent, ensure_ascii=False)
    
    print(f"Finished: {input_path} → {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('input', help='...')
    parser.add_argument('-o', '--output', help='...')
    parser.add_argument('--indent', type=int, default=4, help='...')
    
    args = parser.parse_args()
    
    try:
        output_path = parquet_to_custom_json(
            args.input,
            args.output,
            args.indent
        )
    except Exception as e:
        print(f"错误: {str(e)}")