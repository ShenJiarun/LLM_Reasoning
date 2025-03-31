# import pandas as pd
# import json
# import argparse
# from pathlib import Path

# def parquet_to_json(input_path, output_path=None, indent=None, orient='records'):
#     """
#     将Parquet文件转换为JSON文件
    
#     参数:
#         input_path (str): 输入的Parquet文件路径
#         output_path (str, optional): 输出的JSON文件路径。如果未提供，则使用输入路径但扩展名为.json
#         indent (int, optional): JSON缩进空格数。None表示不缩进
#         orient (str): JSON格式方向，可以是'records'、'split'、'index'等，默认为'records'
        
#     返回:
#         str: 输出的JSON文件路径
#     """
#     # 读取Parquet文件
#     df = pd.read_parquet(input_path)
    
#     # 确定输出路径
#     if output_path is None:
#         input_path = Path(input_path)
#         output_path = input_path.with_suffix('.json')
    
#     # 转换为JSON并保存
#     df.to_json(output_path, indent=indent, orient=orient, force_ascii=False)
    
#     print(f"成功将 {input_path} 转换为 {output_path}")
#     return str(output_path)

# if __name__ == "__main__":
#     # 设置命令行参数
#     parser = argparse.ArgumentParser(description='将Parquet文件转换为JSON文件')
#     parser.add_argument('input', help='输入的Parquet文件路径')
#     parser.add_argument('-o', '--output', help='输出的JSON文件路径（可选）')
#     parser.add_argument('-i', '--indent', type=int, help='JSON缩进空格数（可选）')
#     parser.add_argument('--orient', default='records', 
#                        help='JSON格式方向（records/split/index等），默认为records')
    
#     args = parser.parse_args()
    
#     # 执行转换
#     try:
#         output_path = parquet_to_json(
#             args.input,
#             args.output,
#             args.indent,
#             args.orient
#         )
#         print(f"转换完成，结果已保存到: {output_path}")
#     except Exception as e:
#         print(f"转换过程中发生错误: {str(e)}")

import pandas as pd
import json
from pathlib import Path

def parquet_to_custom_json(input_path, output_path=None, indent=4):
    """
    将 Parquet 文件转换为指定的 JSON 格式
    
    参数:
        input_path (str): 输入的 Parquet 文件路径
        output_path (str, optional): 输出的 JSON 文件路径
        indent (int): JSON 缩进空格数（默认 4）
        
    返回:
        str: 输出的 JSON 文件路径
    """
    # 读取 Parquet 文件
    df = pd.read_parquet(input_path)
    
    # 转换为目标格式的列表
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            "instruction": row.get("instruction", ""),  # 如果列名不同，请调整
            "output": row.get("output", "")             # 如果列名不同，请调整
        })
    
    # 确定输出路径
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.with_suffix('.json')
    
    # 写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=indent, ensure_ascii=False)
    
    print(f"转换完成: {input_path} → {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将 Parquet 转换为指定格式的 JSON')
    parser.add_argument('input', help='输入的 Parquet 文件路径')
    parser.add_argument('-o', '--output', help='输出的 JSON 文件路径（可选）')
    parser.add_argument('--indent', type=int, default=4, help='JSON 缩进空格数（默认 4）')
    
    args = parser.parse_args()
    
    try:
        output_path = parquet_to_custom_json(
            args.input,
            args.output,
            args.indent
        )
    except Exception as e:
        print(f"错误: {str(e)}")