import json

def load_json_file(file_path):
    """
    加载并解析JSON文件
    
    参数:
        file_path (str): JSON文件的路径
        
    返回:
        dict/list: 解析后的JSON数据
        
    异常:
        FileNotFoundError: 当文件不存在时
        json.JSONDecodeError: 当文件不是有效的JSON格式时
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("JSON文件加载成功！")
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        raise
    except json.JSONDecodeError as e:
        print(f"错误：文件 '{file_path}' 不是有效的JSON格式。")
        print(f"错误详情：{str(e)}")
        raise

if __name__ == "__main__":
    # 示例用法
    file_path = "/root/reason_cot_data.json"
    
    try:
        json_data = load_json_file(file_path)
        breakpoint()
        print("\n文件内容:")
        print(json.dumps(json_data, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"加载JSON文件时出错: {str(e)}")