import json
import re

def clean_text(text):
    """
    清洗文本：去除无意义的特殊符号和HTML标签
    """
    if not text:
        return ""
    # 去除粗体、标题等Markdown/文本杂质符号（如 ###, ***）
    text = re.sub(r'[\s#\*]+', ' ', text)
    # 去除网页换行标签 <br>
    text = re.sub(r'<br\s*/?>', '', text)
    return text.strip()

def process_pipeline(input_file, output_file):
    """
    主处理流水线
    """
    processed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # 按照自定义的分隔符 === 分割每条对话
        raw_dialogues = f.read().split('===')
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for dialogue in raw_dialogues:
            lines = dialogue.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # 提取用户和助手的文本
            user_content = ""
            assistant_content = ""
            for line in lines:
                if line.startswith("用户:"):
                    user_content = line.replace("用户:", "").strip()
                elif line.startswith("助手:"):
                    assistant_content = line.replace("助手:", "").strip()
            
            # 1. 执行文本清洗
            cleaned_user = clean_text(user_content)
            cleaned_assistant = clean_text(assistant_content)
            
            # 2. 质量过滤：如果对话内容过短（比如“在吗”“在的”），判定为低质量数据，直接过滤掉
            if len(cleaned_user) < 5 or len(cleaned_assistant) < 5:
                skipped_count += 1
                continue
            
            # 3. 转化为大模型标准的微调格式 (Instruction Dataset 常用格式)
            data_structure = {
                "instruction": "你是一个专业的物联网与网络技术专家，请回答用户的问题。",
                "input": cleaned_user,
                "output": cleaned_assistant
            }
            
            # 4. 写入 JSONL 文件（每一行是一个独立的 JSON 对象）
            out_f.write(json.dumps(data_structure, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"数据处理完成！成功转换: {processed_count} 条，过滤低质量数据: {skipped_count} 条。")

if __name__ == "__main__":
    # 执行处理
    process_pipeline("raw_data.txt", "train_data.jsonl")