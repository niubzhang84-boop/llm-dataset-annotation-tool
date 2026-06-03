import json
import requests

# 本地 Ollama 的 API 地址（默认是这个）
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_ai_label(question, answer):
    """
    调用本地大模型，对问答对进行分类标签预测
    """
    # 构造大模型极其需要的 Prompt（提示词工程）
    prompt = f"""
你是一个专业的数据标注员。请阅读以下【用户问题】和【专家回答】，并将其精准归类到以下三个标签之一：
- [网络设备配置] (涉及交换机、路由器、IP配置等)
- [物联网硬件] (涉及传感器、嵌入式、硬件芯片等)
- [Linux操作系统] (涉及Linux命令、内核、系统操作等)

【用户问题】: {question}
【专家回答】: {answer}

注意：你只能输出标签名字，例如：[网络设备配置]，绝对不要输出任何其他解释、标点或多余的话！
"""

    data = {
        "model": "qwen2.5:1.5b",  # 确保和你下载的模型名称一致
        "prompt": prompt,
        "stream": False           # 关闭流式输出，一次性返回结果
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "").strip()
            return ai_response
        else:
            return "[未知分类]"
    except Exception as e:
        print(f"请求大模型出错: {e}")
        return "[未知分类]"

def main():
    input_file = "train_data.jsonl"
    output_file = "labeled_data.jsonl"
    
    print("开始调用大模型进行自动化标注，请稍候...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            question = data.get("input", "")
            answer = data.get("output", "")
            
            # 让大模型打标签
            label = get_ai_label(question, answer)
            
            # 将新标签塞进原来的数据结构里
            data["category"] = label
            
            # 实时写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"已处理问题: '{question[:10]}...' -> AI标注标签: {label}")

    print(f"\n全部标注完成！新数据集已保存至: {output_file}")

if __name__ == "__main__":
    main()