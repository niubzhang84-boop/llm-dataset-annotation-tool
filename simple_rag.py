import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 升级为中文专属的高精度向量化模型（约 400MB，换源后下载很快）
# ==========================================
print("🔄 正在加载工业级中文向量化模型（text2vec），请稍候...")
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 2. 依然是我们的私有知识库
knowledge_base = [
    "张浩凡开发的智能标注系统采用了PyQt5多线程架构，底层连接本地Ollama服务。",
    "远元生物科技公司的核心服务器IP地址为 192.168.10.254，网关为 192.168.10.1。",
    "物联网传感器在检测到温度超过60度时，会自动触发蜂鸣器报警并向Linux系统发送中断信号。"
]

print("🔄 正在将私有知识库进行高精度向量化转换...")
kb_embeddings = embedding_model.encode(knowledge_base)


def get_most_relevant_context(user_query):
    query_embedding = embedding_model.encode(user_query)
    similarities = np.dot(kb_embeddings, query_embedding) / (
                np.linalg.norm(kb_embeddings, axis=1) * np.linalg.norm(query_embedding))
    best_match_idx = np.argmax(similarities)

    # 打印出算出来的相似度分值，让你看清算法在后台是怎么“思考”的
    print(f"📊 算法匹配分值: {similarities[best_match_idx]:.4f}")
    return knowledge_base[best_match_idx]


def ask_local_llm_with_rag(query):
    # 检索背景资料
    context = get_most_relevant_context(query)
    print(f"🔍 知识库检索成功！为您捞出的参考资料是:\n👉 \"{context}\"")

    # ==========================================
    # 2. 终极强约束 Prompt 升级（针对小模型打补丁，逼它老实）
    # ==========================================
    prompt = f"""你是一个极度严谨的知识库答问助手。请严格遵守以下三大纪律：
1. 只能根据给出的【参考资料】回答【用户问题】。
2. 如果【参考资料】中没有提到与【用户问题】核心相关的具体答案，你必须严格回答：“知识库中未查到该信息”，绝对不能用你脑子里的通用知识进行扩展回答或瞎编！
3. 严禁说任何废话。

【参考资料】: {context}

【用户问题】: {query}
"""

    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0  # 🌟 将大模型随机性直接降为0，防止它胡思乱想和自我发挥
        }
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        return f"发生错误: {e}"


if __name__ == "__main__":
    # 工业级展示：展示精准检索与大模型严格遵循上下文的能力
    user_question = "物联网传感器温度超过60度会怎样？"  # 这个问题参考资料里有明确答案
    print(f"\n🙋 用户提问: {user_question}")

    ai_answer = ask_local_llm_with_rag(user_question)
    print(f"\n🤖 大模型结合知识库的终极回答:\n{ai_answer}")