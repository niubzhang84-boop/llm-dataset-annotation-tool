# 大模型微调语料全自动清洗与高可视化标注系统 v1.0

本项目是一款专为大模型（LLM）微调（Fine-tuning）语料准备而设计的全自动数据清洗、过滤与智能分类标注工具。项目底层无缝对接本地开源大模型（Qwen2.5），并采用多线程架构保障高并发下的工业级交互体验。

## ✨ 核心技术亮点

1. **工业级流式传输（Stream）**：采用 Requests Stream 机制，逐行迭代解析大模型返回的文本流碎片，彻底根治了本地大模型长时间推理导致的 `Read timed out` 连接卡死异常。
2. **PyQt5 多线程架构（QThread）**：将高密度的 LLM API 阻塞请求与 UI 渲染线程完全解耦，通过自定义 `pyqtSignal` 机制进行跨线程安全通信，完美解决软件界面“未响应”的交互痛点。
3. **闭环式数据流水线（Pipeline）**：
   * **Rule-based 清洗**：正则切除原始文本中的 Markdown、HTML 乱码标签，通过 Token 长度机制进行低质量文本过滤。
   * **LLM-as-a-judge 标注**：构建结构化 Prompt 强约束本地大模型，全自动将技术问答对分类为 `[物联网硬件]`、`[网络设备配置]`、`[Linux操作系统]`。
   * **一键规整**：自动化输出标准微调格式的 `JSONL` 数据集。

## 🚀 快速开始

### 1. 环境准备
确保本地已安装 [Ollama](https://ollama.com/) 并拉取轻量化大模型：
```bash
ollama run qwen2.5:1.5b
2. 安装依赖
Bash
pip install PyQt5 requests
3. 启动运行
Bash
python ui_labeling_app.py
