import sys
import json
import requests
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 🚀 1. 创建后台大模型处理线程，防止界面卡死
class LLMWorkerThread(QThread):
    # 定义两个信号：一个用于传输实时日志，一个用于传输结束通知
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int) # 参数：成功条数，失败条数

    def __init__(self, input_file, output_file):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.is_running = True

    def get_ai_label(self, question, answer):
        prompt = f"""你是一个专业的数据标注员。请阅读以下【用户问题】和【专家回答】，并将其精准归类到以下三个标签之一：
- [网络设备配置] (涉及交换机、路由器、IP配置等)
- [物联网硬件] (涉及传感器、嵌入式、硬件芯片等)
- [Linux操作系统] (涉及Linux命令、内核、系统操作等)

【用户问题】: {question}
【专家回答】: {answer}

注意：你只能输出标签名字，例如：[网络设备配置]，绝对不要输出任何其他解释、标点或多余的话！"""

        data = {
            "model": "qwen2.5:1.5b", 
            "prompt": prompt,
            "stream": True # 开启流式
        }

        full_response = ""
        try:
            response = requests.post(OLLAMA_API_URL, json=data, stream=True, timeout=30)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if not self.is_running: # 支持中途取消
                        return "[用户中止]"
                    if line:
                        decoded_line = line.decode('utf-8')
                        json_data = json.loads(decoded_line)
                        full_response += json_data.get("response", "")
                        if json_data.get("done", False):
                            break
                return full_response.strip()
            else:
                return "[状态码异常]"
        except Exception as e:
            return f"[异常: {str(e)}]"

    def run(self):
        processed_count = 0
        skipped_count = 0
        
        try:
            # 模拟第一步：读取与规则清洗
            with open(self.input_file, 'r', encoding='utf-8') as f:
                raw_dialogues = f.read().split('===')
            
            with open(self.output_file, 'w', encoding='utf-8') as f_out:
                for dialogue in raw_dialogues:
                    if not self.is_running:
                        break
                        
                    lines = dialogue.strip().split('\n')
                    if len(lines) < 2:
                        continue
                        
                    user_content = ""
                    assistant_content = ""
                    for line in lines:
                        if line.startswith("用户:"):
                            user_content = line.replace("用户:", "").strip()
                        elif line.startswith("助手:"):
                            assistant_content = line.replace("助手:", "").strip()
                    
                    if len(user_content) < 5 or len(assistant_content) < 5:
                        skipped_count += 1
                        continue
                    
                    # 模拟第二步：大模型在线打标签
                    self.log_signal.emit(f"⏳ 正在请求大模型分析问题: '{user_content[:10]}...'")
                    label = self.get_ai_label(user_content, assistant_content)
                    
                    data_structure = {
                        "instruction": "你是一个专业的物联网与网络技术专家，请回答用户的问题。",
                        "input": user_content,
                        "output": assistant_content,
                        "category": label
                    }
                    
                    f_out.write(json.dumps(data_structure, ensure_ascii=False) + '\n')
                    processed_count += 1
                    # 发送流式成功日志给主界面
                    self.log_signal.emit(f"✅ AI 标注成功 -> 标签: {label}\n" + "-"*40)
                    
            self.finished_signal.emit(processed_count, skipped_count)
            
        except Exception as e:
            self.log_signal.emit(f"❌ 运行中发生致命错误: {str(e)}")
            self.finished_signal.emit(0, 0)

    def stop(self):
        self.is_running = False

# 💻 2. 创建用户交互主界面
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("大模型微调数据集自动化清洗与标注系统 v1.0")
        self.resize(700, 500)
        
        # 整体布局
        layout = QVBoxLayout()
        
        # 文件选择区域
        file_layout = QHBoxLayout()
        self.btn_select_in = QPushButton("选择原始文本 (.txt)")
        self.btn_select_in.clicked.connect(self.select_input_file)
        self.lbl_in_path = QLabel("未选择文件")
        file_layout.addWidget(self.btn_select_in)
        file_layout.addWidget(self.lbl_in_path, 1)
        layout.addLayout(file_layout)
        
        file_layout_out = QHBoxLayout()
        self.btn_select_out = QPushButton("设置导出路径 (.jsonl)")
        self.btn_select_out.clicked.connect(self.select_output_file)
        self.lbl_out_path = QLabel("未设置路径")
        file_layout_out.addWidget(self.btn_select_out)
        file_layout_out.addWidget(self.lbl_out_path, 1)
        layout.addLayout(file_layout_out)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 开始全自动清洗与标注")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_start.clicked.connect(self.start_process)
        
        self.btn_stop = QPushButton("🛑 停止终止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.btn_stop.clicked.connect(self.stop_process)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        layout.addLayout(control_layout)
        
        # 实时日志监视器
        layout.addWidget(QLabel("📊 实时处理状态监控控制台:"))
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.log_viewer)
        
        self.setLayout(layout)
        
        # 默认自动锁定到昨天的示例文件（方便测试）
        self.input_file_path = "raw_data.txt"
        self.output_file_path = "labeled_data_pyqt.jsonl"
        self.lbl_in_path.setText(self.input_file_path)
        self.lbl_out_path.setText(self.output_file_path)

    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择原始数据文件", "", "Text Files (*.txt)")
        if file_path:
            self.input_file_path = file_path
            self.lbl_in_path.setText(file_path)

    def select_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "设置输出数据集路径", "", "JSONL Files (*.jsonl)")
        if file_path:
            self.output_file_path = file_path
            self.lbl_out_path.setText(file_path)

    def start_process(self):
        self.log_viewer.clear()
        self.log_viewer.append("🔄 正在初始化流水线，检查本地 Ollama 状态...")
        
        # 启动后台工作线程
        self.worker = LLMWorkerThread(self.input_file_path, self.output_file_path)
        # 连接信号与槽函数
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.process_finished)
        
        self.worker.start()
        
        # 切换按钮状态
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def update_log(self, text):
        self.log_viewer.append(text)
        # 自动滚动到底部
        self.log_viewer.moveCursor(self.log_viewer.textCursor().End)

    def stop_process(self):
        if self.worker:
            self.worker.stop()
            self.log_viewer.append("\n⚠️ 用户触发强制中止，正在安全收尾...")

    def process_finished(self, success, skipped):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "处理完成", f"🏆 自动化处理流运行结束！\n\n成功标注并导出: {success} 条数据\n过滤低质量数据: {skipped} 条")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())