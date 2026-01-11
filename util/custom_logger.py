import logging
import os
import sys
from datetime import datetime

class StreamToLogger:
    """
    将 sys.stdout/stderr 重定向到 Logger 的辅助类
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class ExperimentLogger:
    """
    自定义实验日志工具
    功能：
    1. 同时输出到控制台 (Console) 和日志文件 (File)
    2. 自动按时间戳生成日志文件名，避免覆盖
    3. 格式化输出：[时间] [级别] 消息
    4. 自动捕获 print() 输出到日志文件
    """
    def __init__(self, log_dir="../logs", experiment_name="experiment"):
        # 确保日志目录存在
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 保存原始句柄以防循环重定向
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.log"
        self.log_filepath = os.path.join(self.log_dir, filename)

        # 初始化 Logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [] # 清除旧的 handlers 防止重复打印

        # 1. 文件处理器 (File Handler)
        file_handler = logging.FileHandler(self.log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 2. 控制台处理器 (Stream Handler) - 始终使用原始 stdout
        console_handler = logging.StreamHandler(self.original_stdout)
        console_handler.setLevel(logging.INFO)
        
        # 3. 设置格式
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 4. 重定向标准输出和错误输出
        sys.stdout = StreamToLogger(self.logger, logging.INFO)
        sys.stderr = StreamToLogger(self.logger, logging.ERROR)
        
        self.info(f"🚀 日志系统初始化完成。日志文件路径: {os.path.abspath(self.log_filepath)}")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
    
    def get_log_path(self):
        return self.log_filepath
