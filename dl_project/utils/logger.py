import logging
import os
import time
from datetime import datetime


def setup_logger(save_dir, name='training'):
    """设置日志记录器"""
    # 创建日志目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    # 配置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        self.start_time = time.time()

        with open(self.log_file, 'w') as f:
            f.write(f"训练开始时间: {datetime.now()}\n")
            f.write("=" * 50 + "\n")

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

        print(log_message)

    def log_metrics(self, epoch, train_loss, train_acc, test_loss, test_acc, f1):
        """记录评估指标"""
        message = (f"Epoch {epoch:3d} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                   f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
                   f"F1: {f1:.4f}")
        self.log(message)

    def finalize(self):
        """结束训练日志"""
        end_time = time.time()
        duration = end_time - self.start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        with open(self.log_file, 'a') as f:
            f.write("=" * 50 + "\n")
            f.write(f"训练结束时间: {datetime.now()}\n")
            f.write(f"总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}\n")