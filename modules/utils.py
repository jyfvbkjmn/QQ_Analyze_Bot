import re  # 添加re导入
import time
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, total):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.processed = 0
        self.total = total
        self.skipped = 0
        self.errors = 0
        
    def update(self, processed, skipped=0, errors=0):
        self.processed = processed
        self.skipped += skipped
        self.errors += errors
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        speed = processed / max(1, elapsed)
        
        if current_time - self.last_update > 5 or processed % 1000 == 0:
            self.last_update = current_time
            percent = self.processed / self.total * 100
            remaining = (self.total - self.processed) / max(1, speed)
            
            print(f"{datetime.now()} - 进度: {percent:.1f}% | "
                  f"已处理: {self.processed}/{self.total} | "
                  f"速度: {speed:.1f}条/秒 | "
                  f"耗时: {elapsed:.1f}s | "
                  f"剩余: {remaining:.1f}s | "
                  f"跳过: {self.skipped} | "
                  f"错误: {self.errors}")
    
    def final_report(self):
        total_time = time.time() - self.start_time
        avg_speed = self.processed / max(1, total_time)
        
        print(f"\n{datetime.now()} - 分析完成!")
        print(f"总耗时: {total_time:.1f}秒")
        print(f"平均速度: {avg_speed:.1f}条/秒")
        print(f"成功分析: {self.processed - self.skipped - self.errors}条")
        print(f"跳过分析: {self.skipped}条")
        print(f"分析错误: {self.errors}条")

def preprocess_text(text):
    """高效文本预处理"""
    text = str(text)
    if '[CQ:image' in text: return '[图片]'
    if '[CQ:face' in text: return '[表情]'
    if '[CQ:json' in text: return '[分享]'
    if '[CQ:forward' in text: return '[转发]'
    return re.sub(r'\[CQ:[^\]]+\]', '', text).strip()  # 使用re模块