import re
import time
import gc
import jieba
import numpy as np
from snownlp import SnowNLP
from datetime import datetime  # 添加datetime导入
from .utils import PerformanceMonitor, preprocess_text

# 带缓存的情感分析
sentiment_cache = {}

def analyze_sentiment(text):
    try:
        clean_text = preprocess_text(text)
        if clean_text in ('[图片]', '[表情]', '[分享]', '[转发]') or len(clean_text) < 3:
            return 0, "skip"
        if clean_text in sentiment_cache:
            return sentiment_cache[clean_text], "cached"
        s = SnowNLP(clean_text)
        score = (s.sentiments - 0.5) * 2
        sentiment_cache[clean_text] = score
        return score, "success"
    except:
        return 0, "error"

# 批量处理函数
def process_batch(batch):
    results = []
    skipped = 0
    errors = 0
    for text in batch:
        score, status = analyze_sentiment(text)
        results.append(score)
        if status == "skip": skipped += 1
        elif status == "error": errors += 1
    return results, skipped, errors

# 主分析函数
def analyze_messages(df):
    """分析消息情感"""
    print(f"{datetime.now()} - 开始情感分析...")
    
    # 预加载模型
    print(f"{datetime.now()} - 预加载SnowNLP模型...")
    jieba.initialize()
    s = SnowNLP("初始化模型")
    _ = s.sentiments
    
    texts = df['消息内容'].tolist()
    total = len(texts)
    monitor = PerformanceMonitor(total)
    chunk_size = 500
    sentiment_scores = []
    
    for i in range(0, total, chunk_size):
        batch = texts[i:i+chunk_size]
        batch_results, skipped, errors = process_batch(batch)
        sentiment_scores.extend(batch_results)
        processed = min(i + chunk_size, total)
        monitor.update(processed, skipped, errors)
        del batch, batch_results
        gc.collect()
    
    monitor.final_report()
    
    # 添加情感得分列
    df['情感得分'] = sentiment_scores
    return df