# data_loader 模块
# 在 modules/data_loader.py 开头添加
import os
import sys
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import pandas as pd
import re
import os
from config import DATA_PATH

def load_data(file_path=DATA_PATH):
    """加载并预处理数据"""
    # 如果未指定路径，使用配置文件中的路径
    if not file_path:
        file_path = DATA_PATH
    
    # 确保文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 加载数据
    df = pd.read_csv(file_path, parse_dates=['时间'])
    
    # 清理昵称
    df['昵称'] = df['昵称'].apply(lambda x: re.sub(r'[^\w\u4e00-\u9fff\s.,!?]', '', str(x)).strip())
    
    print(f"加载完成: {len(df)}条消息，{df['昵称'].nunique()}位用户")
    return df
def preprocess_messages(df):
    """增强消息预处理：识别消息组和@符号"""
    # 1. 创建消息组ID - 同一用户连续发送的消息视为同一话题
    df['消息组ID'] = (df['昵称'] != df['昵称'].shift(1)).cumsum()
    
    # 2. 标记包含@符号的消息并提取目标用户
    df['包含@'] = df['消息内容'].str.contains(r'@(\w+)')
    df['@目标用户'] = df['消息内容'].str.extract(r'@(\w+)', expand=False)
    
    # 3. 对于包含@的消息，将其关联到目标用户的最新消息组
    # 创建用户最后消息组的映射
    last_group_map = {}
    for idx, row in df.iterrows():
        user = row['昵称']
        if not pd.isna(row['@目标用户']):
            target = row['@目标用户']
            # 如果目标用户有最近的消息组，关联到该组
            if target in last_group_map:
                df.at[idx, '关联组ID'] = last_group_map[target]
        # 更新用户最后活动消息组
        last_group_map[user] = row['消息组ID']
    
    return df