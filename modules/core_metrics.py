# core_metrics.py
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from .utils import PerformanceMonitor
import bisect  # 用于二分查找
import numba  # 用于高性能计算
import torch
from torch_geometric.data import Data
import networkx as nx

def preprocess_messages(df):
    """增强消息预处理：识别消息组和@符号 - 优化性能版本"""
    # 1. 创建消息组ID - 同一用户连续发送的消息视为同一话题
    df['消息组ID'] = (df['昵称'] != df['昵称'].shift(1)).astype(int).cumsum()
    
    # 2. 标记包含@符号的消息并提取目标用户
    # 使用快速正则表达式检查
    df['包含@'] = df['消息内容'].str.contains(r'@\w+', na=False)
    df['@目标用户'] = df['消息内容'].str.extract(r'@(\w+)', expand=False)
    
    # 3. 创建用户最后消息组的映射 - 使用字典优化
    last_group_map = {}
    df['关联组ID'] = -1  # 使用-1表示无关联，比NaN更快
    
    # 4. 使用向量化操作优化关联组ID设置
    target_mask = df['包含@'] & df['@目标用户'].notna()
    target_users = df.loc[target_mask, '@目标用户']
    
    # 使用字典快速查找最后消息组
    group_mapping = df.groupby('昵称')['消息组ID'].last().to_dict()
    
    # 设置关联组ID
    df.loc[target_mask, '关联组ID'] = target_users.map(group_mapping).fillna(-1).astype(int)
    
    # 5. 添加时间戳 - 使用int64提高性能
    df['时间戳'] = df['时间'].astype(np.int64) // 10**9
    
    return df

@numba.jit(nopython=True)
def find_first_response(timestamps, start_idx, end_idx, speaker_idx, responders_idx, threshold):
    """使用Numba优化的函数查找第一个响应"""
    for i in range(start_idx, end_idx):
        if responders_idx[i] != speaker_idx and timestamps[i] - timestamps[start_idx] <= threshold:
            return i
    return -1

def calculate_core_metrics(df, windows):
    """优化核心指标计算 - 高性能版本"""
    print(f"{datetime.now()} - 开始计算核心指标...")
    
    # 应用增强预处理
    df = preprocess_messages(df)
    
    # 创建用户ID映射 - 使用整数索引比字符串更快
    users = df['昵称'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    df['用户ID'] = df['昵称'].map(user_to_idx)
    
    # 按时间排序并创建索引
    df = df.sort_values('时间戳').reset_index(drop=True)
    timestamps = df['时间戳'].values
    user_ids = df['用户ID'].values
    emotion_scores = df['情感得分'].values if '情感得分' in df.columns else np.zeros(len(df))
    group_ids = df['消息组ID'].values
    linked_group_ids = df['关联组ID'].values
    
    # 准备结果存储 - 使用列表预分配提高性能
    max_possible_pairs = min(500000, len(df) * 10)  # 合理的最大预期对数
    response_data = []
    sentiment_data = []
    interaction_data = []
    
    # 创建消息组索引
    group_df = df.groupby('消息组ID').agg({
        '时间戳': 'min',
        '用户ID': 'first',
        '@目标用户': 'first',
        '关联组ID': 'first'
    }).reset_index()
    group_df = group_df.rename(columns={'用户ID': '发起者ID'})
    
    # 排序时间戳数组用于二分查找
    sorted_timestamps = np.sort(timestamps)
    
    # 进度监控
    total_groups = len(group_df)
    monitor = PerformanceMonitor(total_groups)
    
    # 主处理循环 - 按组处理
    for idx, group_row in group_df.iterrows():
        speaker_id = group_row['发起者ID']
        timestamp = group_row['时间戳']
        linked_group = group_row['关联组ID']
        
        # 使用二分查找快速定位时间窗口
        start_idx = bisect.bisect_left(sorted_timestamps, timestamp)
        end_idx = bisect.bisect_right(sorted_timestamps, timestamp + max(windows.values()))
        
        # 如果时间窗口内没有消息，跳过
        if start_idx >= len(timestamps) or end_idx <= start_idx:
            monitor.update(idx + 1)
            continue
            
        # 查找第一个响应 - 使用Numba优化
        first_response_idx = find_first_response(
            timestamps, start_idx, end_idx, 
            speaker_id, user_ids, windows['response']
        )
        
        # 如果没有找到响应，跳过
        if first_response_idx == -1:
            monitor.update(idx + 1)
            continue
            
        responder_id = user_ids[first_response_idx]
        response_time = timestamps[first_response_idx] - timestamp
        
        # 添加到响应数据
        response_data.append({
            '发言者': users[speaker_id],
            '响应者': users[responder_id],
            '响应时间(秒)': response_time
        })
        
        # 添加到情感数据
        if response_time <= windows['sentiment']:
            sentiment_data.append({
                '发言者': users[speaker_id],
                '响应者': users[responder_id],
                '情感得分': emotion_scores[first_response_idx]
            })
        
        # 添加到互动数据
        interaction_data.append({
            '发言者': users[speaker_id],
            '响应者': users[responder_id]
        })
        
        # 更新进度
        if (idx + 1) % 100 == 0:
            monitor.update(idx + 1)
    
    # 最终报告
    monitor.final_report()
    
    # 返回预处理后的DataFrame
    return response_data, sentiment_data, interaction_data, df

def build_graph_data(df):
    """构建图数据结构"""
    # 创建用户ID映射
    users = df['昵称'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    
    # 构建边列表
    edges = []
    edge_attrs = []
    
    # 遍历所有消息组
    for group_id, group_data in df.groupby('消息组ID'):
        speaker = group_data['昵称'].iloc[0]
        speaker_idx = user_to_idx[speaker]
        
        # 找到响应者
        responders = group_data[group_data['昵称'] != speaker]['昵称'].unique()
        for responder in responders:
            responder_idx = user_to_idx[responder]
            
            # 计算边属性
            response_times = group_data[group_data['昵称'] == responder]['时间戳'] - group_data.iloc[0]['时间戳']
            avg_response_time = response_times.mean() if not response_times.empty else 0
            
            # 添加边
            edges.append([speaker_idx, responder_idx])
            edge_attrs.append({
                'avg_response_time': avg_response_time,
                'interaction_count': len(response_times)
            })
    
    # 构建PyG图数据
    if edges:  # 确保边不为空
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # 创建空图作为回退
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 节点特征（使用用户行为统计）
    node_features = []
    for user in users:
        user_data = df[df['昵称'] == user]
        # 简单特征：消息数量、响应时间均值、情感均值
        features = [
            len(user_data),
            user_data['响应时间(秒)'].mean() if '响应时间(秒)' in user_data.columns else 0,
            user_data['情感得分'].mean() if '情感得分' in user_data.columns else 0
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index), edge_attrs, user_to_idx