import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import config

def calculate_dynamic_weights(df):
    """根据数据分布动态计算权重"""
    weights = {
        '互动频率': 0.3,
        '响应速度': 0.4,
        '情感倾向': 0.3
    }
    
    # 防止除零错误
    if len(df) < 2:
        return weights
    
    # 计算指标变异系数
    cv_interaction = df['互动次数'].std() / (df['互动次数'].mean() + 1e-5)
    cv_response = df['响应速度得分'].std() / (df['响应速度得分'].mean() + 1e-5)
    cv_sentiment = df['情感倾向得分'].std() / (df['情感倾向得分'].mean() + 1e-5)
    
    # 如果所有值都相同，变异系数为0
    cv_interaction = 0.1 if np.isnan(cv_interaction) else cv_interaction
    cv_response = 0.1 if np.isnan(cv_response) else cv_response
    cv_sentiment = 0.1 if np.isnan(cv_sentiment) else cv_sentiment
    
    # 根据变异系数调整权重 - 变异小的指标权重增加
    total_cv = cv_interaction + cv_response + cv_sentiment
    if total_cv > 0:
        weights['互动频率'] = 0.3 * (1 - cv_interaction/total_cv*0.5)
        weights['响应速度'] = 0.4 * (1 - cv_response/total_cv*0.5)
        weights['情感倾向'] = 0.3 * (1 - cv_sentiment/total_cv*0.5)
    
    # 归一化权重
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights

def build_affection_metrics(response_data, sentiment_data, interaction_data, preprocessed_df, weights):
    """构建暗恋参数"""
    print(f"{datetime.now()} - 构建暗恋参数...")
    
    # 创建DataFrame
    response_df = pd.DataFrame(response_data) if response_data else pd.DataFrame(columns=['发言者', '响应者', '响应时间(秒)'])
    sentiment_df = pd.DataFrame(sentiment_data) if sentiment_data else pd.DataFrame(columns=['发言者', '响应者', '情感得分'])
    interaction_df = pd.DataFrame(interaction_data) if interaction_data else pd.DataFrame(columns=['发言者', '响应者'])
    
    if not response_df.empty:
        response_df['响应速度得分'] = 1 / (response_df['响应时间(秒)'] + 1)
    else:
        response_df = pd.DataFrame(columns=['发言者', '响应者', '响应时间(秒)', '响应速度得分'])
    
    if not sentiment_df.empty:
        sentiment_agg = sentiment_df.groupby(['发言者', '响应者'])['情感得分'].mean().reset_index()
        sentiment_agg.rename(columns={'情感得分': '情感倾向得分'}, inplace=True)
    else:
        sentiment_agg = pd.DataFrame(columns=['发言者', '响应者', '情感倾向得分'])
    
    if not interaction_df.empty:
        interaction_freq = interaction_df.groupby(['发言者', '响应者']).size().reset_index(name='互动次数')
    else:
        interaction_freq = pd.DataFrame(columns=['发言者', '响应者', '互动次数'])
    
    # 合并所有指标
    combined_df = interaction_freq.merge(
        response_df[['发言者', '响应者', '响应速度得分']],
        on=['发言者', '响应者'],
        how='left'
    ).merge(
        sentiment_agg,
        on=['发言者', '响应者'],
        how='left'
    )
    
    # 填充缺失值
    fill_values = {
        '响应速度得分': 0.01,
        '情感倾向得分': 0,
        '互动次数': 0
    }
    combined_df.fillna(fill_values, inplace=True)
    
    # 归一化指标
    scaler = MinMaxScaler()
    for col in ['互动次数', '响应速度得分', '情感倾向得分']:
        if col in combined_df.columns:
            combined_df[col + '_norm'] = scaler.fit_transform(combined_df[[col]])
        else:
            combined_df[col + '_norm'] = 0
    
    # 计算动态权重
    dynamic_weights = calculate_dynamic_weights(combined_df)
    
    # 分层加权计算
    # 第一层：基础互动指标
    interaction_score = dynamic_weights['互动频率'] * combined_df['互动次数_norm']
    
    # 第二层：响应与情感
    response_score = dynamic_weights['响应速度'] * combined_df['响应速度得分_norm']
    sentiment_score = dynamic_weights['情感倾向'] * combined_df['情感倾向得分_norm']
    
    # 几何平均与算术平均结合
    communication_score = np.sqrt(response_score * sentiment_score) * 0.7 + (response_score + sentiment_score) * 0.3
    
    # 显式互动处理
    if '@目标用户' in preprocessed_df.columns:
        # 获取显式互动计数
        explicit_interactions = preprocessed_df[preprocessed_df['包含@']].groupby(
            ['昵称', '@目标用户']).size().reset_index(name='显式互动')
        explicit_interactions.rename(columns={'昵称': '发言者', '@目标用户': '响应者'}, inplace=True)
        
        # 合并显式互动计数
        combined_df = combined_df.merge(
            explicit_interactions,
            on=['发言者', '响应者'],
            how='left'
        ).fillna({'显式互动': 0})
        
        # 使用衰减函数：初始互动价值高，但边际效应递减
        max_explicit = combined_df['显式互动'].max()
        if max_explicit > 0:
            combined_df['显式互动_score'] = 1 - np.exp(-0.3 * combined_df['显式互动'] / max_explicit * 5)
        else:
            combined_df['显式互动_score'] = 0
    else:
        combined_df['显式互动_score'] = 0
    
    # 综合计算暗恋参数
    combined_df['暗恋参数'] = (
        interaction_score * 0.4 + 
        communication_score * 0.5 + 
        combined_df['显式互动_score'] * 0.1
    )
    
    # 归一化暗恋参数到0-100范围
    scaler = MinMaxScaler()
    combined_df['暗恋参数'] = scaler.fit_transform(combined_df[['暗恋参数']]) * 100
    
    # 创建可视化所需的列
    combined_df['互动频率'] = combined_df['互动次数_norm'] * 100
    combined_df['响应速度'] = combined_df['响应速度得分_norm'] * 100
    combined_df['情感倾向'] = combined_df['情感倾向得分_norm'] * 100
    combined_df['情感强度'] = combined_df['情感倾向得分'] * 100  # 用于情感分布雷达图
    
    # 返回结果
    return combined_df