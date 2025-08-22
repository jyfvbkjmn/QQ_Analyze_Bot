import os
from pathlib import Path

# 获取项目根目录（当前文件所在目录）
ROOT_DIR = Path(__file__).resolve().parent

# 使用绝对路径定义数据文件路径
DATA_PATH = "/app/data/消息记录.csv"  # 容器内路径
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'results'))

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 分析参数配置
WINDOWS = {
    'response': 300,    # 响应时间窗口（秒）
    'sentiment': 600,   # 情感分析窗口（秒）
    'interaction': 900  # 互动频率窗口（秒）
}

# 权重配置
AFFECTION_WEIGHTS = {
    '互动频率': 0.3,
    '响应速度': 0.4,
    '情感倾向': 0.3
}

# 可视化参数
TOP_N_USERS = 20       # 显示的用户数量
SENTIMENT_THRESHOLD = 40  # 情感阈值
SNAKEY_THRESHOLD = 30    # 桑基图阈值
NETWORK_THRESHOLD = 0.3  # 网络图关系强度阈值（新增）

# GNN配置
GNN_CONFIG = {
    'model_type': 'GAT',  # GCN, GAT, GraphSAGE
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.005,
    'epochs': 100,
    'use_gpu': False,
    'gnn_threshold': 0.6  # 关系强度阈值
}
# 在 config.py 中添加
COLUMN_NAMES = {
    'interaction_freq': '互动频率',
    'response_speed': '响应速度',
    'sentiment_tendency': '情感倾向',
    'sentiment_intensity': '情感强度',
    'affection_param': '暗恋参数'
}