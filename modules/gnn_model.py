import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import config
import traceback

class GNNModel(torch.nn.Module):
    """简化的图神经网络模型（CPU版本）"""
    def __init__(self, num_nodes, embedding_dim=16, hidden_dim=32):
        super(GNNModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, edge_index):
        # 获取节点嵌入
        embeddings = self.embedding(torch.arange(self.embedding.num_embeddings))
        
        # 消息传递（简化版）
        messages = torch.zeros_like(embeddings)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            messages[dst] += embeddings[src]
        
        # 聚合消息
        aggregated = F.relu(self.conv1(messages))
        updated = F.relu(self.conv2(aggregated))
        
        return embeddings + updated

def build_graph_data(combined_df):
    """从DataFrame构建图数据（不使用PyG）"""
    # 创建用户到索引的映射
    users = pd.unique(pd.concat([combined_df['发言者'], combined_df['响应者']]))
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    
    # 构建边索引
    edge_index = []
    for _, row in combined_df.iterrows():
        source = user_to_idx[row['发言者']]
        target = user_to_idx[row['响应者']]
        edge_index.append([source, target])
    
    # 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return edge_index, user_to_idx, len(users)

def train_gnn_model(edge_index, num_nodes, epochs=100):
    """训练简化的GNN模型"""
    model = GNNModel(num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"{datetime.now()} - 开始训练GNN模型（CPU模式）...")
    print(f"节点数: {num_nodes}, 边数: {edge_index.size(1)}")
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # 前向传播
        embeddings = model(edge_index)
        
        # 计算损失 - 重建邻接矩阵
        adj_pred = torch.sigmoid(torch.mm(embeddings, embeddings.t()))
        adj_true = torch.zeros(num_nodes, num_nodes)
        adj_true[edge_index[0], edge_index[1]] = 1.0
        
        loss = F.binary_cross_entropy(adj_pred, adj_true)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
    
    print(f"{datetime.now()} - GNN模型训练完成")
    return model

def enhance_with_gnn(combined_df):
    """使用GNN增强关系分析（CPU版本）"""
    try:
        # 构建图数据
        edge_index, user_to_idx, num_nodes = build_graph_data(combined_df)
        
        # 训练GNN模型
        model = train_gnn_model(edge_index, num_nodes)
        
        # 获取节点嵌入
        model.eval()
        with torch.no_grad():
            embeddings = model(edge_index).numpy()
        
        # 计算节点相似度（关系强度）
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim_matrix = cosine_similarity(embeddings)
        
        # 收集关系强度
        relation_strengths = {}
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if cos_sim_matrix[i, j] > 0.3:  # 相似度阈值
                    user_i = list(user_to_idx.keys())[i]
                    user_j = list(user_to_idx.keys())[j]
                    relation_strengths[(user_i, user_j)] = cos_sim_matrix[i, j]
                    relation_strengths[(user_j, user_i)] = cos_sim_matrix[i, j]
        
        # 将GNN结果整合到现有数据
        def get_gnn_strength(speaker, responder):
            key1 = (speaker, responder)
            key2 = (responder, speaker)
            return max(relation_strengths.get(key1, 0), relation_strengths.get(key2, 0))
        
        combined_df['GNN强度'] = combined_df.apply(
            lambda row: get_gnn_strength(row['发言者'], row['响应者']), axis=1
        )
        
        # 结合现有暗恋参数和GNN强度
        scaler = MinMaxScaler()
        combined_df['暗恋参数_GNN'] = scaler.fit_transform(
            combined_df[['暗恋参数', 'GNN强度']].mean(axis=1).values.reshape(-1, 1)
        ) * 100
        
        return combined_df
    except Exception as e:
        print(f"GNN增强分析失败: {str(e)}")
        traceback.print_exc()
        return combined_df  # 返回原始数据