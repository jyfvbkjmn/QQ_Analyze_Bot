QQ群聊关系分析系统
https://img.shields.io/badge/license-MIT-blue.svg

https://img.shields.io/badge/python-3.8%2B-blue

https://img.shields.io/badge/docker-ready-blue

基于图神经网络(GNN)的QQ群聊关系分析系统，能够从聊天记录中挖掘用户关系、识别关键影响者，并提供专业可视化报告。本系统结合自然语言处理和图神经网络技术，为社群运营提供数据支持。

✨ 核心功能
•
​情感分析​：自动识别消息情感倾向和强度

•
​关系强度计算​：量化用户间互动关系强度

•
​GNN增强分析​：使用图神经网络挖掘深层关系

•
​多维度可视化​：7种专业图表展示分析结果

•
​一键部署​：支持Docker容器化部署


🚀 快速开始
前提条件
•
Python 3.8+

•
Docker (可选)

•
消息记录.csv文件

安装步骤
bash
复制
# 克隆仓库
git clone https://github.com/yourusername/QQ_Analyze_Bot.git
cd QQ_Analyze_Bot

# 安装依赖
pip install -r requirements.txt

# 准备数据（将您的聊天记录放入data目录）
cp your_chat_data.csv data/消息记录.csv
运行分析
bash
复制
# 运行完整分析流程
python main.py

# 使用Docker运行
docker-compose up -d
⚙️ 配置说明
编辑 config.py自定义分析参数：

python
下载
复制
运行
# 分析窗口设置
WINDOWS = {
    'response': 300,   # 响应时间窗口(秒)
    'sentiment': 600,   # 情感分析窗口
    'interaction': 900  # 互动频率窗口
}

# GNN配置
GNN_CONFIG = {
    'use_gpu': False,  # 是否使用GPU加速
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.01,
    'epochs': 200
}

# 可视化参数
TOP_N_USERS = 20       # 显示的用户数量
SENTIMENT_THRESHOLD = 40  # 情感阈值
📈 示例报告
关系网络图
docs/network_example.png

情感分布雷达图
docs/sentiment_radar_example.png

GNN增强分析
docs/gnn_distribution_example.png

🧩 模块说明
模块

功能描述

data_loader.py

数据加载与预处理

sentiment_analyzer.py

中文情感分析

core_metrics.py

核心指标计算

affection_metrics.py

关系强度建模

gnn_model.py

GNN增强分析

visualization.py

可视化报告生成

🧪 测试数据
使用示例数据测试系统：

bash
复制
# 使用测试数据
python test.py --sample
测试数据包含：

•
63位用户

•
19,000+条消息

•
模拟多种互动场景

🤝 贡献指南
欢迎贡献代码！请遵循以下流程：

1.
Fork 项目仓库

2.
创建新分支 (git checkout -b feature/your-feature)

3.
提交更改 (git commit -am 'Add some feature')

4.
推送到分支 (git push origin feature/your-feature)

5.
创建 Pull Request

📚 文档目录
•
数据准备指南

•
算法原理说明

•
API接口文档

•
高级配置选项

📄 许可证
本项目采用 MIT 许可证

✉️ 联系信息
如有任何问题，请联系：

•
项目维护者：Your Name

•
邮箱：your.email@example.com

•
项目主页：https://github.com/yourusername/QQ_Analyze_Bot

​让数据讲述社群故事​ - 通过深度分析挖掘QQ群聊中的隐藏关系模式，为社群运营提供科学依据。