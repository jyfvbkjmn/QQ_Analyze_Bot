import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import matplotlib.font_manager as fm
from config import RESULTS_DIR
import traceback
from config import COLUMN_NAMES

def set_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试使用系统字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 查找系统中已安装的中文字体
        chinese_fonts = [f.name for f in fm.fontManager.ttflist 
                         if any(x in f.name.lower() for x in ['song', 'hei', 'yahei'])]
        if chinese_fonts:
            print(f"使用中文字体: {chinese_fonts[0]}")
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
        else:
            print("警告: 未找到中文字体，尝试使用SimHei")
            plt.rcParams['font.sans-serif'] = ['SimHei']
    except Exception as e:
        print(f"字体设置失败: {e}, 使用默认设置")

def plot_heatmap(combined_df, results_dir):
    """生成优化热力图展示用户间互动强度"""
    print(f"{datetime.now()} - 生成优化热力图...")
    
    # 创建用户互动矩阵
    interaction_matrix = combined_df.pivot_table(
        index='发言者', 
        columns='响应者', 
        values='暗恋参数', 
        aggfunc='mean'
    ).fillna(0)
    
    # 选择互动最多的前20个用户
    top_users = interaction_matrix.sum(axis=1).nlargest(20).index
    interaction_matrix = interaction_matrix.loc[top_users, top_users]
    
    # 创建热力图
    plt.figure(figsize=(15, 12))
    sns.heatmap(interaction_matrix, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=.5)
    plt.title('用户间互动强度热力图')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(results_dir, '优化热力图.png')
    plt.savefig(output_path)
    plt.close()
    print(f"优化热力图已保存: {output_path}")

def plot_enhanced_network(combined_df, results_dir, sentiment_threshold=40):
    """优化关系网络图 - 提高可读性和美观度"""
    print(f"{datetime.now()} - 生成优化关系网络图...")
    if combined_df.empty:
        print("无足够数据生成网络图")
        return
        
    filtered = combined_df[combined_df['暗恋参数'] >= sentiment_threshold]
    if filtered.empty:
        filtered = combined_df[combined_df['暗恋参数'] >= combined_df['暗恋参数'].quantile(0.7)]
    
    if filtered.empty:
        print("无足够数据生成网络图")
        return
    
    # 创建有向图
    G = nx.from_pandas_edgelist(
        filtered,
        '发言者', 
        '响应者',
        edge_attr='暗恋参数',
        create_using=nx.DiGraph()
    )
    
    if len(G.nodes) == 0:
        print("无有效节点生成网络图")
        return
    
    # 使用社区检测算法分组节点
    communities = nx.community.greedy_modularity_communities(G.to_undirected())
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    
    # 为每个社区分配颜色
    num_communities = len(communities)
    community_colors = plt.cm.tab20.colors[:num_communities]
    
    # 计算节点位置 - 使用改进的力导向布局
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42, scale=2)
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(24, 20))
    
    # 绘制节点 - 按社区分组
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=comm,
            node_size=[min(800, max(100, G.out_degree(node, weight='weight') * 10)) for node in comm],
            node_color=[community_colors[i]],
            alpha=0.9,
            label=f'社区 {i+1}',
            ax=ax
        )
    
    # 绘制边 - 使用弯曲的边减少交叉
    for u, v, d in G.edges(data=True):
        # 计算边的弯曲度
        rad = 0.2 if community_map.get(u, -1) == community_map.get(v, -1) else 0.4
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u, v)],
            width=max(1, min(5, d['暗恋参数']/20)),
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle=f'arc3,rad={rad}',
            ax=ax
        )
    
    # 智能标签放置 - 只显示重要节点
    # 计算节点重要性（入度+出度）
    node_importance = {node: G.in_degree(node, weight='weight') + G.out_degree(node, weight='weight') for node in G.nodes}
    important_nodes = sorted(node_importance, key=node_importance.get, reverse=True)[:int(len(G.nodes)*0.3)]
    
    # 使用NetworkX的标签偏移功能避免重叠
    label_pos = {}
    for node in important_nodes:
        x, y = pos[node]
        # 根据位置调整标签偏移
        if x < 0.5:
            label_pos[node] = (x - 0.05, y)
        else:
            label_pos[node] = (x + 0.05, y)
    
    nx.draw_networkx_labels(
        G, label_pos,
        labels={node: node for node in important_nodes},
        font_size=12,
        font_weight='bold',
        font_color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
        ax=ax
    )
    
    # 添加图例
    ax.legend(scatterpoints=1, fontsize=12, loc='best')
    
    plt.title('群聊关注关系网络（优化）', fontsize=24)
    plt.axis('off')
    
    # 保存高分辨率图像
    plt.savefig(os.path.join(results_dir, '优化关系网络图.png'), dpi=300, bbox_inches='tight')
    print(f"优化关系网络图已保存")
    
    # 创建交互式HTML版本
    create_interactive_network(G, pos, community_map, community_colors, results_dir)  # 确保调用这个函数

def create_interactive_network(G, pos, community_map, community_colors, results_dir):
    """创建交互式网络图（HTML）"""
    print(f"{datetime.now()} - 生成交互式网络图...")
    
    # 准备节点数据
    nodes = []
    for node in G.nodes:
        nodes.append({
            'id': node,
            'label': node,
            'x': pos[node][0],
            'y': pos[node][1],
            'size': min(30, max(5, G.out_degree(node, weight='weight') / 5)),
            'color': f'rgb({int(community_colors[community_map[node]][0]*255)}, {int(community_colors[community_map[node]][1]*255)}, {int(community_colors[community_map[node]][2]*255)})',
            'community': community_map[node] + 1
        })
    
    # 准备边数据
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'value': d['暗恋参数'],
            'color': 'rgba(100, 100, 100, 0.5)'
        })
    
    # HTML模板
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>交互式关系网络图</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                font-family: 'Microsoft YaHei', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                overflow: hidden;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .controls {{
                margin: 15px 0;
                padding: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .control-group {{
                display: flex;
                align-items: center;
            }}
            .control-label {{
                font-weight: bold;
                margin-right: 10px;
            }}
            .slider {{
                width: 150px;
            }}
            #network {{
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: white;
                overflow: hidden;
            }}
            .node {{
                cursor: pointer;
            }}
            .link {{
                stroke: #999;
                stroke-opacity: 0.6;
            }}
            .tooltip {{
                position: absolute;
                padding: 10px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #ddd;
                border-radius: 5px;
                pointer-events: none;
                font-size: 14px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>交互式关系网络图</h1>
            <p>点击节点可查看详细信息，使用鼠标滚轮缩放，拖拽平移</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <span class="control-label">节点大小:</span>
                <input type="range" min="1" max="10" value="5" class="slider" id="nodeSizeSlider">
            </div>
            <div class="control-group">
                <span class="control-label">连线粗细:</span>
                <input type="range" min="1" max="10" value="3" class="slider" id="linkWidthSlider">
            </div>
            <div class="control-group">
                <span class="control-label">显示标签:</span>
                <input type="checkbox" id="showLabels" checked>
            </div>
            <div class="control-group">
                <span class="control-label">社区筛选:</span>
                <select id="communityFilter">
                    <option value="all">全部社区</option>
                    {community_options}
                </select>
            </div>
        </div>
        
        <div id="network"></div>
        <div class="tooltip" style="display: none;"></div>
        
        <script>
            // 网络数据
            const nodes = {nodes};
            const edges = {edges};
            
            // 设置画布尺寸
            const width = window.innerWidth * 0.9;
            const height = window.innerHeight * 0.7;
            
            // 创建SVG容器
            const svg = d3.select("#network")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // 创建缩放和平移行为
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", zoomed);
            
            svg.call(zoom);
            
            // 创建力导向模拟
            const simulation = d3.forceSimulation(nodes)
                .force("charge", d3.forceManyBody().strength(-100))
                .force("link", d3.forceLink(edges).id(d => d.id).distance(100))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => d.size * 5));
            
            // 创建连线
            const link = svg.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(edges)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", 3)
                .attr("stroke", d => d.color);
            
            // 创建节点
            const node = svg.append("g")
                .attr("class", "nodes")
                .selectAll("circle")
                .data(nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", d => d.size)
                .attr("fill", d => d.color)
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            // 创建标签
            const label = svg.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(nodes)
                .enter().append("text")
                .attr("class", "label")
                .attr("text-anchor", "middle")
                .attr("dy", d => d.size + 10)
                .text(d => d.label)
                .attr("font-size", 12)
                .attr("fill", "#333")
                .attr("pointer-events", "none");
            
            // 创建工具提示
            const tooltip = d3.select(".tooltip");
            
            // 节点鼠标事件
            node.on("mouseover", function(event, d) {{
                // 高亮相关节点和边
                link.attr("stroke-opacity", l => (l.source === d || l.target === d) ? 1 : 0.1);
                node.attr("opacity", n => (n === d || edges.some(e => e.source === n && e.target === d) || edges.some(e => e.source === d && e.target === n)) ? 1 : 0.1);
                label.attr("opacity", n => (n === d || edges.some(e => e.source === n && e.target === d) || edges.some(e => e.source === d && e.target === n)) ? 1 : 0.1);
                
                // 显示工具提示
                tooltip.style("display", "block")
                    .html(`<strong>${{d.label}}</strong><br>
                        社区: ${{d.community}}<br>
                        关注他人: ${{d.outDegree || 0}}<br>
                        被关注: ${{d.inDegree || 0}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 30) + "px");
            }});
            
            node.on("mouseout", function() {{
                // 恢复原始状态
                link.attr("stroke-opacity", 0.6);
                node.attr("opacity", 1);
                label.attr("opacity", 1);
                tooltip.style("display", "none");
            }});
            
            // 缩放函数
            function zoomed(event) {{
                svg.selectAll(".nodes, .links, .labels")
                    .attr("transform", event.transform);
            }}
            
            // 拖拽函数
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            // 更新节点位置
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }});
            
            // 添加控件事件监听
            document.getElementById('nodeSizeSlider').addEventListener('input', function() {{
                const sizeFactor = this.value;
                node.attr("r", d => d.size * sizeFactor);
            }});
            
            document.getElementById('linkWidthSlider').addEventListener('input', function() {{
                link.attr("stroke-width", this.value);
            }});
            
            document.getElementById('showLabels').addEventListener('change', function() {{
                label.style("display", this.checked ? "block" : "none");
            }});
            
            document.getElementById('communityFilter').addEventListener('change', function() {{
                const selectedCommunity = this.value;
                node.style("display", d => 
                    selectedCommunity === "all" || d.community == selectedCommunity ? "block" : "none");
                label.style("display", d => 
                    (selectedCommunity === "all" || d.community == selectedCommunity) && 
                    document.getElementById('showLabels').checked ? "block" : "none");
            }});
        </script>
    </body>
    </html>
    """
    
    # 为每个节点添加入度和出度信息
    for node in nodes:
        node_id = node['id']
        node['inDegree'] = G.in_degree(node_id, weight='weight')
        node['outDegree'] = G.out_degree(node_id, weight='weight')
    
    # 生成社区筛选选项
    community_options = ""
    communities = set(community_map.values())
    for comm in communities:
        community_options += f'<option value="{comm+1}">社区 {comm+1}</option>'
    
    html_content = html_template.format(
        nodes=json.dumps(nodes),
        edges=json.dumps(edges),
        community_options=community_options
    )
    
    with open(os.path.join(results_dir, "交互式关系网络图.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"交互式关系网络图已保存")

def plot_sankey_diagram(combined_df, results_dir, sankey_threshold=30, max_links=100, value_column='暗恋参数', output_name="优化关注关系桑基图.html"):
    """优化桑基图展示关注关系流"""
    print(f"{datetime.now()} - 生成优化桑基图...")
    if combined_df.empty:
        print("无足够数据生成桑基图")
        return
        
    # 聚合数据 - 合并相同发言者-响应者组合的关系
    grouped = combined_df.groupby(['发言者', '响应者'])[value_column].sum().reset_index()
    
    # 过滤弱关系
    filtered = grouped[grouped[value_column] >= sankey_threshold]
    if filtered.empty:
        filtered = grouped[grouped[value_column] >= grouped[value_column].quantile(0.7)]
    
    if filtered.empty:
        print("无足够数据生成桑基图")
        return
    
    # 限制数据量
    if len(filtered) > max_links:
        filtered = filtered.nlargest(max_links, value_column)
        print(f"数据量过大，仅显示前{max_links}个关系")
    
    # 准备节点列表
    all_users = pd.unique(pd.concat([filtered['发言者'], filtered['响应者']]))
    user_index = {user: idx for idx, user in enumerate(all_users)}
    
    # 准备链接数据
    source = [user_index[s] for s in filtered['发言者']]
    target = [user_index[r] for r in filtered['响应者']]
    value = filtered[value_column].tolist()
    
    if not value:
        print("无有效数据生成桑基图")
        return
    
    # 使用更鲜明的配色方案
    node_colors = px.colors.qualitative.Vivid
    link_colors = []
    
    # 修复颜色转换问题
    def parse_color(color_str):
        """安全解析颜色值，支持RGB和十六进制格式"""
        try:
            if color_str.startswith('rgb('):
                # 解析RGB字符串，例如 'rgb(255,0,0)'
                parts = color_str[4:-1].split(',')
                r = int(parts[0].strip())
                g = int(parts[1].strip())
                b = int(parts[2].strip())
                return (r, g, b)
            else:
                # 假设是十六进制字符串
                return px.colors.hex_to_rgb(color_str)
        except Exception:
            # 默认返回蓝色
            return (0, 0, 255)
    
    for i in range(len(source)):
        # 使用源节点的颜色，但增加透明度
        color_idx = source[i] % len(node_colors)
        color_str = node_colors[color_idx]
        
        # 安全解析颜色
        rgb = parse_color(color_str)
        
        # 构建RGBA字符串
        link_colors.append(f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)")
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=1.0),
            label=all_users,
            color=[node_colors[i % len(node_colors)] for i in range(len(all_users))]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            # 添加悬停信息
            hovertemplate='%{source.label} → %{target.label}<br>关注度: %{value}<extra></extra>'
        )
    )])
    
    # 优化布局
    fig.update_layout(
        title_text="优化关注关系桑基图",
        font_size=12,
        title_font_size=20,
        height=700
    )
    
    # 保存为HTML
    output_path = os.path.join(results_dir, output_name)
    fig.write_html(output_path, auto_open=False)
    print(f"优化桑基图已保存: {output_path}")

def plot_user_radar_chart(combined_df, results_dir, top_n=10):
    """用户特征雷达图"""
    print(f"{datetime.now()} - 生成用户特征雷达图...")
    if combined_df.empty:
        print("无足够数据生成雷达图")
        return
        
    user_stats = pd.DataFrame({'用户': pd.unique(combined_df['发言者'])})
    user_stats['关注他人程度'] = user_stats['用户'].map(
        combined_df.groupby('发言者')['暗恋参数'].sum().to_dict()
    )
    user_stats['被关注程度'] = user_stats['用户'].map(
        combined_df.groupby('响应者')['暗恋参数'].sum().to_dict()
    )
    user_stats['互动广度'] = user_stats['用户'].map(
        combined_df.groupby('发言者')['响应者'].nunique().to_dict()
    )
    
    scaler = MinMaxScaler()
    user_stats[['关注他人程度', '被关注程度', '互动广度']] = scaler.fit_transform(
        user_stats[['关注他人程度', '被关注程度', '互动广度']].fillna(0)
    )
    
    user_stats['综合得分'] = user_stats[['关注他人程度', '被关注程度', '互动广度']].mean(axis=1)
    top_users = user_stats.nlargest(top_n, '综合得分')
    
    categories = ['关注他人程度', '被关注程度', '互动广度']
    fig = go.Figure()
    
    for _, row in top_users.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].tolist() + [row[categories][0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=row['用户']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Top {top_n} 用户特征雷达图",
        height=600,
        width=800
    )
    
    fig.write_html(os.path.join(results_dir, "用户特征雷达图.html"))
    print(f"用户特征雷达图已保存")

def plot_sentiment_radar(combined_df, results_dir, sentiment_threshold=40):
    """生成情感分布雷达图"""
    print(f"{datetime.now()} - 生成情感分布雷达图...")
    
    # 检查所需列是否存在
    required_columns = ['情感强度', '情感倾向']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    
    if missing_columns:
        print(f"警告: 缺少必要列 {missing_columns}，无法生成情感分布雷达图")
        return False
    
    # 过滤情感强度高于阈值的记录
    filtered = combined_df[combined_df['情感强度'] >= sentiment_threshold]
    
    if filtered.empty:
        print("无足够情感数据生成雷达图")
        return False
    
    # 聚合情感数据
    sentiment_stats = filtered.groupby('情感倾向').agg({
        '情感强度': 'mean',
        '情感倾向': 'count'  # 使用情感倾向列计数
    }).rename(columns={'情感倾向': '消息数量'}).reset_index()
    
    # 创建雷达图
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=sentiment_stats['情感强度'],
        theta=sentiment_stats['情感倾向'],
        fill='toself',
        name='情感强度'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='情感分布雷达图',
        height=500
    )
    
    # 保存为HTML
    output_path = os.path.join(results_dir, '情感分布雷达图.html')
    fig.write_html(output_path, auto_open=False)
    print(f"情感分布雷达图已保存: {output_path}")
    return True



def visualize_results(combined_df, results_dir, top_n=20, sentiment_threshold=40, sankey_threshold=30, network_threshold=0.3):
    """生成所有可视化结果"""
    print(f"{datetime.now()} - 生成优化可视化结果...")
    
    try:
        # 1. 生成优化热力图
        print(f"{datetime.now()} - 生成优化热力图...")
        if '发言者' in combined_df.columns and '响应者' in combined_df.columns and '暗恋参数' in combined_df.columns:
            plot_heatmap(combined_df, results_dir)
            print("优化热力图已保存")
        else:
            print("缺少必要列，跳过热力图生成")
        
        # 2. 生成优化关系网络图
        print(f"{datetime.now()} - 生成优化关系网络图...")
        if '发言者' in combined_df.columns and '响应者' in combined_df.columns and '暗恋参数' in combined_df.columns:
            plot_enhanced_network(combined_df, results_dir, sentiment_threshold=sentiment_threshold)
            print("优化关系网络图已保存")
        else:
            print("缺少必要列，跳过关系网络图生成")
        
        # 3. 生成关注关系桑基图
        print(f"{datetime.now()} - 生成优化桑基图...")
        if '发言者' in combined_df.columns and '响应者' in combined_df.columns and '暗恋参数' in combined_df.columns:
            plot_sankey_diagram(combined_df, results_dir, sankey_threshold=sankey_threshold)
            print("优化关注关系桑基图已保存")
        else:
            print("缺少必要列，跳过桑基图生成")
        
        # 4. 生成用户特征雷达图
        print(f"{datetime.now()} - 生成用户特征雷达图...")
        user_radar_success = plot_user_radar_chart(combined_df, results_dir, top_n=top_n)
        if user_radar_success:
            print("用户特征雷达图已保存")
        else:
            print("用户特征雷达图未生成")
        
        # 5. 生成情感分布雷达图
        print(f"{datetime.now()} - 生成情感分布雷达图...")
        sentiment_radar_success = plot_sentiment_radar(combined_df, results_dir, sentiment_threshold=sentiment_threshold)
        if sentiment_radar_success:
            print("情感分布雷达图已保存")
        else:
            print("情感分布雷达图未生成")
        
        # 6. 生成交互式关系网络图 - 已包含在 plot_enhanced_network 中，无需单独调用
        print(f"{datetime.now()} - 交互式关系网络图已在优化关系网络图步骤中生成")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        traceback.print_exc()

def plot_gnn_strength_distribution(combined_df, results_dir):
    """优化GNN关系强度分布图"""
    print(f"{datetime.now()} - 生成优化GNN关系强度分布图...")
    
    if 'GNN强度' not in combined_df.columns:
        print("警告: 缺少GNN强度列，无法生成分布图")
        return False
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图: 完整分布（对数尺度）
    sns.histplot(combined_df['GNN强度'], kde=True, bins=50, ax=ax1)
    ax1.set_yscale('log')  # 使用对数尺度
    ax1.set_title('GNN关系强度分布(对数尺度)')
    ax1.set_xlabel('GNN强度')
    ax1.set_ylabel('计数(对数)')
    
    # 右图: 高强度区域（>0.1）的详细分布
    high_strength = combined_df[combined_df['GNN强度'] > 0.1]
    if not high_strength.empty:
        sns.histplot(high_strength['GNN强度'], kde=True, bins=30, ax=ax2)
        ax2.set_title('高强度区域分布(GNN强度 > 0.1)')
        ax2.set_xlabel('GNN强度')
        ax2.set_ylabel('计数')
    else:
        ax2.text(0.5, 0.5, '无高强度数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('高强度区域分布(无数据)')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(results_dir, '优化GNN关系强度分布.png')
    plt.savefig(output_path)
    plt.close()
    
    # 添加统计信息
    stats = {
        '总关系数': len(combined_df),
        '高强度关系数(GNN强度 > 0.1)': len(high_strength),
        '高强度比例': f"{len(high_strength)/len(combined_df)*100:.2f}%",
        '平均GNN强度': combined_df['GNN强度'].mean(),
        'GNN强度中位数': combined_df['GNN强度'].median()
    }
    
    print("GNN强度统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"优化GNN关系强度分布图已保存: {output_path}")
    return True
def plot_enhancement_comparison(combined_df, results_dir):
    """优化参数对比散点图"""
    print(f"{datetime.now()} - 生成优化参数对比图...")
    
    required_cols = ['暗恋参数', '暗恋参数_GNN']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"警告: 缺少必要列 {missing_cols}，无法生成对比图")
        return False
    
    # 计算差异
    combined_df = combined_df.copy()
    combined_df['参数差异'] = combined_df['暗恋参数_GNN'] - combined_df['暗恋参数']
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图: 原始散点图
    scatter = ax1.scatter(combined_df['暗恋参数'], combined_df['暗恋参数_GNN'], 
                         c=combined_df['参数差异'], cmap='RdBu_r', alpha=0.6, s=30)
    ax1.plot([0, 100], [0, 100], 'r--', alpha=0.7, label='y=x')
    ax1.set_xlabel('原始暗恋参数')
    ax1.set_ylabel('GNN增强暗恋参数')
    ax1.set_title('原始参数 vs GNN增强参数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('参数差异(GNN-原始)')
    
    # 右图: 差异分布直方图
    ax2.hist(combined_df['参数差异'], bins=30, alpha=0.7, color='purple')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='无差异线')
    ax2.set_xlabel('参数差异(GNN-原始)')
    ax2.set_ylabel('计数')
    ax2.set_title('参数差异分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(results_dir, '优化参数对比分析.png')
    plt.savefig(output_path)
    plt.close()
    
    # 添加统计信息
    diff_stats = {
        '平均差异': combined_df['参数差异'].mean(),
        '差异标准差': combined_df['参数差异'].std(),
        '正差异比例': f"{(combined_df['参数差异'] > 0).sum()/len(combined_df)*100:.2f}%",
        '负差异比例': f"{(combined_df['参数差异'] < 0).sum()/len(combined_df)*100:.2f}%",
        '最大正差异': combined_df['参数差异'].max(),
        '最大负差异': combined_df['参数差异'].min()
    }
    
    print("参数差异统计信息:")
    for key, value in diff_stats.items():
        print(f"  {key}: {value}")
    
    print(f"优化参数对比分析图已保存: {output_path}")
    return True
def visualize_gnn_results(combined_df, results_dir):
    """可视化GNN增强分析结果（优化版）"""
    if '暗恋参数_GNN' not in combined_df.columns:
        print("警告: 无GNN增强数据可可视化")
        return
    
    try:
        # 1. GNN强度分布分析
        print(f"{datetime.now()} - 生成GNN强度分布分析...")
        plot_gnn_strength_distribution(combined_df, results_dir)
        
        # 2. 参数对比分析
        print(f"{datetime.now()} - 生成参数对比分析...")
        plot_enhancement_comparison(combined_df, results_dir)
    
        print(f"{datetime.now()} - GNN增强分析结果可视化完成")
    except Exception as e:
        print(f"GNN可视化过程中出现错误: {e}")
        traceback.print_exc()