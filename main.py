import os
import sys
import site
import time
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import config
from modules.gnn_model import enhance_with_gnn
import traceback

# 强制实时输出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

print("="*50)
print("QQ聊天记录分析系统启动")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")
print("="*50)

# ======================== 项目初始化和路径设置 ========================
# 获取项目根目录（main.py所在目录）
ROOT_DIR = Path(__file__).resolve().parent

# 1. 创建必要目录
required_dirs = ['data', 'results', 'modules']
missing_dirs = []

for dir_name in required_dirs:
    dir_path = ROOT_DIR / dir_name
    if not dir_path.exists():
        missing_dirs.append(dir_name)
        os.makedirs(dir_path, exist_ok=True)

if missing_dirs:
    print(f"已创建缺失目录: {', '.join(missing_dirs)}")
    
# 2. 检查必要文件
required_files = {
    '配置文件': ROOT_DIR / 'config.py'
}

for name, path in required_files.items():
    if not path.exists():
        print(f"警告: {name}不存在于 {path}")
        if name == '配置文件':
            # 创建默认配置文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write("# 配置文件\n\n")
                f.write("import os\n")
                f.write("from pathlib import Path\n\n")
                f.write("# 项目根目录\n")
                f.write("ROOT_DIR = Path(__file__).resolve().parent\n\n")
                f.write("# 数据文件路径\n")
                f.write("DATA_PATH = ROOT_DIR / 'data' / '消息记录.csv'\n\n")
                f.write("# 结果输出目录\n")
                f.write("RESULTS_DIR = ROOT_DIR / 'results'\n")
                f.write("os.makedirs(RESULTS_DIR, exist_ok=True)\n\n")
                f.write("# 分析参数配置\n")
                f.write("WINDOWS = {\n")
                f.write("    'response': 300,   # 响应时间窗口（秒）\n")
                f.write("    'sentiment': 600,   # 情感分析窗口（秒）\n")
                f.write("    'interaction': 900  # 互动频率窗口（秒）\n")
                f.write("}\n\n")
                f.write("# 权重配置\n")
                f.write("AFFECTION_WEIGHTS = {\n")
                f.write("    '互动频率': 0.3,\n")
                f.write("    '响应速度': 0.4,\n")
                f.write("    '情感倾向': 0.3\n")
                f.write("}\n\n")
                f.write("# 可视化参数\n")
                f.write("TOP_N_USERS = 20       # 显示的用户数量\n")
                f.write("SENTIMENT_THRESHOLD = 40  # 情感阈值\n")
                f.write("SNAKEY_THRESHOLD = 30    # 桑基图阈值\n")
                f.write("NETWORK_THRESHOLD = 0.3  # 网络图阈值\n")
                f.write("# GNN配置\n")
                f.write("GNN_CONFIG = {\n")
                f.write("    'use_gpu': False,  # 是否使用GPU\n")
                f.write("    'hidden_dim': 128,\n")
                f.write("    'num_layers': 2,\n")
                f.write("    'model_type': 'GAT',\n")
                f.write("    'dropout': 0.3,\n")
                f.write("    'learning_rate': 0.01,\n")
                f.write("    'epochs': 200,\n")
                f.write("    'gnn_threshold': 0.5\n")
                f.write("}\n")
            print(f"已创建默认配置文件: {path}")

# 3. 添加项目路径到系统路径
sys.path.insert(0, str(ROOT_DIR))
MODULES_DIR = ROOT_DIR / 'modules'
sys.path.insert(0, str(MODULES_DIR))

# 4. 尝试添加虚拟环境的site-packages路径
try:
    # 获取当前Python解释器的site-packages目录
    site_packages = next(p for p in site.getsitepackages() if 'site-packages' in p)
    sys.path.insert(0, site_packages)
    
    # 尝试添加虚拟环境的Lib/site-packages
    venv_path = Path(sys.prefix)
    venv_site_packages = venv_path / 'Lib' / 'site-packages'
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))
except (StopIteration, AttributeError):
    pass

# 5. 动态导入项目模块（带有错误处理）
try:
    from modules import data_loader, sentiment_analyzer, core_metrics, affection_metrics, visualization
    import config
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("可能原因: 1) 模块文件缺失 2) 依赖包未安装")
    print("请尝试安装依赖包: pip install pandas numpy matplotlib seaborn jieba snownlp networkx plotly scikit-learn numba lxml html5lib torch torch-geometric")
    sys.exit(1)

# ======================== 主程序 ========================
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

def main():
    # 设置中文字体
    set_chinese_font()
    
    print(f"{datetime.now()} - 开始QQ聊天记录分析")
    
    # === 详细路径检查 ===
    print("=== 文件系统检查 ===")
    print(f"配置的数据文件路径: {config.DATA_PATH}")
    print(f"文件是否存在: {os.path.exists(config.DATA_PATH)}")
    
    data_dir = os.path.dirname(config.DATA_PATH)
    if os.path.exists(data_dir):
        print(f"数据目录内容: {os.listdir(data_dir)}")
    else:
        print(f"数据目录不存在: {data_dir}")
    
    print(f"结果目录: {config.RESULTS_DIR}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"结果目录内容: {os.listdir(config.RESULTS_DIR) if os.path.exists(config.RESULTS_DIR) else '目录不存在'}")
    print("===================")
    
    # 检查数据文件是否存在
    if not os.path.exists(config.DATA_PATH):
        print(f"错误: 数据文件不存在于 {config.DATA_PATH}")
        print("请确认:")
        print("1. 文件已放在正确位置")
        print("2. 文件名是否为 '消息记录.csv'")
        print("3. Docker卷挂载配置正确")
        return
    
    try:
        # 1. 加载数据
        print(f"{datetime.now()} - 加载数据...")
        df = data_loader.load_data(config.DATA_PATH)
        
        # 2. 情感分析
        print(f"{datetime.now()} - 开始情感分析...")
        df = sentiment_analyzer.analyze_messages(df)
        
        # 3. 计算核心指标
        print(f"{datetime.now()} - 开始计算核心指标...")
        response_data, sentiment_data, interaction_data, preprocessed_df = core_metrics.calculate_core_metrics(df, config.WINDOWS)
        
        # 4. 构建暗恋参数
        print(f"{datetime.now()} - 构建暗恋参数...")
        combined_df = affection_metrics.build_affection_metrics(
            response_data, 
            sentiment_data, 
            interaction_data,
            preprocessed_df,
            config.AFFECTION_WEIGHTS
        )
        
        # ===== 插入位置：打印DataFrame列名 =====
        print(f"DataFrame列名: {combined_df.columns.tolist()}")
        # ===================================
        # 5. 保存结果
        output_path = os.path.join(config.RESULTS_DIR, '暗恋参数分析结果.csv')
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"{datetime.now()} - 分析结果已保存到 {output_path}")
        
    except Exception as e:
        print(f"核心分析过程中出现错误: {e}")
        traceback.print_exc()
        return
    
    # 6. GNN增强分析
    print(f"{datetime.now()} - 开始GNN增强分析（CPU模式）")
    try:
        combined_df = enhance_with_gnn(combined_df)
        gnn_output_path = os.path.join(config.RESULTS_DIR, '暗恋参数_GNN增强分析结果.csv')
        combined_df.to_csv(gnn_output_path, index=False, encoding='utf-8-sig')
        print(f"{datetime.now()} - GNN增强分析完成，结果已保存到 {gnn_output_path}")
        
        # 7. 可视化GNN结果
        print(f"{datetime.now()} - 开始可视化GNN增强分析结果")
        visualization.visualize_gnn_results(combined_df, config.RESULTS_DIR)
        print(f"{datetime.now()} - GNN增强分析结果可视化完成")
    except Exception as e:
        print(f"GNN增强分析失败: {str(e)}")
        traceback.print_exc()
    
    # 8. 常规可视化
    print(f"{datetime.now()} - 开始常规可视化...")
    try:
        # 安全获取网络图阈值，如果不存在则使用默认值0.3
        network_threshold = getattr(config, 'NETWORK_THRESHOLD', 0.3)
        
        visualization.visualize_results(
            combined_df,
            config.RESULTS_DIR,
            top_n=config.TOP_N_USERS,
            sentiment_threshold=config.SENTIMENT_THRESHOLD,
            sankey_threshold=config.SNAKEY_THRESHOLD,
            network_threshold=network_threshold
        )
        print(f"{datetime.now()} - 常规可视化完成")
    except Exception as e:
        print(f"常规可视化失败: {str(e)}")
        traceback.print_exc()
    
    print(f"{datetime.now()} - 分析完成!")
    print(f"结果保存在: {config.RESULTS_DIR}")

if __name__ == "__main__":
    # GPU检测逻辑优化
    try:
        import torch
        # 安全获取GNN配置，如果不存在则使用默认值
        use_gpu = getattr(config, 'GNN_CONFIG', {}).get('use_gpu', False)
        if use_gpu and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"{datetime.now()} - 检测到可用GPU: {device_name}")
            torch.cuda.set_device(0)
        elif use_gpu:
            print(f"{datetime.now()} - 警告: 配置要求使用GPU但未检测到可用GPU，将使用CPU")
    except ImportError:
        print(f"{datetime.now()} - 警告: 未安装PyTorch，GPU加速不可用")
    
    try:
        main()
    except Exception as e:
        print(f"程序异常结束: {str(e)}")
        traceback.print_exc()
    
    # 仅当在Windows命令行环境中才等待用户输入
    if os.name == 'nt' and sys.stdout.isatty():
        input("按任意键继续...")  # 防止窗口立即关闭