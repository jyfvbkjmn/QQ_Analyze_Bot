# 构建阶段
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS builder

# 关键修复：禁用交互提示 + 预设时区
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai

# 替换APT源（兼容空目录）
RUN { [ -d /etc/apt/sources.list.d ] && find /etc/apt/sources.list.d -name "*.list" -exec sed -i \
    -e "s|http://.*ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g" \
    -e "s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g" {} \ ; } || true \
&& { [ -f /etc/apt/sources.list ] && sed -i \
    -e "s|http://.*ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g" \
    -e "s|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list ; } || true

# 合并安装命令（减少层数 + 避免重复更新）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgraphviz-dev \
    graphviz \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --trusted-host download.pytorch.org \
    -r requirements.txt
# 复制数据文件到镜像中
COPY data/消息记录.csv /app/data/消息记录.csv
# 运行时阶段
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
COPY --from=builder /app/data/消息记录.csv /app/data/消息记录.csv
# 精确复制Graphviz组件
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /usr/share/fonts /usr/share/fonts
COPY --from=builder /usr/lib/x86_64-linux-gnu/graphviz /usr/lib/graphviz
COPY --from=builder /usr/bin/dot /usr/bin/dot

# 设置环境变量（精简）
ENV PYTHONPATH=/opt/conda/lib/python3.10/site-packages \
    LD_LIBRARY_PATH=/usr/lib/graphviz:$LD_LIBRARY_PATH

# 设置工作目录并复制代码
WORKDIR /app

# 复制入口脚本并设置权限
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 确保结果目录存在并设置权限
RUN mkdir -p /app/results && chmod -R a+w /app/results

COPY main.py .
COPY config.py .
COPY modules ./modules
# 设置入口点
ENTRYPOINT ["/app/entrypoint.sh"]

# 入口命令
CMD ["python", "-u", "main.py"]