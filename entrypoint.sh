#!/bin/bash

# 确保结果目录存在并设置权限
mkdir -p /app/results
chmod -R a+w /app/results

# 执行分析
echo "$(date) - 开始QQ聊天记录分析..."
python /app/main.py
ANALYSIS_EXIT_CODE=$?

# 检查分析结果
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "$(date) - 分析成功完成！"
    echo "结果保存在: /app/results"
    ls -l /app/results
else
    echo "$(date) - 分析失败，错误码: $ANALYSIS_EXIT_CODE"
fi

# 永久运行容器
echo "$(date) - 容器将持续运行以便查看结果"
echo "使用 'docker exec -it <容器名> bash' 进入容器"
exec tail -f /dev/null