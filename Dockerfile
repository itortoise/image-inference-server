FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[prod]"

# 复制代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# 暴露端口
EXPOSE 8000 8001 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v2/health/ready')" || exit 1

# 启动服务
CMD ["python", "-m", "inference_server.main", "--config", "configs/service.yaml"]
