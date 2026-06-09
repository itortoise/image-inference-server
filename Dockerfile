FROM python:3.10-slim

# 引入 uv（Rust 编写的极速包管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 阶段1：安装第三方依赖（利用 Docker 缓存层）
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# 阶段2：复制代码并安装当前项目
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
RUN uv sync --no-dev --frozen

# 暴露端口
EXPOSE 8000 8001 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v2/health/ready')" || exit 1

# 启动服务（uv run 自动激活虚拟环境）
CMD ["uv", "run", "--no-dev", "python", "-m", "inference_server.main", "--config", "configs/service.yaml"]
