# 基础镜像
FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y pandoc texlive-xetex && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install pypandoc

# 默认命令（可根据实际入口调整）
CMD ["python3", "-m", "evaltool.evalsearch.main"] 