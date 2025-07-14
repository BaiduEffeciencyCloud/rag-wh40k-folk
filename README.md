# RAG 检索评估体系

## 依赖环境

### Python依赖
- Python >= 3.8
- 推荐使用虚拟环境（venv/conda）
- 安装依赖：
  ```bash
  pip3 install -r requirements.txt
  pip3 install pypandoc
  ```

### 系统依赖
- pandoc（用于markdown转pdf）
- xelatex/texlive-xetex（用于PDF渲染和中文支持）
- 推荐安装命令（Debian/Ubuntu）：
  ```bash
  sudo apt-get update
  sudo apt-get install -y pandoc texlive-xetex
  ```
- MacOS:
  ```bash
  brew install pandoc
  brew install --cask mactex
  ```

### Docker一键部署
本项目已提供Dockerfile，支持一键构建和运行：
```bash
# 构建镜像
sudo docker build -t rag-eval .
# 运行容器（挂载数据目录、指定参数）
sudo docker run --rm -v $PWD:/app rag-eval --query myquery.json --qp straightforward --se dense
```

## 主要功能
- 支持RAG检索评估、自动报告归档、可视化、PDF报告自动生成
- 支持单条query和批量query评估
- 评估报告可直接交给LLM进行解读

## 其他说明
- 若需自定义入口或参数，请修改Dockerfile中的CMD或直接进入容器交互
- 详细用法见各模块注释和示例 
