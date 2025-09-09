FROM opensearchproject/opensearch:3.1.0
# 下载并安装 opensearch-analysis-ik 插件
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch https://release.infinilabs.com/analysis-ik/stable/opensearch-analysis-ik-3.1.0.zip
# 创建 IK 配置文件目录
RUN mkdir -p /usr/share/opensearch/config/opensearch-analysis-ik 