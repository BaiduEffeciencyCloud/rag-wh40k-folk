#!/bin/bash

echo "🔄 正在重启 OpenSearch 服务..."

# 1. 停止并清理服务
echo "📥 停止服务并清理卷..."
docker-compose down -v

# 2. 重新启动服务
echo "📤 重新启动服务..."
docker-compose up -d

# 3. 等待 OpenSearch 启动
echo "⏳ 等待 OpenSearch 启动..."
sleep 30

# 4. 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 5. 等待 OpenSearch 完全就绪
echo "⏳ 等待 OpenSearch 完全就绪..."
until curl -k -s -u "admin:Francis198!!!" "https://localhost:9200/_cluster/health" > /dev/null 2>&1; do
    echo "   OpenSearch 还在启动中，继续等待..."
    sleep 10
done

echo "✅ OpenSearch 已就绪！"

# 6. 赋权同义词词典文件
echo "🔐 设置同义词词典文件权限..."
docker exec opensearch chown opensearch:opensearch /usr/share/opensearch/config/analysis/warhammer_synonyms.txt
docker exec opensearch chmod 644 /usr/share/opensearch/config/analysis/warhammer_synonyms.txt

# 7. 验证权限
echo "🔍 验证文件权限..."
docker exec opensearch ls -la /usr/share/opensearch/config/analysis/warhammer_synonyms.txt

# 8. 赋权自定义词典文件
echo "🔐 设置自定义词典文件权限..."
docker exec opensearch chown opensearch:opensearch /usr/share/opensearch/config/analysis/warhammer_dict.txt
docker exec opensearch chmod 644 /usr/share/opensearch/config/analysis/warhammer_dict.txt

# 9. 验证所有文件权限
echo "🔍 验证所有词典文件权限..."
docker exec opensearch ls -la /usr/share/opensearch/config/analysis/

echo "🎉 重启和赋权完成！"
echo "📊 现在可以访问 OpenSearch: https://localhost:9200"
echo "📈 现在可以访问 Dashboards: http://localhost:5601"
