# OpenSearch 故障排除手册

**生成时间**: 2025-08-26  
**版本**: 1.0  
**适用环境**: Docker + OpenSearch 3.1.0 + IK分词器

## 📋 目录
- [常见问题](#常见问题)
- [问题诊断流程](#问题诊断流程)
- [解决方案](#解决方案)
- [预防措施](#预防措施)
- [脚本工具](#脚本工具)

## 🚨 常见问题

### 1. OpenSearch Dashboards 显示 "Server is not ready yet"

**症状描述**:
- 访问 `localhost:5601` 时显示 "OpenSearch Dashboards server is not ready yet"
- 容器状态显示为 "Up"，但服务无法正常访问

**可能原因**:
- 版本不兼容
- OpenSearch 还在启动过程中
- 插件安装失败
- 网络配置问题

### 2. OpenSearch API 无法访问

**症状描述**:
- `curl` 访问 `localhost:9200` 失败
- 端口 9200 在监听，但 API 返回错误
- 容器内部进程正常，但 HTTP 接口未就绪

**可能原因**:
- OpenSearch 启动时间不够
- 配置文件错误
- 安全配置问题

### 3. 版本兼容性问题

**症状描述**:
- 构建成功但运行时出错
- 日志显示版本不兼容信息
- 插件无法正常工作

## 🔍 问题诊断流程

### 第一步：检查容器状态
```bash
docker-compose ps
```

### 第二步：检查容器日志
```bash
# OpenSearch 日志
docker-compose logs opensearch | tail -20

# Dashboards 日志
docker-compose logs opensearch-dashboards | tail -20
```

### 第三步：检查网络连接
```bash
# 检查端口监听
netstat -an | grep 9200

# 检查容器内部状态
docker exec opensearch jps
```

### 第四步：检查配置文件
```bash
# 查看 OpenSearch 配置
docker exec opensearch cat /usr/share/opensearch/config/opensearch.yml

# 查看插件列表
docker exec opensearch /usr/share/opensearch/bin/opensearch-plugin list
```

## 🛠️ 解决方案

### 问题1：版本不兼容

**解决方案**:
1. 确保 Dockerfile 和 docker-compose.yml 中的版本一致
2. 推荐使用相同的主版本号（如都是 3.1.0）

**修改示例**:
```yaml
# docker-compose.yml
opensearch-dashboards:
  image: opensearchproject/opensearch-dashboards:3.1.0  # 与 OpenSearch 版本一致
```

### 问题2：插件安装失败

**解决方案**:
1. 使用正确的插件下载地址
2. 确保插件版本与 OpenSearch 版本兼容

**正确的 Dockerfile**:
```dockerfile
FROM opensearchproject/opensearch:3.1.0
ENV OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g"

# 使用官方推荐的插件源
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch https://release.infinilabs.com/analysis-ik/stable/opensearch-analysis-ik-3.1.0.zip

# 创建配置目录
RUN mkdir -p /usr/share/opensearch/config/opensearch-analysis-ik
```

### 问题3：启动时间不足

**解决方案**:
1. 给 OpenSearch 足够的启动时间（通常需要 2-5 分钟）
2. 使用等待脚本检测服务就绪状态

**等待脚本示例**:
```bash
#!/bin/bash
echo "等待 OpenSearch 启动..."
for i in {1..60}; do
    response=$(curl -k -s -u "admin:${OPENSEARCH_PASSWORD}" https://localhost:9200 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo "✅ OpenSearch 已启动！"
        break
    fi
    echo "尝试 $i/60，等待 5 秒..."
    sleep 5
done
```

### 问题4：配置错误

**关键配置项**:
```yaml
# docker-compose.yml 环境变量
environment:
  - discovery.type=single-node                    # 单节点模式
  - bootstrap.memory_lock=true                   # 内存锁定
  - "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m"    # JVM 参数
  - plugins.security.ssl.http.enabled=true      # 启用 HTTPS
  - plugins.security.ssl.transport.enabled=true # 启用传输层安全
```

## 🚀 预防措施

### 1. 版本管理
- 始终使用兼容的版本组合
- 在升级前测试兼容性
- 记录成功的版本组合

### 2. 配置标准化
- 使用环境变量而不是硬编码
- 创建配置模板
- 定期备份配置文件

### 3. 监控和日志
- 设置日志轮转
- 监控容器资源使用
- 定期检查服务健康状态

## 📁 脚本工具

### 1. 服务状态检查脚本
```bash
#!/bin/bash
echo "=== OpenSearch 状态检查 ==="
docker-compose ps
echo "端口监听状态:"
netstat -an | grep 9200
```

### 2. 日志分析脚本
```bash
#!/bin/bash
echo "=== 错误日志检查 ==="
docker-compose logs opensearch | grep -i error | tail -10
docker-compose logs opensearch-dashboards | grep -i error | tail -10
```

### 3. 配置验证脚本
```bash
#!/bin/bash
echo "=== 配置验证 ==="
docker exec opensearch cat /usr/share/opensearch/config/opensearch.yml | grep -E "(version|port|network)"
```

## 🔧 快速修复命令

### 重启服务
```bash
docker-compose down
docker-compose up -d
```

### 强制重建
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 清理数据（谨慎使用）
```bash
docker-compose down -v  # 删除卷数据
docker-compose up -d
```

## 📞 故障排除检查清单

- [ ] 容器状态是否为 "Up"
- [ ] 端口是否正确监听
- [ ] 版本是否兼容
- [ ] 插件是否正确安装
- [ ] 配置文件是否正确
- [ ] 是否等待足够启动时间
- [ ] 网络配置是否正确
- [ ] 安全配置是否正确

## 📚 参考资源

- [OpenSearch 官方文档](https://opensearch.org/docs/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [IK 分词器文档](https://github.com/medcl/elasticsearch-analysis-ik)

---

**注意**: 本手册基于实际故障排除经验编写，如遇新问题请及时更新。
