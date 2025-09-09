# OpenSearch 快速修复卡片

**生成时间**: 2025-08-26  
**版本**: 1.0

## 🚨 紧急情况快速修复

### 问题：Dashboards 显示 "Server is not ready yet"

**立即执行**:
```bash
# 1. 检查版本兼容性
docker-compose ps

# 2. 检查日志
docker-compose logs opensearch | tail -10

# 3. 等待启动（关键！）
sleep 180  # 等待3分钟

# 4. 测试API
curl -k -u "admin:Francis198!!!" https://localhost:9200
```

### 问题：API 无法访问

**立即执行**:
```bash
# 1. 检查容器状态
docker-compose ps

# 2. 检查端口
netstat -an | grep 9200

# 3. 重启服务
docker-compose restart opensearch

# 4. 等待启动
sleep 120
```

## 🔧 常用修复命令

### 重启服务
```bash
docker-compose down && docker-compose up -d
```

### 强制重建
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 检查状态
```bash
docker-compose ps
docker-compose logs opensearch | tail -20
```

## ⚠️ 关键注意事项

1. **版本必须一致**: OpenSearch 和 Dashboards 版本要匹配
2. **启动需要时间**: 即使容器显示"Up"，也要等待2-5分钟
3. **插件URL正确**: 使用官方推荐的插件下载地址
4. **配置要完整**: 确保所有必要的环境变量都已设置

## 📱 联系信息

- **手册位置**: `document/OpenSearch故障排除手册.md`
- **详细文档**: 查看完整故障排除手册
- **更新时间**: 2025-08-26

---

**记住**: 耐心等待是成功的关键！OpenSearch需要时间完全启动。
