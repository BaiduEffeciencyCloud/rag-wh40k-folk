# POST-SEARCH 模块实施细则

## 1. 模块结构和文件组织

### 1.1 核心模块结构

postsearch/
├── init.py
├── pipeline.py # 管道主类
├── interfaces.py # 处理器接口定义
├── processors/
│ ├── init.py
│ ├── rerank.py # 重排序处理器
│ ├── mmr.py # MMR多样性处理器
│ ├── filter.py # 过滤处理器
│ └── dedup.py # 去重处理器
└── factory.py # 工厂类


### 1.2 配置文件

配置文件应放置在项目根目录下

engineconfig/
├── post_search_config.py # Post-Search策略配置
└── adaptive_config.py # 自适应策略配置（更新）


## 2. 抽象方法设计

### 2.1 PostSearchProcessor接口
```python
class PostSearchProcessor(ABC):
    @abstractmethod
    def process(self, results: List[Dict], **kwargs) -> List[Dict]:
        """处理检索结果"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取处理器名称"""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict:
        """获取配置模式"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict) -> bool:
        """验证配置"""
        pass
```

### 2.2 PostSearchPipeline接口
```python
class PostSearchPipeline:
    def process(self, results: List[Dict], strategy: List[str], **kwargs) -> List[Dict]:
        """按策略顺序处理结果"""
        pass
    
    def add_processor(self, name: str, processor: PostSearchProcessor):
        """添加处理器"""
        pass
    
    def get_available_processors(self) -> List[str]:
        """获取可用处理器列表"""
        pass
    
    def validate_strategy(self, strategy: List[str]) -> bool:
        """验证策略"""
        pass
```

## 3. Orchestrator改动

### 3.1 新增自适应Orchestrator
```python
class AdaptiveRAGOrchestrator(RAGOrchestrator):
    def __init__(self, query_processor, search_engine, aggregator):
        super().__init__(query_processor, search_engine, aggregator)
        self.post_search = PostSearchPipeline()
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Any:
        # 1. 查询处理（获取查询类型和路由策略）
        processed = self.qp.process(query)
        
        # 2. 检索召回
        results = self.se.search(processed, top_k=top_k)
        
        # 3. Post-Search处理
        post_search_strategy = self._get_post_search_strategy(processed)
        if post_search_strategy:
            results = self.post_search.process(
                results, 
                strategy=post_search_strategy,
                query_info=processed
            )
        
        # 4. 聚合
        result = self.agg.aggregate(results, query, params=kwargs)
        return result
    
    def _get_post_search_strategy(self, processed_query: Dict) -> List[str]:
        # 根据查询类型获取Post-Search策略
        pass
```

### 3.2 需要修改的调用点

#### 3.2.1 searchcontroller.py
- 需要支持自适应模式参数
- 根据参数选择使用 `RAGOrchestrator` 还是 `AdaptiveRAGOrchestrator`

#### 3.2.2 evaltool/ragas_evaluator.py
- 评估时需要支持自适应模式
- 可能需要为不同查询类型建立不同的评估指标

#### 3.2.3 其他测试文件
- 需要更新测试用例以支持新的Orchestrator

## 4. 配置管理改动

### 4.1 新增配置项
```python
# config/post_search_config.py
POST_SEARCH_STRATEGIES = {
    'simple_rule': [],  # 简单规则查询不需要后处理
    'complex_reasoning': ['rerank', 'mmr'],  # 复杂推理需要重排序和多样性
    'enumeration': ['dedup', 'mmr']  # 穷举查询需要去重和多样性
}

# config/adaptive_config.py
ADAPTIVE_STRATEGIES = {
    'simple_rule': {
        'processor': 'straight',
        'search_engine': 'dense',
        'post_search': [],
        'aggregator': 'simple'
    },
    'complex_reasoning': {
        'processor': 'cot',
        'search_engine': 'hybrid',
        'post_search': ['rerank', 'mmr'],
        'aggregator': 'reasoning'
    },
    'enumeration': {
        'processor': 'expand',
        'search_engine': 'enumeration',
        'post_search': ['mmr'],
        'aggregator': 'enumeration'
    }
}
```

## 5. 测试用例更新/新增

### 5.1 单元测试

test/postsearch/
├── init.py
├── test_pipeline.py # 管道测试
├── test_processors/
│ ├── init.py
│ ├── test_rerank.py # 重排序测试
│ ├── test_mmr.py # MMR测试
│ ├── test_filter.py # 过滤测试
│ └── test_dedup.py # 去重测试
└── test_factory.py # 工厂测试


### 5.2 集成测试

test/integration/
├── test_adaptive_orchestrator.py # 自适应编排器测试
├── test_post_search_integration.py # Post-Search集成测试
└── test_query_type_classification.py # 查询分类测试


### 5.3 需要测试的场景
1. **查询类型分类准确性**
   - 测试不同类型查询的分类准确性
   - 边界情况处理

2. **不同查询类型的策略选择**
   - 验证策略选择的正确性
   - 配置变更的影响

3. **Post-Search处理器的效果**
   - 每个处理器的独立功能测试
   - 处理器的组合效果测试

4. **管道处理的顺序和组合**
   - 验证处理顺序的正确性
   - 错误处理和恢复机制

5. **配置验证和错误处理**
   - 无效配置的处理
   - 异常情况的处理

6. **性能测试**
   - 处理时间测试
   - 内存使用测试
   - 并发处理能力

## 6. 实施优先级

### 6.1 第一阶段：基础框架
- 创建PostSearch模块结构
- 实现基础接口和管道
- 创建简单的处理器（如dedup）
- 编写基础单元测试

### 6.2 第二阶段：核心功能
- 实现查询类型分类器
- 实现自适应Orchestrator
- 添加rerank和MMR处理器
- 完善配置管理

### 6.3 第三阶段：集成和优化
- 更新所有调用点
- 完善测试用例
- 性能优化和配置调优
- 文档完善

## 7. 关键设计原则

### 7.1 模块独立性
- PostSearch模块完全独立，不依赖其他模块的具体实现
- 通过接口进行交互，降低耦合度

### 7.2 配置驱动
- 所有策略通过配置文件管理
- 支持运行时策略切换

### 7.3 可扩展性
- 新增处理器只需实现接口
- 支持自定义处理策略

### 7.4 可测试性
- 每个组件都可以独立测试
- 提供完整的测试覆盖

## 8. 风险评估和缓解

### 8.1 技术风险
- **风险**：PostSearch处理可能影响性能
- **缓解**：性能测试和优化，支持开关控制

### 8.2 集成风险
- **风险**：与现有系统的集成复杂性
- **缓解**：渐进式实施，保持向后兼容

### 8.3 配置风险
- **风险**：配置错误可能导致系统异常
- **缓解**：配置验证和默认值处理

## 9. 成功标准

### 9.1 功能标准
- 查询类型分类准确率 > 90%
- PostSearch处理正确率 > 95%
- 系统响应时间增加 < 20%

### 9.2 质量标准
- 单元测试覆盖率 > 90%
- 集成测试通过率 100%
- 代码审查通过率 100%

### 9.3 性能标准
- 支持并发处理
- 内存使用合理
- 错误恢复时间 < 5秒

## 10. 总结

这个Post-Search模块实施细则提供了完整的实施路线图，包括：
- 清晰的模块结构和接口设计
- 详细的集成方案
- 完整的测试策略
- 分阶段的实施计划
- 风险控制和成功标准

通过这个方案，可以确保Post-Search模块的顺利实施和系统的稳定运行。
