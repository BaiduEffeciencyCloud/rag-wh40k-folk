import json
import logging
import random
import os
from typing import Dict, Any
from orchestrator.adaptive import AdaptiveProcessor
from orchestrator.processor import RAGOrchestrator
# 2. 创建RAGOrchestrator实例
from qProcessor.processorfactory import QueryProcessorFactory
from searchengine.search_factory import SearchEngineFactory
from postsearch.factory import PostSearchFactory
from aAggregation.aggfactory import AggFactory
from engineconfig.qp_config import NON_WH40K_RESPONSES
from storage.storage_factory import StorageFactory
logger = logging.getLogger(__name__)

# 从配置文件读取EMBEDDING_TYPE和DB_TYPE
try:
    import config
    from config import EMBEDDING_TYPE, DB_TYPE
except ImportError:
    EMBEDDING_TYPE = "qwen"
    DB_TYPE = "opensearch"
    logger.warning("无法从config.py导入EMBEDDING_TYPE和DB_TYPE，使用默认值")

class MasterController:
    def __init__(self):
        self.adaptive_processor = AdaptiveProcessor()
        self.template_manager = TemplateManager()
        self.fallback_handler = FallbackHandler()
    
    def process_query(self, query: str, advance:bool=False) -> Dict[str, Any]:
        """处理查询并返回processor配置"""
        try:
            # 1. 识别意图
            intent = self._identify_intent(query)
            processor_config = None
            # 2. 生成处理器配置,对于unknown类型和non40k类型一律使用空的处理逻辑
            if intent in {'unknown','non40k'}:
                processor_config ={}
            else:
                processor_config = self._generate_processor_config(intent, advance)
            # 3. 返回结果
            return {
                'intent_type': intent,
                'processor_config': processor_config,
                'query': query
            }
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return self._handle_fallback(query, e)
        
    def _identify_intent(self, query: str) -> str:
        """识别查询意图"""
        try:
            # 使用AdaptiveProcessor进行意图识别
            result = self.adaptive_processor.process(query)
            # result是一个字典，需要从中提取intent_result
            intent_result = result.get('intent_result')
            if intent_result is None:
                logger.error("意图识别结果中缺少intent_result")
                return self._fallback_intent_recognition(query)
            
            intent = intent_result.intent_type
            confidence = intent_result.confidence
            
            logger.info(f"意图识别结果: {intent}, 置信度: {confidence}")
            
            # 如果置信度太低，使用默认意图
            if confidence < 0.5:
                logger.warning(f"置信度过低({confidence})，使用默认意图")
                return 'query'
                
            return intent
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            # 回退到简单的关键词匹配
            return self._fallback_intent_recognition(query)
        
    def _fallback_intent_recognition(self, query: str) -> str:
        """回退的意图识别逻辑 - 直接返回query作为兜底策略"""
        logger.warning(f"ML意图识别失败，使用兜底策略: query")
        return 'query'
        
    def _generate_processor_config(self, intent: str, advance:bool=False) -> Dict[str, Any]:
        """根据意图生成processor配置"""
        try:
            if intent == 'rule' and advance:
                intent="adrule"
            return self.template_manager.load_template(intent)
        except Exception as e:
            logger.error(f"生成处理器配置失败: {e}")
            return self.fallback_handler.get_default_config()
        
    def _handle_fallback(self, query:str, error: Exception) -> Dict[str, Any]:
        """处理回退逻辑"""
        logger.warning(f"使用回退配置: {error}")
        default_config = self.fallback_handler.get_default_config()
        return {
            'intent_type': 'query',
            'processor_config': default_config,
            'query': query
        }
    
    def execute_query(self, query: str, advance:bool=False) -> Dict[str, Any]:
        """执行查询：识别意图并调用processor执行检索"""
        try:
            processor_config = self.process_query(query, advance)
            query_intent = processor_config.get('intent_type')
            if query_intent in {'unknown','non40k'}:
                return {
                    'query': query,
                    'results': self._get_wh40k_response()
                }
            # 从配置中获取参数
            qp_type = processor_config.get('qp', 'straight')
            se_type = processor_config.get('se', 'hybrid')
            ps_type = processor_config.get('ps', 'simple')
            agg_type = processor_config.get('agg', 'query')
            
            # 创建组件实例
            query_processor = QueryProcessorFactory.create_processor(qp_type)
            search_engine = SearchEngineFactory.create(se_type)
            post_search = PostSearchFactory.create_processor(ps_type)
            aggregator = AggFactory.get_aggregator(agg_type)
            
            # 创建RAGOrchestrator

            orchestrator = RAGOrchestrator(query_processor, search_engine, post_search, aggregator)
            
            # 传递EMBEDDING_TYPE和DB_TYPE参数，以及正确的意图
            results = orchestrator.run_with_config(query, processor_config, 
                                                embedding_model=EMBEDDING_TYPE, 
                                                db_type=DB_TYPE,
                                                intent=processor_config.get('intent_type', 'query'))
            
            # 4. 返回结果
            return {
                'query': query,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return {
                'intent_type': 'query',
                'processor_config': self.fallback_handler.get_default_config(),
                'query': query,
                'error': str(e),
                'results': None
            }
    def _get_wh40k_response(self) -> str:
        """随机选择战锤40K阵营的回复"""
        responses = NON_WH40K_RESPONSES.values()
        return random.choice(list(responses))


class TemplateManager:
    def __init__(self):
        self.templates_dir = "orchestrator/templates"
    
    def load_template(self, intent: str) -> Dict[str, Any]:
        """加载指定意图的模板"""
        try:
            # 根据CLOUD_STORAGE配置选择加载方式
            if config.CLOUD_STORAGE:
                logger.info(f"使用云端存储加载模板: {intent}")
                return self._load_from_cloud(intent)
            else:
                logger.info(f"使用本地存储加载模板: {intent}")
                return self._load_from_local(intent)
        except Exception as e:
            logger.error(f"加载模板失败: {e}")
            return self._get_default_template()
    
    def _load_from_local(self, intent: str) -> Dict[str, Any]:
        """从本地文件系统加载模板"""
        try:
            template_path = os.path.join(self.templates_dir, f"{intent}.json")
            if not os.path.exists(template_path):
                logger.warning(f"本地模板文件不存在: {template_path}")
                return self._get_default_template()
                
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
                logger.info(f"成功从本地加载模板: {intent}")
                return template
        except Exception as e:
            logger.error(f"从本地加载模板失败: {e}")
            return self._get_default_template()
    
    def _load_from_cloud(self, intent: str) -> Dict[str, Any]:
        """从云端OSS加载模板"""
        try:
            
            # 创建OSS存储实例
            storage = StorageFactory.create_storage(
                access_key_id=config.OSS_ACCESS_KEY,
                access_key_secret=config.OSS_ACCESS_KEY_SECRET,
                endpoint=config.OSS_ENDPOINT,
                bucket_name=config.OSS_BUCKET_NAME_PARAMS,  # 使用参数Bucket
                region=config.OSS_REGION
            )
            
            # 构建OSS路径: pipeline/{intent}.json
            oss_path = f"pipeline/{intent}.json"
            
            # 检查文件是否存在
            if not storage.exists(oss_path):
                logger.warning(f"云端模板文件不存在: {oss_path}")
                return self._get_default_template()
            
            # 从OSS下载文件内容
            content = storage.load(oss_path)
            if content is None:
                logger.error(f"从云端下载模板失败: {oss_path}")
                return self._load_from_local(intent)
            
            # 解析JSON内容
            template = json.loads(content.decode('utf-8'))
            logger.info(f"成功从云端加载模板: {intent}")
            return template
            
        except Exception as e:
            logger.error(f"从云端加载模板失败: {e}")
            # 如果云端加载失败，尝试本地加载作为fallback
            logger.info(f"云端加载失败，尝试本地加载: {intent}")
            return self._load_from_local(intent)
    
    def _get_default_template(self) -> Dict[str, Any]:
        """获取默认模板"""
        logger.warning("使用默认模板")
        return {
            "qp": "straight",
            "se": "hybrid",
            "ps": "simple",
            "agg": "query",
            "rerank": True,
            "topk": 20
        }




class FallbackHandler:
    def __init__(self):
        pass
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "qp": "straight",
            "se": "hybrid",
            "ps": "simple",
            "agg": "query",
            "rerank": True,
            "topk": 20
        }
