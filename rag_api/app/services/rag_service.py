from typing import Dict, Any
import sys
import os
from orchestrator.master import MasterController
# 添加项目根目录到Python路径，以便导入现有的RAG模块
# 获取rag-api-local目录的父目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))  # rag-api-local/app/services
rag_api_dir = os.path.dirname(os.path.dirname(current_dir))  # rag-api-local
project_root = os.path.dirname(rag_api_dir)  # 项目根目录

# 将项目根目录添加到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class RAGService:
    def __init__(self):
        self.master_controller = None
        self._initialized = False
    
    def _ensure_master_controller(self):
        """确保MasterController已初始化（延迟初始化）"""
        if not self._initialized:
            self._initialize_master_controller()
    
    def _initialize_master_controller(self):
        """初始化MasterController，确保工作目录正确"""
        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()
            
            # 切换到项目根目录，这样相对路径就能正确解析
            os.chdir(project_root)

            self.master_controller = MasterController()
            self._initialized = True
            
            # 恢复原来的工作目录
            os.chdir(original_cwd)
            
        except ImportError as e:
            print(f"无法导入MasterController: {e}")
            self.master_controller = None
            self._initialized = False
        except Exception as e:
            print(f"MasterController初始化失败: {e}")
            self.master_controller = None
            self._initialized = False
            # 确保恢复工作目录
            try:
                os.chdir(original_cwd)
            except:
                pass
    
    async def process_query(self, query: str, advance: bool = False) -> Dict[str, Any]:
        """处理查询请求"""
        try:
            # 确保MasterController已初始化
            self._ensure_master_controller()
            
            if not self.master_controller:
                return {
                    "error": "MasterController未初始化",
                    "query": query
                }
            
            # 调用现有的MasterController
            result = self.master_controller.execute_query(query, advance)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
    
    async def identify_intent(self, query: str) -> Dict[str, Any]:
        """识别查询意图"""
        try:
            # 确保MasterController已初始化
            self._ensure_master_controller()
            
            if not self.master_controller:
                return {
                    "error": "MasterController未初始化",
                    "query": query
                }
            
            # 调用现有的MasterController进行意图识别
            intent_result = self.master_controller.process_query(query)
            return {
                "intent_type": intent_result.get('intent_type'),
                "confidence": 0.9,  # 从intent_result中获取
                "processor_config": intent_result.get('processor_config')
            }
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
