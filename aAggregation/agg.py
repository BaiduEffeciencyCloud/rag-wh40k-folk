from .agginterface import AggregationInterface
from typing import List, Dict, Any, Iterable
import logging
import os
import json
import re
from utils.llm_utils import call_llm, build_messages
from config import DEFAULT_TEMPERATURE, MAX_ANSWER_TOKENS, LLM_TYPE
import config as cfg
from llm.llm_factory import LLMFactory

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'template')

logger = logging.getLogger(__name__)

class DynamicAggregation(AggregationInterface):
    def __init__(self, template: str = 'default'):
        self.template = template or 'default'

    # === 列表结果规范化（服务端兜底定形）===
    @staticmethod
    def _norm_list_text(text: str) -> str:
        """将单行/无分隔的列表文本规范为：首行“共X个”+ 每行以\t开头、以\n结尾。
        - 条目边界基于 [来源N] 或 [来源N,M]（含中文逗号）
        - 保留已有的换行；若无换行则按来源标记拆分
        """
        if not text:
            return text
        # 分离首部统计
        header = ''
        m = re.match(r"^(共\d+个)", text)
        if m:
            header = m.group(1)
            text = text[m.end():].lstrip()

        # 若已有换行且每行基本规整，直接确保每行以\n结尾并补\t
        if "\n" in text:
            lines = [ln for ln in text.split("\n") if ln != ""]
            norm_lines = []
            for ln in lines:
                if not ln:
                    continue
                if not ln.startswith("\t"):
                    norm_lines.append("\t" + ln)
                else:
                    norm_lines.append(ln)
            body = "\n".join(norm_lines)
            return ((header + "\n") if header else "") + (body + ("\n" if not body.endswith("\n") else ""))

        # 无换行：用来源标记切分
        src_pat = r"\[来源[0-9,，]+\]"
        parts = re.split(f"({src_pat})", text)
        if not parts or len(parts) < 3:
            # 没有明显来源标记，尝试基于 “）” 后紧跟下一项“（” 的启发式切分
            # 例如： 名称（描述）名称（描述）... → 按 “）” 与下一项“（” 的边界断开
            heuristic_splits = re.split(r"(?<=）)(?=[^\n\t]*（)", text)
            if len(heuristic_splits) > 1:
                items = [seg.strip() for seg in heuristic_splits if seg.strip()]
                body = "\n".join(["\t" + it for it in items])
                return ((header + "\n") if header else "") + body
            # 仍无法切分则原样返回（仅在前面补 header）
            return ((header + "\n") if header else "") + text

        items = []
        buf = ""
        for seg in parts:
            buf += seg
            if re.fullmatch(src_pat, seg):
                items.append(buf.strip())
                buf = ""
        if buf.strip():
            items.append(buf.strip())

        body = "\n".join(["\t" + it for it in items])
        return ((header + "\n") if header else "") + body

    @classmethod
    def _norm_list_stream(cls, chunks: Iterable[str]) -> Iterable[str]:
        """真正的流式格式化：实时识别完整条目并格式化输出，确保不丢失信息且保持格式。"""
        buffer = ""
        header_sent = False

        for ch in chunks:
            if not isinstance(ch, str):
                ch = str(ch)
            buffer += ch

            # 尝试实时格式化已累积的内容
            while True:
                # 提取首部统计（如果还没发送）
                if not header_sent:
                    header_match = re.match(r"^(共\d+个)", buffer)
                    if header_match:
                        yield header_match.group(1)
                        yield "\n"
                        buffer = buffer[header_match.end():].lstrip()
                        header_sent = True
                        continue

                # 查找下一个完整的条目（以 [来源...] 结束）
                src_pat = r"\[来源[0-9,，]+\]"
                src_match = re.search(src_pat, buffer)

                if src_match:
                    # 找到完整条目，提取并格式化
                    end_pos = src_match.end()
                    item_text = buffer[:end_pos].strip()

                    if item_text:
                        # 输出格式化的条目（前置制表符 + 条目 + 换行）
                        yield "\t" + item_text + "\n"
                        buffer = buffer[end_pos:].lstrip()
                        continue

                # 没有找到完整条目，跳出内循环
                break

        # 处理剩余的buffer（可能是不完整的条目）
        if buffer.strip():
            # 如果还有剩余内容，直接输出（保持原始格式）
            yield buffer

    def extract_answer(self, search_results) -> List[str]:
        # 兼容 List[Dict] 和 List[List[Dict]]
        if not search_results:
            return []
        if isinstance(search_results, list) and search_results and isinstance(search_results[0], dict):
            search_results = [search_results]
        return [item['text'] for collection in search_results for item in collection if 'text' in item and item['text']]

    def _deduplicate_answers(self, answers: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for ans in answers:
            if ans is None or ans == '':
                continue
            if ans not in seen:
                deduped.append(ans)
                seen.add(ans)
        return deduped

    def _load_prompt(self, context: List[str], query: str, params: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        1. 读取system.json作为system prompt
        2. 读取template.json作为user prompt
        3. context和query始终独立组装并拼接到user_prompt后面
        4. 使用build_messages组合为List
        5. 返回prompt列表
        """
        try:
            # 1. 生成context和query部分（与模板读取彻底解耦）
            context_str = '\n'.join([f'【{i+1}】{text}' for i, text in enumerate(context)])
            query_str = query

            # 2. 读取system prompt
            system_path = os.path.join(TEMPLATE_DIR, 'system.json')
            with open(system_path, 'r', encoding='utf-8') as f:
                system_data = json.load(f)
            system_prompt = system_data.get('system_prompt', '')

            # 3. 动态选择user prompt模板
            # 优先使用params中的intent，如果没有则使用初始化时的template
            template_name = 'default'
            if params and 'intent' in params:
                template_name = params['intent']
                logger.info(f"使用动态意图模板: {template_name}")
            else:
                template_name = self.template
                logger.info(f"使用初始化模板: {template_name}")

            
            
            # 4. 读取user prompt模板
            user_template_path = os.path.join(TEMPLATE_DIR, f'{template_name}.json')
            if not os.path.exists(user_template_path):
                logger.warning(f"意图模板 {template_name}.json 不存在，使用默认模板")
                user_template_path = os.path.join(TEMPLATE_DIR, 'default.json')

            with open(user_template_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            user_prompt = user_data.get('user_prompt', '')
            

            # 5. 如果user_prompt为空，自动降级为default.json
            if not user_prompt.strip() and not user_template_path.endswith('default.json'):
                with open(os.path.join(TEMPLATE_DIR, 'default.json'), 'r', encoding='utf-8') as f:
                    default_data = json.load(f)
                user_prompt = default_data.get('user_prompt', '')

            # 6. 如果default.json也为空，兜底只输出context和query
            if not user_prompt.strip():
                user_prompt = f"上下文信息:{context_str}\n用户问题：\"{query_str}\""
            
            # 7. 使用build_messages组合为List，确保包含type字段
            messages = build_messages(
                system_prompt={"content": system_prompt, "type": "text"},
                user_prompt={"content": f"{user_prompt}\n上下文信息:{context_str}\n用户问题：\"{query_str}\"", "type": "text"}
            )
            return messages
        except Exception as e:
            logger.error(f"加载prompt模板失败: {str(e)}")
            return [{"role": "error", "content": f"聚合出错: {str(e)}"}]

    def semantic_parse(self, query: str, context: List[str], params: Dict[str, Any] = None) -> str:
        # 这里只拼装prompt，不直接调用LLM
        return self._load_prompt(context, query, params)

    def aggregate(self, search_results, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        answers = self.extract_answer(search_results)
        deduped = self._deduplicate_answers(answers)
        prompt_text = self.semantic_parse(query, deduped, params) if deduped else "未检索到有效答案"
        # 1. 生成llm实例
        llm_instance = LLMFactory.create_llm(LLM_TYPE)
        # 2. 用llm_instance.build_messages生成prompt（Qwen等自动适配格式）
        messages = llm_instance.build_messages(system_prompt="", user_prompt=prompt_text)
        # 3. 根据配置选择流式/同步调用
        if getattr(cfg, 'STREAM_MODE', False):
            try:
                # 返回迭代器，调用方可逐步消费
                stream_iter = llm_instance.call_llm_stream(
                    prompt=messages,
                    granularity="char",
                    event_mode=False,
                    fallback_to_sync=True
                )
                # 若是 list 模板/意图，进行服务端定形（保障每行 \t 与行尾 \n）
                tpl = (params.get('intent') if params and 'intent' in params else self.template) or 'default'
                if str(tpl).lower() == 'list':
                    stream_iter = self._norm_list_stream(stream_iter)
                else:
                    pass
                return {
                    'answers': deduped,
                    'final_answer_stream': stream_iter,
                    'count': len(deduped),
                    'query': query
                }
            except Exception as e:
                logger.error(f"LLM流式调用失败: {str(e)}，回退同步模式")
                try:
                    final_answer = llm_instance.call_llm(prompt=messages)
                except Exception as ie:
                    logger.error(f"LLM同步回退也失败: {str(ie)}")
                    final_answer = "聚合出错: LLM调用失败"
                # 同步兜底，同样应用定形
                tpl = (params.get('intent') if params and 'intent' in params else self.template) or 'default'
                if str(tpl).lower() == 'list':
                    final_answer = self._norm_list_text(final_answer)
                else:
                    pass
                return {
                    'answers': deduped,
                    'final_answer': final_answer,
                    'count': len(deduped),
                    'query': query
                }
        else:
            try:
                final_answer = llm_instance.call_llm(prompt=messages)
            except Exception as e:
                logger.error(f"LLM调用失败: {str(e)}")
                final_answer = "聚合出错: LLM调用失败"
            # 同步路径，按 list 模板定形
            tpl = (params.get('intent') if params and 'intent' in params else self.template) or 'default'
            if str(tpl).lower() == 'list':
                final_answer = self._norm_list_text(final_answer)
            else:
                pass
            return {
                'answers': deduped,
                'final_answer': final_answer,
                'count': len(deduped),
                'query': query
            } 