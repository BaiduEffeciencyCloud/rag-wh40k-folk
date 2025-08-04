from typing import Dict, Any, List, Tuple

from .metrics import MetricCalculator
from .report_generator import ReportGenerator

class Evaluator:
    """
    评估流程的总控制器。
    负责协调模型预测、指标计算和报告生成。
    
    该模块的核心设计思想是作为一个自动化的"质量门禁"(Quality Gate)。
    它产出的核心指标 `frame_accuracy` 将作为上层调用者（如训练流水线）
    决策的依据。例如，只有当 `frame_accuracy` 超过某个预设阈值时，
    训练好的模型才被认为是“合格的”，并被允许导出和部署。
    这种机制可以有效防止因数据质量下降或模型训练偶发问题
    而导致的低质量模型流入生产环境。
    """
    def __init__(self, model: Any, config: Dict[str, Any], mode: str = "full_analysis"):
        """
        初始化评估器。

        Args:
            model (Any): 训练好的模型对象。该对象必须包含一个 `predict` 方法，
                         其签名应为: `predict(self, texts: List[str]) -> Tuple[List[str], List[List[str]], List[Dict[str, float]]]`
                         分别返回意图列表、槽位标签列表和意图置信度字典列表。
            config (Dict[str, Any]): 评估配置，应包含报告输出路径。
                                     e.g., {"summary_report_path": "reports/summary.md",
                                            "detailed_report_path": "reports/detailed.json"}
            mode (str): 评估模式， 'fast_check' 或 'full_analysis'。
        """
        if not hasattr(model, 'predict') or not callable(model.predict):
            raise TypeError("模型对象必须包含一个可调用的 'predict' 方法。")
        
        self.model = model
        self.config = config
        self.mode = mode
        self.metric_calculator = MetricCalculator()
        self.report_generator = ReportGenerator()

    def run(self, texts: List[str], true_intents: List[str], true_slots: List[List[str]]) -> float:
        """
        执行评估流程。

        Args:
            texts (List[str]): 用于评估的文本列表。
            true_intents (List[str]): 真实的意图标签列表。
            true_slots (List[List[str]]): 真实的槽位标签列表。

        Returns:
            float: Frame Accuracy 分数，作为评估的主要返回值。
        """
        # 1. 使用模型进行预测
        pred_intents, pred_slots, intent_confidences = self.model.predict(texts)

        # 2. 计算各项指标
        intent_accuracy = self.metric_calculator.calculate_intent_accuracy(true_intents, pred_intents)
        slot_f1_report = self.metric_calculator.calculate_slot_f1(true_slots, pred_slots)
        frame_accuracy = self.metric_calculator.calculate_frame_accuracy(
            true_intents, pred_intents, true_slots, pred_slots
        )

        # 3. 准备报告数据
        summary_data = {
            "frame_accuracy": frame_accuracy,
            "intent_accuracy": intent_accuracy,
            "slot_f1_report": slot_f1_report
        }

        # 4. 生成报告
        summary_report_path = self.config.get("summary_report_path", "reports/summary.md")
        self.report_generator.generate_summary_report(summary_data, summary_report_path)

        if self.mode == "full_analysis":
            detailed_data = {
                "summary_metrics": summary_data,
                "per_sample_results": [
                    {
                        "text": text,
                        "true_intent": true_intent,
                        "pred_intent": pred_intent,
                        "true_slots": true_slot,
                        "pred_slots": pred_slot,
                        "intent_confidences": intent_confidence,  # 添加置信度信息
                    }
                    for text, true_intent, pred_intent, true_slot, pred_slot, intent_confidence in zip(
                        texts, true_intents, pred_intents, true_slots, pred_slots, intent_confidences
                    )
                ]
            }
            detailed_report_path = self.config.get("detailed_report_path", "reports/detailed.json")
            self.report_generator.generate_detailed_report(detailed_data, detailed_report_path)
            
        return frame_accuracy 