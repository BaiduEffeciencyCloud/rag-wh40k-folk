import json
import os
import numpy as np
from typing import Dict, Any, List

class NumpyEncoder(json.JSONEncoder):
    """ 自定义JSON编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ReportGenerator:
    """
    评估报告生成器
    
    负责将评估结果格式化为人类可读的报告（如 Markdown）或机器可读的格式（如 JSON）。
    """

    @staticmethod
    def generate_summary_report(summary_data: Dict[str, Any], report_path: str) -> None:
        """
        生成简要的 Markdown 格式评估报告。

        Args:
            summary_data: 包含顶层评估指标的字典。
                          预期格式:
                          {
                              "frame_accuracy": float,
                              "intent_accuracy": float,
                              "slot_f1_report": {
                                  "micro avg": {"precision": float, "recall": float, "f1-score": float}
                              }
                          }
            report_path: 报告输出路径 (e.g., 'reports/summary.md')
        """
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        slot_metrics = summary_data.get("slot_f1_report", {}).get("micro avg", {})
        
        content = f"""# 评估摘要报告

## 核心指标

| 指标 (Metric) | 分数 (Score) |
|---|---|
| 🎯 **Frame Accuracy (北极星)** | {summary_data.get("frame_accuracy", 0):.4f} |
| 💬 **Intent Accuracy** | {summary_data.get("intent_accuracy", 0):.4f} |

## 槽位填充 (Slot Filling) - 总体

| Precision | Recall | F1-Score |
|---|---|---|
| {slot_metrics.get("precision", 0):.4f} | {slot_metrics.get("recall", 0):.4f} | **{slot_metrics.get("f1-score", 0):.4f}** |
"""

        # 添加混淆矩阵部分
        if "confusion_matrix" in summary_data:
            confusion_matrix = summary_data["confusion_matrix"]
            labels = confusion_matrix.get("labels", [])
            matrix = confusion_matrix.get("matrix", [])
            
            if labels and matrix:
                content += "\n## 意图混淆矩阵\n\n"
                content += "| 预测 \\ 真实 |"
                for label in labels:
                    content += f" **{label}** |"
                content += "\n|---|---"
                content += "|" * len(labels)
                content += "\n"
                
                for i, row in enumerate(matrix):
                    content += f"| **{labels[i]}** |"
                    for val in row:
                        content += f" {val} |"
                    content += "\n"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def generate_detailed_report(detailed_data: Dict[str, Any], report_path: str) -> None:
        """
        生成详细的 JSON 格式评估报告。

        Args:
            detailed_data: 包含所有详细评估结果的字典。
            report_path: 报告输出路径 (e.g., 'reports/detailed.json')
        """
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder) 