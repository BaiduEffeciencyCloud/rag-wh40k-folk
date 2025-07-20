# 意图识别与槽位填充训练服务目录结构

本目录用于意图识别/槽位填充模型的离线训练、评估与导出。

## 目录结构

```
intent_training/
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 清洗/切分后的数据
├── feature/
│   └── feature_engineering.py
├── model/
│   ├── intent_classifier.py
│   ├── slot_filler.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── utils/
│   ├── metrics.py
│   └── logger.py
├── config.py
├── run_train.sh            # 一键训练脚本
└── README.md
```

## 说明
- `data/`：数据存放与处理
- `feature/`：特征工程相关代码
- `model/`：模型训练、评估、导出
- `utils/`：通用工具函数
- `config.py`：参数配置
- `run_train.sh`：自动化训练脚本

请在各子目录下补充相应代码和数据文件。 