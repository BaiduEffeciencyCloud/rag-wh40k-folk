import os
import shutil
from datetime import datetime
import json

def _copytree(src, dst):
    if not os.path.exists(src):
        return
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            _copytree(s, d)
        else:
            shutil.copy2(s, d)

def export_assets(assets_dir, model_dir=None, config_path=None, report_dir=None, real_export=False, mock_data=True, empty=False, extreme=False, write_error=False):
    """
    导出pipeline所有资产到assets/时间戳目录下，结构包括model/config/report/dict等。
    支持真实模型、配置、报告导出。
    """
    ts = datetime.now().strftime("%y%m%d%H%M%S")
    ts_dir = os.path.join(assets_dir, ts)
    os.makedirs(ts_dir, exist_ok=True)
    subdirs = ["model", "config", "report", "dict"]
    for sub in subdirs:
        os.makedirs(os.path.join(ts_dir, sub), exist_ok=True)
    # 生成README和meta
    with open(os.path.join(ts_dir, "export_README.md"), "w", encoding="utf-8") as f:
        f.write("# 导出内容说明\n\n本目录为自动导出的pipeline资产归档。\n")
    with open(os.path.join(ts_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "desc": "自动导出元信息"}, f, ensure_ascii=False, indent=2)
    # 真实导出逻辑
    if real_export:
        # 拷贝模型目录
        if model_dir and os.path.exists(model_dir):
            _copytree(model_dir, os.path.join(ts_dir, "model"))
        # 拷贝配置文件
        if config_path and os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(ts_dir, "config", os.path.basename(config_path)))
        # 拷贝报告目录
        if report_dir and os.path.exists(report_dir):
            _copytree(report_dir, os.path.join(ts_dir, "report"))
        return
    # mock导出内容（兼容原有测试）
    if not empty:
        for sub in subdirs:
            fname = "mock_model.pkl" if sub == "model" else f"mock_{sub}.txt"
            fpath = os.path.join(ts_dir, sub, fname)
            try:
                if write_error and sub == "report":
                    raise IOError("模拟写入失败")
                with open(fpath, "w", encoding="utf-8") as f:
                    if extreme and sub == "model":
                        f.write("0" * 1024 * 1024 * 2)
                    else:
                        f.write(f"mock {sub} content")
            except Exception as e:
                with open(os.path.join(ts_dir, "export_meta.json"), "a", encoding="utf-8") as meta:
                    meta.write(f"\n# {sub}写入异常: {e}") 