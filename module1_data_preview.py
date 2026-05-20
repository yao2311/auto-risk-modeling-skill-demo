#!/usr/bin/env python3
"""
module1_data_preview.py - 模块一：数据预览与字段识别（DeerFlow Skill CLI）

Actions:
  load    Step 1: 读取文件，打印基本信息 + 前 10 行预览
  detect  Step 2: 智能识别 info/Y/X 列，输出分类建议
  run     Step 3: 用户确认后，保存 _stage1_result.parquet + _stage1_meta.json

Usage:
  # Step 1: 基本信息 + 预览
  python module1_data_preview.py --action load --file /path/to/data.csv

  # Step 2: 智能列分类检测
  python module1_data_preview.py --action detect --file /path/to/data.csv

  # Step 3: 确认后落盘
  python module1_data_preview.py --action run --file /path/to/data.csv \
    --y-col is_default --info-cols user_id,apply_date \
    --output-dir /mnt/user-data/outputs
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# 全局常量
# ============================================================

# 业务缺失值定义（与模块二保持一致）
ABNORMAL_NUM_VALUES = [-1, -999, -1111, -9999]
ABNORMAL_STR_VALUES = ['-1', '-999', '-1111', '-9999']

# 信息列识别规则：按优先级排列，列名包含任一关键词即匹配
INFO_COL_RULES = {
    "ID类":     ["id", "no", "num", "code", "seq", "test_id", "sample_id", "loan_id"],
    "姓名类":   ["name", "姓名"],
    "身份证类": ["id_number", "id_no", "id_card", "id_number_org", "cert"],
    "手机号类": ["mobile", "phone", "tel", "cellphone"],
    "时间类":   ["date", "time", "time_point", "month", "mob", "dt", "apply_dt"],
    "组织机构类": ["org", "branch", "channel", "dept", "region", "area", "source"],
}

INFO_COL_EMOJI = {
    "ID类":     "🆔",
    "姓名类":   "👤",
    "身份证类": "📋",
    "手机号类": "📱",
    "时间类":   "📅",
    "组织机构类": "🏢",
    "未识别":   "🏷️",
}

# 常见 Y 列候选名
Y_COL_CANDIDATES = [
    "target", "label", "y", "is_default", "is_fraud", "is_bad",
    "flag", "risk", "good_bad", "bad_flag", "target_flag",
    "overdue", "is_overdue", "default", "fraud",
]


# ============================================================
# 工具函数
# ============================================================

def get_file_size(path: str) -> str:
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f} MB"
    else:
        return f"{size / 1024 ** 3:.2f} GB"


def read_data(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """读取 CSV / Excel / Parquet，返回 DataFrame"""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            for enc in ["gbk", "gb2312", "latin1"]:
                try:
                    return pd.read_csv(file_path, encoding=enc)
                except UnicodeDecodeError:
                    continue
            raise
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")




# ============================================================
# 公共输出：基本信息 + 前 10 行预览（load / detect 共用）
# ============================================================

def _print_basic_info_and_preview(df: pd.DataFrame, file_path: str):
    """输出基本信息 + 前 10 行预览，load 和 detect action 共用"""
    file_name = os.path.basename(file_path)
    file_size = get_file_size(file_path)
    file_format = os.path.splitext(file_path)[-1].lower()
    n_rows, n_cols = df.shape
    n_duplicates = df.duplicated().sum()

    print()
    print("=" * 60)
    print("  【模块一 数据预览 — 基本信息】")
    print("=" * 60)
    print(f"  文件名:   {file_name}")
    print(f"  格式:     {file_format}")
    print(f"  大小:     {file_size}")
    print(f"  维度:     {n_rows} 行 × {n_cols} 列")
    print(f"  重复行数: {n_duplicates} 行")
    print("=" * 60)

    # 前 10 行预览
    preview = df.head(10)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print()
    print("=" * 60)
    print("  【数据预览（前 10 行）】")
    print("=" * 60)
    print(preview.to_string())
    print("=" * 60)


# ============================================================
# Action: load — 基本信息 + 前 10 行预览
# ============================================================

def action_load(df: pd.DataFrame, file_path: str):
    """Step 1：输出基本信息 + 数据预览"""
    _print_basic_info_and_preview(df, file_path)

# ============================================================
# Action: detect — 智能列分类检测
# ============================================================

def _match_info_category(col_name_lower: str):
    """根据列名匹配信息列类别，返回 (类别名, 匹配关键词) 或 None"""
    for category, keywords in INFO_COL_RULES.items():
        for kw in keywords:
            if kw in col_name_lower:
                return category, kw
    return None


def _detect_y_column(df: pd.DataFrame, col: str):
    """检测某列是否为 Y 列候选，返回 (置信度, 原因列表, 值分布信息)"""
    series = df[col]
    reasons = []
    col_lower = col.lower().strip()

    # 条件 1：列名匹配
    name_match = any(cand in col_lower for cand in Y_COL_CANDIDATES)

    # 条件 2：0/1 二值
    unique_vals = sorted(series.dropna().unique())
    is_binary_01 = False
    if len(unique_vals) == 2:
        if set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}:
            is_binary_01 = True
        elif set(unique_vals) == {False, True}:
            is_binary_01 = True

    # 值分布
    val_counts = series.value_counts()
    total = len(series)
    dist_info = []
    for val, cnt in val_counts.items():
        dist_info.append(f"{val}={cnt}条({cnt/total*100:.1f}%)")

    if name_match and is_binary_01:
        confidence = "high"
        reasons.append("列名匹配常见Y列名")
        reasons.append("值为0/1二值")
    elif name_match:
        confidence = "medium"
        reasons.append("列名匹配常见Y列名")
        reasons.append(f"值为非0/1 ({len(unique_vals)}个唯一值)")
    elif is_binary_01:
        confidence = "medium"
        reasons.append("值为0/1二值")
        reasons.append("列名不在常见Y列名列表中")
    else:
        confidence = "low"

    return confidence, reasons, dist_info


def action_detect(df: pd.DataFrame, file_path: str):
    """Step 2：智能识别 info/Y/X 列，输出结构化分类建议"""
    # 先输出基本信息 + 前 10 行预览（与 load 共用）
    _print_basic_info_and_preview(df, file_path)
    n_rows, n_cols = df.shape

    # --- 第 1 层：信息列匹配 ---
    info_by_category = {cat: [] for cat in INFO_COL_RULES}
    info_matched_cols = set()

    for col in df.columns:
        col_lower = col.lower().strip()
        match_result = _match_info_category(col_lower)
        if match_result:
            category, keyword = match_result
            info_by_category[category].append(col)
            info_matched_cols.add(col)

    # --- 第 2 层：未匹配 info 规则的列 ---
    unmatched_cols = [c for c in df.columns if c not in info_matched_cols]

    # --- 第 3 层：从剩余列中检测 Y 列候选 ---
    y_candidates = {"high": [], "medium": [], "low": []}
    y_candidate_cols = set()

    for col in unmatched_cols:
        confidence, reasons, dist_info = _detect_y_column(df, col)
        if confidence in ("high", "medium"):
            y_candidates[confidence].append({
                "col": col,
                "reasons": reasons,
                "dist_info": dist_info,
            })
            y_candidate_cols.add(col)

    # --- 第 4 层：候选 X 列 = 全部列 - 已匹配 info - Y 候选 ---
    candidate_x_cols = [c for c in unmatched_cols if c not in y_candidate_cols]
    num_x = [c for c in candidate_x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_x = [c for c in candidate_x_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # ============================================================
    # 输出结构化结果
    # ============================================================

    print()
    print("=" * 60)
    print("  【智能列分类检测】")
    print("=" * 60)
    print(f"  总列数: {n_cols}  |  总行数: {n_rows}")
    print("=" * 60)

    # --- 信息列 ---
    print()
    print("  📌 信息列 / 标识列（自动识别）")
    print("  " + "-" * 56)
    matched_count = 0
    for category in INFO_COL_RULES:
        cols = info_by_category[category]
        emoji = INFO_COL_EMOJI.get(category, "🏷️")
        if cols:
            print(f"  {emoji} {category}（{len(cols)} 列）：")
            for c in cols:
                print(f"      └── {c}")
            matched_count += len(cols)
    if matched_count == 0:
        print("  （未检测到明显的标识列）")

    # --- Y 列候选 ---
    print()
    print("  🎯 疑似目标变量列（Y 列候选）")
    print("  " + "-" * 56)
    if y_candidates["high"]:
        print(f"  🔴 高置信度（列名匹配 + 0/1 二值）：")
        for item in y_candidates["high"]:
            print(f"      └── {item['col']}")
            for reason in item["reasons"]:
                print(f"           ✓ {reason}")
            print(f"           值分布：{', '.join(item['dist_info'])}")
    if y_candidates["medium"]:
        print(f"  🟡 中置信度（仅满足一项条件）：")
        for item in y_candidates["medium"]:
            print(f"      └── {item['col']}")
            for reason in item["reasons"]:
                print(f"           ✓ {reason}")
            print(f"           值分布：{', '.join(item['dist_info'])}")
    if not y_candidates["high"] and not y_candidates["medium"]:
        print("  （未检测到疑似 Y 列，需用户手动指定）")

    # --- 候选 X 列 ---
    print()
    print("  📊 候选特征列（X 列）")
    print("  " + "-" * 56)
    print(f"  共计 {len(candidate_x_cols)} 列")
    print(f"      └── 数值型: {len(num_x)} 列")
    print(f"      └── 字符型: {len(cat_x)} 列")
    if len(candidate_x_cols) <= 30:
        if num_x:
            print(f"      数值型: {num_x}")
        if cat_x:
            print(f"      字符型: {cat_x}")
    else:
        print(f"      （列数较多，仅统计数量）")

    # --- 列名总览 ---
    print()
    print("=" * 60)
    print("  【全部列名一览】")
    print("  " + "-" * 56)
    all_cols_list = list(df.columns)
    for i, col in enumerate(all_cols_list, 1):
        marker = ""
        if col in info_matched_cols:
            marker = " [信息列]"
        elif col in y_candidate_cols:
            marker = " [Y列候选]"
        elif col in candidate_x_cols:
            marker = " [X列候选]"
        print(f"  {i:>3}. {col}{marker}")
    print("=" * 60)

    print()
    print("=" * 60)
    print("  【下一步：逐类确认】")
    print("=" * 60)
    print("  请按以下顺序与用户确认：")
    print("  1️⃣  信息列/标识列 — 确认或调整自动识别的结果")
    print("  2️⃣  目标变量列（Y 列）— 确认或手动指定")
    print("  3️⃣  特征列（X 列）— 自动计算，确认即可")
    print("=" * 60)


# ============================================================
# Action: run — 用户确认后落盘中间数据
# ============================================================

def action_run(df: pd.DataFrame, file_path: str, y_col: str, info_cols_str: str,
               output_dir: str = "./"):
    """用户确认列角色后，计算 X 列并保存中间结果"""
    # 解析 info_cols
    info_cols = []
    if info_cols_str:
        info_cols = [c.strip() for c in info_cols_str.split(",") if c.strip()]
        invalid = [c for c in info_cols if c not in df.columns]
        if invalid:
            print(f"❌ 以下信息列不存在：{invalid}")
            return False

    # 校验 y_col
    if y_col:
        if y_col not in df.columns:
            print(f"❌ Y列 '{y_col}' 不存在于数据中")
            return False
        if y_col in info_cols:
            print(f"❌ Y列 '{y_col}' 同时出现在信息列中，请检查")
            return False

    # 计算 x_cols
    excluded = list(info_cols)
    if y_col:
        excluded.append(y_col)
    x_cols = [c for c in df.columns if c not in excluded]

    # 分类 X 列
    num_x_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_x_cols = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # 打印确认
    print()
    print("=" * 60)
    print("  【列角色确认 - 已锁定】")
    print("=" * 60)

    if y_col:
        y_counts = df[y_col].value_counts()
        y_total = len(df)
        print(f"  🎯 Y列: {y_col}")
        for val, cnt in y_counts.items():
            print(f"      └── {val} : {cnt} 条 ({cnt / y_total * 100:.1f}%)")
    else:
        print(f"  🎯 Y列: 未指定")

    if info_cols:
        print(f"\n  📌 信息列 / 标识列 : {len(info_cols)} 列")
        for col in info_cols:
            print(f"      └── {col}")
    else:
        print(f"\n  📌 信息列 / 标识列 : 0 列（未指定）")

    print(f"\n  📊 X列（特征）: {len(x_cols)} 列")
    print(f"      └── 数值型: {len(num_x_cols)} 列")
    print(f"      └── 字符型: {len(cat_x_cols)} 列")
    print("=" * 60)
    print("  ✅ 列角色已确认并保存")
    print("=" * 60)

    # 保存中间结果
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, "_stage1_result.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"✅ 数据已保存：{parquet_path}")

    meta = {
        "y_col": y_col,
        "info_cols": info_cols,
        "x_cols": x_cols,
        "num_cols": num_x_cols,
        "cat_cols": cat_x_cols,
        "file_name": os.path.basename(file_path),
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(output_dir, "_stage1_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ 元数据已保存：{meta_path}")

    # 模块二启动提示
    print()
    print("=" * 60)
    print("  ✅ 模块一完成")
    print("=" * 60)
    print("  中间数据已保存：")
    print(f"    📄 {parquet_path}")
    print(f"    📄 {meta_path}")
    print()
    print("  📢 下一步：是否启动模块二「样本特征统计」？")
    print("     模块二将对 X 特征列进行精细化统计分析")
    print("     （缺失率、有值率、分位数、偏度/峰度、IV值 等）")
    print("     请回复 y（启动）或 n（跳过）")
    print("=" * 60)

    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeerFlow Skill - 模块一: 数据预览与字段识别"
    )
    parser.add_argument(
        "--action", required=True,
        choices=["load", "detect", "run"],
        help="执行动作"
    )
    parser.add_argument(
        "--file", default=None,
        help="数据文件路径（CSV / Excel / Parquet）[load/detect/run 需要]"
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="CSV 文件编码，默认 utf-8"
    )
    parser.add_argument(
        "--y-col", default=None,
        help="目标变量列名 [run 需要]"
    )
    parser.add_argument(
        "--info-cols", default=None,
        help="信息列/标识列，多个用逗号分隔 [run 需要]"
    )
    parser.add_argument(
        "--output-dir", default="/mnt/user-data/outputs",
        help="输出目录，默认 /mnt/user-data/outputs"
    )

    args = parser.parse_args()

    # load / detect / run：需要 --file
    if not args.file:
        print(f"❌ --action {args.action} 需要 --file 参数")
        sys.exit(1)
    if not os.path.exists(args.file):
        print(f"❌ 文件不存在：{args.file}")
        sys.exit(1)

    try:
        df = read_data(args.file, args.encoding)
    except Exception as e:
        print(f"❌ 文件读取失败：{e}")
        sys.exit(1)

    if df is None or len(df) == 0:
        print("❌ 数据为空")
        sys.exit(1)

    if args.action == "load":
        action_load(df, args.file)

    elif args.action == "detect":
        action_detect(df, args.file)

    elif args.action == "run":
        success = action_run(
            df, args.file,
            y_col=args.y_col,
            info_cols_str=args.info_cols,
            output_dir=args.output_dir
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()