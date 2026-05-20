#!/usr/bin/env python3
"""
module2_feature_stats.py - 模块二：样本特征统计（DeerFlow Skill CLI）

Actions:
  stats   统计分析：缺失率/有值率 + 数值描述统计 + 分箱IV + 类别型唯一值/Top5占比
  clean   清洗 + 规则筛选：应用业务缺失值清洗，按用户规则筛选特征，导出清洗后数据

Usage:
  # 统计分析（用户交互选择分箱方法）
  python module2_feature_stats.py --action stats \
    --input-dir /mnt/user-data/outputs \
    --output-dir /mnt/user-data/outputs

  # 清洗 + 规则筛选
  python module2_feature_stats.py --action clean \
    --input-dir /mnt/user-data/outputs \
    --output-dir /mnt/user-data/outputs \
    --retention-rule "有值率 >= 90% and IV >= 0.02"

  # 清洗全部保留
  python module2_feature_stats.py --action clean \
    --input-dir /mnt/user-data/outputs \
    --output-dir /mnt/user-data/outputs \
    --retention-rule "全部保留"
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 全局常量
# ============================================================

ABNORMAL_NUM_VALUES = [-1, -999, -1111, -9999]
ABNORMAL_STR_VALUES = ['-1', '-999', '-1111', '-9999']

# 分箱方法选项
BINNING_METHODS = {
    "equal_frequency": "等频分箱",
    "equal_width": "等距分箱",
    "chi_square": "卡方分箱",
    "decision_tree": "决策树分箱",
    "toad_optimal": "Toad 最优分箱（推荐）",
}

# IV 判定标准
IV_JUDGMENT = [
    (0.02, "无预测力"),
    (0.1, "弱预测力"),
    (0.3, "中等预测力"),
    (float("inf"), "强预测力"),
]

# 规则支持的指标别名 → 宽表列名映射
RETENTION_COL_MAP = {
    "有值率": "有值率",
    "缺失率": "总缺失率",
    "iv": "IV",
    "唯一值数量": "唯一值数量",
}


# ============================================================
# 工具函数
# ============================================================

def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_stage1_result(input_dir: str):
    """读取模块一的输出结果"""
    parquet_path = os.path.join(input_dir, "_stage1_result.parquet")
    meta_path = os.path.join(input_dir, "_stage1_meta.json")

    if not os.path.exists(parquet_path):
        print(f"❌ 未找到 Stage 1 结果文件：{parquet_path}")
        print("   请先执行 module1_data_preview.py --action run")
        sys.exit(1)

    if not os.path.exists(meta_path):
        print(f"❌ 未找到 Stage 1 元数据文件：{meta_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return df, meta


def _get_iv_judgment(iv_value):
    """根据 IV 值返回判定标签"""
    for threshold, label in IV_JUDGMENT:
        if iv_value < threshold:
            return label
    return "未知"


# ============================================================
# WOE / IV 计算核心
# ============================================================

def _safe_woe_iv(good_cnt, bad_cnt, total_good, total_bad, eps=0.5):
    """
    安全计算 WOE 和 IV。
    当某个 bin 中 good 或 bad 为 0 时，使用拉普拉斯平滑（+eps）。
    """
    good_adj = good_cnt + eps
    bad_adj = bad_cnt + eps
    total_good_adj = total_good + eps * 2
    total_bad_adj = total_bad + eps * 2

    good_pct = good_adj / total_good_adj
    bad_pct = bad_adj / total_bad_adj

    woe = np.log(bad_pct / good_pct)
    iv = (bad_pct - good_pct) * woe
    return woe, iv


def _bin_numeric_series(series, method="toad_optimal", target=None, n_bins=10):
    """
    对数值型序列进行分箱，返回 (bin_labels_series, bin_edges)。
    """
    valid_mask = series.notna()
    s = series[valid_mask].copy()

    if len(s) == 0:
        return pd.Series([np.nan] * len(series), index=series.index), None

    if method == "equal_frequency":
        try:
            bin_edges = pd.qcut(s, q=n_bins, duplicates="drop", retbins=True)[1]
            bin_labels = pd.cut(series, bins=bin_edges, include_lowest=True)
        except Exception:
            bin_edges = pd.qcut(s, q=min(n_bins, s.nunique()), duplicates="drop", retbins=True)[1]
            bin_labels = pd.cut(series, bins=bin_edges, include_lowest=True)
        return bin_labels, bin_edges

    elif method == "equal_width":
        bin_labels, bin_edges = pd.cut(series, bins=n_bins, retbins=True, include_lowest=True)
        return bin_labels, bin_edges

    elif method in ("chi_square", "decision_tree", "toad_optimal"):
        return _toad_binning(series, method, target, n_bins)

    return pd.Series([np.nan] * len(series), index=series.index), None


def _toad_binning(series, method="toad_optimal", target=None, n_bins=10):
    """
    使用 Toad 库进行分箱。
    method 映射:
      toad_optimal → toad.transform.Combiner 默认最优分箱
      chi_square   → method='chi'
      decision_tree → method='dt'
    """
    try:
        import toad
    except ImportError:
        print("⚠️  Toad 未安装，回退到等频分箱")
        return _bin_numeric_series(series, "equal_frequency", target, n_bins)

    if target is None:
        print("⚠️  未指定 Y 列，无法使用 Toad 分箱，回退到等频分箱")
        return _bin_numeric_series(series, "equal_frequency", target, n_bins)

    valid_mask = series.notna() & target.notna()
    if valid_mask.sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index), None

    s = series[valid_mask]
    t = target[valid_mask]

    # 确保 target 是 0/1 二值
    unique_t = sorted(t.unique())
    if len(unique_t) != 2 or set(unique_t) != {0, 1}:
        print("⚠️  Y 列非 0/1 二值，无法计算 WOE/IV，回退到等频分箱")
        return _bin_numeric_series(series, "equal_frequency", target, n_bins)

    try:
        toad_method_map = {
            "toad_optimal": "dt",  # toad 默认 method='dt' 即决策树最优分箱
            "chi_square": "chi",
            "decision_tree": "dt",
        }
        toad_method = toad_method_map.get(method, "dt")

        c = toad.transform.Combiner()
        # 构建临时 DataFrame
        tmp = pd.DataFrame({"x": s.values, "y": t.values})
        c.fit(tmp, x="x", y="y", method=toad_method, n_bins=n_bins)

        # 获取分箱边界
        bin_rules = c.export()
        bin_edges = [-np.inf]
        for rule in bin_rules.get("x", []):
            if isinstance(rule, (int, float)):
                bin_edges.append(rule)
        bin_edges.append(np.inf)
        bin_edges = sorted(set(bin_edges))

        bin_labels = pd.cut(series, bins=bin_edges, include_lowest=True)
        return bin_labels, bin_edges

    except Exception as e:
        print(f"⚠️  Toad 分箱失败 ({e})，回退到等频分箱")
        return _bin_numeric_series(series, "equal_frequency", target, n_bins)


def calc_iv_for_column(series, target, method="toad_optimal", n_bins=10, is_numeric=True):
    """
    对单列计算 IV 值。
    返回 dict: {iv_value, is_qualified, judgment, woe_details}
    """
    valid_mask = series.notna() & target.notna()
    if valid_mask.sum() == 0:
        return {"iv_value": None, "is_qualified": False, "judgment": "数据不足", "woe_details": None}

    s = series[valid_mask]
    t = target[valid_mask]

    unique_t = sorted(t.unique())
    if len(unique_t) != 2 or set(unique_t) != {0, 1}:
        return {"iv_value": None, "is_qualified": False, "judgment": "Y列非二值", "woe_details": None}

    total_good = int((t == 0).sum())
    total_bad = int((t == 1).sum())

    if is_numeric:
        bin_labels, bin_edges = _bin_numeric_series(series, method, target, n_bins)
        bin_labels_aligned = bin_labels.loc[valid_mask]
    else:
        bin_labels_aligned = s.astype(str)
        bin_edges = None

    iv_total = 0.0
    woe_details = []

    for bin_val in bin_labels_aligned.unique():
        if pd.isna(bin_val):
            continue
        mask = bin_labels_aligned == bin_val
        good_cnt = int((t[mask] == 0).sum())
        bad_cnt = int((t[mask] == 1).sum())
        woe, iv_bin = _safe_woe_iv(good_cnt, bad_cnt, total_good, total_bad)
        iv_total += iv_bin
        woe_details.append({
            "分箱": str(bin_val),
            "好样本数": good_cnt,
            "坏样本数": bad_cnt,
            "WOE": round(woe, 4),
            "IV贡献": round(iv_bin, 6),
        })

    judgment = _get_iv_judgment(iv_total)
    is_qualified = iv_total >= 0.02

    return {
        "iv_value": round(iv_total, 6),
        "is_qualified": is_qualified,
        "judgment": judgment,
        "woe_details": woe_details,
        "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
    }


# ============================================================
# Action: stats — 统计分析宽表
# ============================================================

def _build_stats_wide_table(df, meta, binning_method, calc_cat_iv):
    """
    构建统计分析宽表，每行一个 X 特征列。
    """
    x_cols = meta.get("x_cols", [])
    y_col = meta.get("y_col", None)
    num_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]
    total = len(df)

    target = df[y_col] if y_col and y_col in df.columns else None

    rows = []
    col_order = [
        "列名", "数据类型",
        "总行数", "有值数", "有值率",
        "NaN缺失数", "NaN缺失率",
        "业务缺失合计数", "业务缺失合计率",
        "总缺失数", "总缺失率",
        # 数值型专属
        "均值", "标准差", "众数",
        "最小值", "P1", "P5", "P25", "P50", "P75", "P95", "P99", "最大值",
        "偏度", "峰度",
        "IV", "IV判定",
        # 类别型专属
        "唯一值数量", "唯一值占比",
        "Top1值", "Top1占比",
        "Top2值", "Top2占比",
        "Top3值", "Top3占比",
        "Top4值", "Top4占比",
        "Top5值", "Top5占比",
    ]

    for col in x_cols:
        series = df[col]
        is_num = pd.api.types.is_numeric_dtype(series)
        nan_count = int(series.isna().sum())

        # --- 缺失值统计 ---
        if is_num:
            abn_counts = {v: int((series == v).sum()) for v in ABNORMAL_NUM_VALUES}
        else:
            abn_counts = {str(v): int((series.astype(str) == v).sum()) for v in ABNORMAL_STR_VALUES}
        abn_total = sum(abn_counts.values())
        total_miss = nan_count + abn_total
        valid_count = total - total_miss

        row = {
            "列名": col,
            "数据类型": str(series.dtype),
            "总行数": total,
            "有值数": valid_count,
            "有值率": f"{valid_count / total * 100:.2f}%",
            "NaN缺失数": nan_count,
            "NaN缺失率": f"{nan_count / total * 100:.2f}%",
            "业务缺失合计数": abn_total,
            "业务缺失合计率": f"{abn_total / total * 100:.2f}%",
            "总缺失数": total_miss,
            "总缺失率": f"{total_miss / total * 100:.2f}%",
        }

        # --- 数值型专属统计 ---
        if is_num:
            mask_valid = series.notna() & ~series.isin(ABNORMAL_NUM_VALUES)
            s = series[mask_valid]
            vcnt = len(s)
            if vcnt > 0:
                mode_vals = s.mode()
                mode_val = round(float(mode_vals.iloc[0]), 4) if len(mode_vals) > 0 else None
                row["均值"] = round(float(s.mean()), 4)
                row["标准差"] = round(float(s.std()), 4)
                row["众数"] = mode_val
                row["最小值"] = round(float(s.min()), 4)
                row["P1"] = round(float(s.quantile(0.01)), 4)
                row["P5"] = round(float(s.quantile(0.05)), 4)
                row["P25"] = round(float(s.quantile(0.25)), 4)
                row["P50"] = round(float(s.quantile(0.50)), 4)
                row["P75"] = round(float(s.quantile(0.75)), 4)
                row["P95"] = round(float(s.quantile(0.95)), 4)
                row["P99"] = round(float(s.quantile(0.99)), 4)
                row["最大值"] = round(float(s.max()), 4)
                row["偏度"] = round(float(s.skew()), 4)
                row["峰度"] = round(float(s.kurt()), 4)
            else:
                for k in ["均值", "标准差", "众数", "最小值",
                          "P1", "P5", "P25", "P50", "P75", "P95", "P99", "最大值",
                          "偏度", "峰度"]:
                    row[k] = None

            # IV 计算（仅数值型）
            if target is not None:
                iv_result = calc_iv_for_column(series, target, method=binning_method,
                                               n_bins=10, is_numeric=True)
                row["IV"] = iv_result["iv_value"]
                row["IV判定"] = iv_result["judgment"]
            else:
                row["IV"] = None
                row["IV判定"] = "Y列未指定"

        else:
            # 类别型：填充数值型字段为 None
            for k in ["均值", "标准差", "众数", "最小值",
                      "P1", "P5", "P25", "P50", "P75", "P95", "P99", "最大值",
                      "偏度", "峰度"]:
                row[k] = None

            # --- 类别型专属统计 ---
            mask_valid = series.notna() & ~series.astype(str).isin(ABNORMAL_STR_VALUES)
            s = series[mask_valid]
            vcnt = len(s)
            if vcnt > 0:
                nunique = s.nunique()
                row["唯一值数量"] = nunique
                row["唯一值占比"] = f"{nunique / vcnt * 100:.2f}%"
                vc = s.value_counts()
                for i in range(1, 6):
                    if i <= len(vc):
                        row[f"Top{i}值"] = str(vc.index[i - 1])
                        row[f"Top{i}占比"] = f"{vc.iloc[i - 1] / vcnt * 100:.2f}%"
                    else:
                        row[f"Top{i}值"] = None
                        row[f"Top{i}占比"] = None
            else:
                row["唯一值数量"] = 0
                row["唯一值占比"] = None
                for i in range(1, 6):
                    row[f"Top{i}值"] = None
                    row[f"Top{i}占比"] = None

            # 类别型 IV（可选）
            if target is not None and calc_cat_iv:
                iv_result = calc_iv_for_column(series, target, method=binning_method,
                                               n_bins=10, is_numeric=False)
                row["IV"] = iv_result["iv_value"]
                row["IV判定"] = iv_result["judgment"]
            else:
                row["IV"] = None
                row["IV判定"] = "未计算" if not calc_cat_iv else "Y列未指定"

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # 按 col_order 排序列（仅保留存在的列）
    existing_cols = [c for c in col_order if c in result_df.columns]
    return result_df[existing_cols], num_cols, cat_cols, target


def action_stats(df, meta, output_dir):
    """统计宽表计算 + 导出"""
    x_cols = meta.get("x_cols", [])
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])
    total = len(df)

    print()
    print("=" * 60)
    print("  【模块二 — 样本特征统计分析】")
    print("=" * 60)
    print(f"  样本总行数: {total}")
    print(f"  X 特征列: {len(x_cols)} (数值型 {len(num_cols)} / 类别型 {len(cat_cols)})")
    print("=" * 60)

    # --- 交互：选择分箱方法 ---
    print()
    print("  📦 请选择 IV 分箱方法：")
    for key, label in BINNING_METHODS.items():
        print(f"     {key} — {label}")
    print()
    print("  请输入方法名（默认: toad_optimal）：")

    binning_method = "toad_optimal"  # 默认，Agent 可以通过参数覆盖

    # --- 交互：类别型 IV ---
    calc_cat_iv = True  # 默认计算，Agent 可以通过参数覆盖

    # --- 构建宽表 ---
    print()
    print("  ⏳ 正在计算特征统计宽表...")
    wide_df, num_cols_used, cat_cols_used, target = _build_stats_wide_table(
        df, meta, binning_method, calc_cat_iv
    )

    # --- 输出摘要 ---
    # 有值率统计
    def parse_pct(val_str):
        try:
            return float(str(val_str).replace("%", ""))
        except (ValueError, AttributeError):
            return np.nan

    valid_pcts = wide_df["有值率"].apply(parse_pct)
    iv_vals = pd.to_numeric(wide_df["IV"], errors="coerce")

    cnt_over_95 = int((valid_pcts >= 95).sum())
    cnt_over_90 = int((valid_pcts >= 90).sum())
    cnt_over_80 = int((valid_pcts >= 80).sum())
    cnt_iv_qualified = int((iv_vals >= 0.02).sum())
    cnt_iv_strong = int((iv_vals >= 0.1).sum())

    print()
    print("=" * 60)
    print("  【统计宽表摘要】")
    print("=" * 60)
    print(f"  特征列总数:          {len(wide_df)}")
    print(f"  ── 缺失情况 ──")
    print(f"  有值率 ≥ 95%:        {cnt_over_95} 列")
    print(f"  有值率 ≥ 90%:        {cnt_over_90} 列")
    print(f"  有值率 ≥ 80%:        {cnt_over_80} 列")
    print(f"  ── IV 情况 ──")
    print(f"  IV ≥ 0.02（有预测力）: {cnt_iv_qualified} 列")
    print(f"  IV ≥ 0.1（中等以上）:  {cnt_iv_strong} 列")
    print("=" * 60)

    # --- 展示宽表前 N 行预览 ---
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 20)
    print()
    print("=" * 70)
    print("  【统计分析宽表 — 预览（前 15 行）】")
    print("=" * 70)
    print(wide_df.head(15).to_string())
    print(f"  ... 共 {len(wide_df)} 行")
    print("=" * 70)

    # --- 导出交互 ---
    print()
    print("=" * 60)
    print("  📢 统计分析宽表已生成。")
    print(f"     是否导出完整统计结果？(y/n)")
    print(f"     （将导出为 feature_stats_{{timestamp}}.xlsx）")
    print("=" * 60)

    # 默认不导出（Agent 通过 --export 控制）
    do_export = getattr(sys.modules[__name__], "_export_flag", False)

    if do_export:
        os.makedirs(output_dir, exist_ok=True)
        ts = timestamp_str()
        excel_path = os.path.join(output_dir, f"feature_stats_{ts}.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            wide_df.to_excel(writer, sheet_name="特征统计宽表", index=False)

            # 汇总 sheet
            summary_data = {
                "项目": [
                    "样本总行数", "X特征列数", "数值型列数", "类别型列数",
                    "有值率≥95%列数", "有值率≥90%列数", "有值率≥80%列数",
                    "IV≥0.02列数", "IV≥0.1列数",
                    "分箱方法", "类别型IV", "导出时间",
                ],
                "值": [
                    total, len(x_cols), len(num_cols), len(cat_cols),
                    cnt_over_95, cnt_over_90, cnt_over_80,
                    cnt_iv_qualified, cnt_iv_strong,
                    BINNING_METHODS.get(binning_method, binning_method),
                    "是" if calc_cat_iv else "否",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="汇总信息", index=False)

        print(f"✅ 统计表已导出：{excel_path}")
        return excel_path, wide_df

    return None, wide_df


# ============================================================
# Action: clean — 规则筛选 + 清洗 + 导出
# ============================================================

def _parse_retention_rule(rule_str, wide_df):
    """
    解析用户保留规则，返回满足条件的列名列表。

    支持格式：
      "有值率 >= 90% and IV >= 0.02"
      "iv >= 0.1"
      "有值率 >= 80% or IV >= 0.3"
      "全部保留"
      "跳过"
    """
    rule_str = rule_str.strip()

    if rule_str in ("全部保留", "全部", "all"):
        return list(wide_df["列名"]), "全部保留"

    if rule_str in ("跳过", "skip", "无", "none"):
        return [], "跳过"

    # 构建列名查找字典（忽略大小写）
    col_lookup = {}
    for col in wide_df.columns:
        col_lookup[col.lower().replace(" ", "")] = col

    def resolve_col(token):
        token_clean = token.strip().lower().replace(" ", "")
        if token_clean in col_lookup:
            return col_lookup[token_clean]
        # 别名映射
        for alias, real_col in RETENTION_COL_MAP.items():
            if alias.lower().replace(" ", "") == token_clean:
                return real_col
        return None

    # 拆分 and/or
    conditions = re.split(r'\s+(and|or)\s+', rule_str, flags=re.IGNORECASE)
    # conditions 形如 ["有值率 >= 90%", "and", "IV >= 0.02"]

    parsed_conditions = []
    logic_ops = []

    for item in conditions:
        item_stripped = item.strip()
        if item_stripped.lower() in ("and", "or"):
            logic_ops.append(item_stripped.lower())
        else:
            # 解析单个条件
            match = re.match(
                r'(.+?)\s*(>=|<=|>|<|==|!=)\s*(.+?)\s*%?\s*$',
                item_stripped, flags=re.IGNORECASE
            )
            if not match:
                print(f"  ⚠️  无法解析条件：「{item_stripped}」，将跳过")
                continue

            col_ref = match.group(1).strip()
            op = match.group(2).strip()
            val_str = match.group(3).strip().replace("%", "")

            real_col = resolve_col(col_ref)
            if real_col is None:
                print(f"  ⚠️  未找到指标「{col_ref}」，可用指标：{list(RETENTION_COL_MAP.keys())}")
                continue

            try:
                val = float(val_str)
            except ValueError:
                print(f"  ⚠️  阈值无法解析：「{val_str}」")
                continue

            parsed_conditions.append((real_col, op, val))

    if not parsed_conditions:
        print("  ⚠️  无有效条件，将全部保留")
        return list(wide_df["列名"]), "全部保留（条件解析失败）"

    # 评估每一行
    def parse_pct_for_eval(v):
        try:
            return float(str(v).replace("%", ""))
        except (ValueError, AttributeError):
            return np.nan

    mask_results = []
    for condition in parsed_conditions:
        col_name, op, threshold = condition
        series = wide_df[col_name].apply(parse_pct_for_eval)

        if op == ">=":
            m = series >= threshold
        elif op == ">":
            m = series > threshold
        elif op == "<=":
            m = series <= threshold
        elif op == "<":
            m = series < threshold
        elif op == "==":
            m = series == threshold
        elif op == "!=":
            m = series != threshold
        else:
            m = pd.Series([False] * len(wide_df))

        mask_results.append(m)

    # 应用逻辑运算
    if not logic_ops:
        final_mask = mask_results[0]
    else:
        final_mask = mask_results[0]
        for i, lop in enumerate(logic_ops):
            if lop == "and":
                final_mask = final_mask & mask_results[i + 1]
            elif lop == "or":
                final_mask = final_mask | mask_results[i + 1]

    retained_cols = wide_df.loc[final_mask, "列名"].tolist()
    removed_cols = wide_df.loc[~final_mask, "列名"].tolist()

    return retained_cols, {
        "rule_str": rule_str,
        "retained_count": len(retained_cols),
        "removed_count": len(removed_cols),
        "retained_cols": retained_cols,
        "removed_cols": removed_cols,
    }


def action_clean(df, meta, retention_rule, output_dir):
    """规则筛选 + 清洗 + 导出"""
    x_cols = meta.get("x_cols", [])
    y_col = meta.get("y_col", None)
    info_cols = meta.get("info_cols", [])

    # 如果没有传 retention_rule 或 rule 为空，提示交互
    if not retention_rule:
        print()
        print("=" * 60)
        print("  【模块二 — 清洗与特征筛选】")
        print("=" * 60)
        print(f"  X 特征列: {len(x_cols)} 列")
        print()
        print("  请输入保留规则，例如：")
        print("    「有值率 >= 90% and IV >= 0.02」")
        print("    「iv >= 0.1」")
        print("    「全部保留」— 仅清洗，不做筛选")
        print("    「跳过」— 不做任何处理")
        print("=" * 60)
        return None  # 让 Agent 处理交互

    abn_num = ABNORMAL_NUM_VALUES
    abn_str = [str(v) for v in abn_num]

    # 需要宽表来解析规则
    binning_method_default = "equal_frequency"

    # 构建轻量级宽表用于规则解析
    wide_df, _, _, _ = _build_stats_wide_table(df, meta, binning_method_default, False)

    # 解析规则
    retained_cols, rule_info = _parse_retention_rule(retention_rule, wide_df)

    if isinstance(rule_info, str) and rule_info == "跳过":
        print()
        print("=" * 60)
        print("  ⏭️  已跳过清洗与筛选")
        print("=" * 60)
        return None

    # 打印筛选预览
    print()
    print("=" * 60)
    print("  【特征筛选预览】")
    print("=" * 60)
    print(f"  规则: {retention_rule}")
    print(f"  保留: {len(retained_cols)} 列")
    print(f"  移除: {len(rule_info.get('removed_cols', []))} 列")
    if len(retained_cols) <= 30:
        print(f"  保留列: {retained_cols}")
        if rule_info.get("removed_cols"):
            print(f"  移除列: {rule_info['removed_cols']}")
    print("=" * 60)

    # 执行清洗
    df_clean = df.copy()
    for col in x_cols:
        series = df_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            df_clean[col] = series.replace(abn_num, np.nan)
        else:
            df_clean[col] = series.replace(abn_str, np.nan)

    # 筛选列：info + Y + 保留的 X
    keep_cols = list(info_cols)
    if y_col:
        keep_cols.append(y_col)
    keep_cols.extend(retained_cols)
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    df_clean = df_clean[keep_cols]

    print(f"\n  ✅ 清洗完成：{abn_num} → NaN")
    print(f"  最终列数: {len(keep_cols)} (info: {len(info_cols)}, Y: {1 if y_col else 0}, X: {len(retained_cols)})")

    # 导出
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp_str()
    output_path = os.path.join(output_dir, f"clean_df_{ts}.xlsx")
    df_clean.to_excel(output_path, index=False)
    print(f"  ✅ 清洗后数据已导出：{output_path}")

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeerFlow Skill - 模块二: 样本特征统计"
    )
    parser.add_argument(
        "--action", required=True,
        choices=["stats", "clean"],
        help="执行动作"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="模块一输出目录（含 _stage1_result.parquet 和 _stage1_meta.json）"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="输出目录（默认与 input-dir 相同）"
    )
    parser.add_argument(
        "--binning-method", default="toad_optimal",
        choices=["equal_frequency", "equal_width", "chi_square", "decision_tree", "toad_optimal"],
        help="IV 分箱方法（默认 toad_optimal）"
    )
    parser.add_argument(
        "--calc-cat-iv", default=True,
        type=lambda x: x.lower() in ("true", "1", "yes", "y"),
        help="是否对类别型特征计算 IV（默认 true）"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="是否导出统计结果 Excel [stats 可选]"
    )
    parser.add_argument(
        "--retention-rule", default=None,
        help="特征保留规则，如 「有值率 >= 90% and IV >= 0.02」[clean 需要]"
    )

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else args.input_dir

    # 读取模块一结果
    df, meta = load_stage1_result(args.input_dir)

    if args.action == "stats":
        # 导出标记
        import builtins
        setattr(sys.modules[__name__], "_export_flag", args.export)
        action_stats(df, meta, output_dir)

    elif args.action == "clean":
        action_clean(df, meta, args.retention_rule, output_dir)


if __name__ == "__main__":
    main()