# geometry_split.py
# -*- coding: utf-8 -*-
"""
几何 holdout split（shot/receiver 域）：
- 先在观测系统层面（shot_id/receiver_id）做 holdout（uniform/random/cluster）
- 再把 holdout 的观测系统 ID 映射为 trace_no（道序号），输出 train/test/val 的 trace 列表
- 训练时可完全不使用 holdout 的 shot/receiver 对应的任何 trace（用 *_train_traces.json 直接筛）

输出文件（在 out_dir 或 out_dir/<mode>/ 下）：
- <file>_train_ids.json / <file>_test_ids.json / <file>_val_ids.json
- <file>_train_traces.json / <file>_test_traces.json / <file>_val_traces.json
- <file>_split_config.json
- <file>_split_visualization_<mode>.png
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


# -----------------------------
# Helpers
# -----------------------------
def _ensure_trace_no(df: pd.DataFrame) -> pd.DataFrame:
    """确保 df 有 trace_no 列；如果没有，用 df.index 作为 trace_no。"""
    if "trace_no" in df.columns:
        return df
    df = df.copy()
    df["trace_no"] = df.index.astype(np.int64)
    return df


def _ids_to_traces(ids: np.ndarray, id_to_traces: Dict[Any, np.ndarray]) -> np.ndarray:
    """把观测系统ID列表映射为 trace_no 列表（去重+排序）。"""
    if ids is None or len(ids) == 0:
        return np.array([], dtype=np.int64)

    chunks = []
    for _id in ids:
        if _id in id_to_traces:
            chunks.append(id_to_traces[_id])

    if len(chunks) == 0:
        return np.array([], dtype=np.int64)

    out = np.concatenate(chunks).astype(np.int64)
    out = np.unique(out)
    return out


def _sort_ids_by_coords(ids: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """返回按坐标排序后的索引（稳定排序：先x后y）。"""
    if coords.ndim != 2 or coords.shape[0] != len(ids):
        raise ValueError("coords must be (n_ids, d) and align with ids length")
    if coords.shape[1] == 2:
        return np.lexsort((coords[:, 1], coords[:, 0]))
    return np.argsort(coords[:, 0])


# -----------------------------
# Split functions (ID-holdout -> trace mapping)
# -----------------------------
def _split_uniform(
    ids: np.ndarray,
    coords: np.ndarray,
    holdout_ratio: float,
    rng: np.random.Generator,
    id_to_traces: Optional[Dict[Any, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    uniform: 先按坐标排序，然后每隔 K 取 1 个ID作为holdout
    返回: train_ids, test_ids, train_traces, test_traces
    """
    sort_indices = _sort_ids_by_coords(ids, coords)
    sorted_ids = ids[sort_indices]

    n_total = len(sorted_ids)
    n_holdout = int(n_total * holdout_ratio)

    if n_holdout <= 0:
        train_ids, test_ids = sorted_ids, np.array([], dtype=sorted_ids.dtype)
    else:
        K = max(1, n_total // n_holdout)
        holdout_indices = np.arange(0, n_total, K)[:n_holdout]
        train_indices = np.setdiff1d(np.arange(n_total), holdout_indices)
        train_ids, test_ids = sorted_ids[train_indices], sorted_ids[holdout_indices]

    if id_to_traces is None:
        # 兜底：不提供映射时，trace集合返回空
        return train_ids, test_ids, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    train_traces = _ids_to_traces(train_ids, id_to_traces)
    test_traces = _ids_to_traces(test_ids, id_to_traces)
    return train_ids, test_ids, train_traces, test_traces


def _split_random(
    ids: np.ndarray,
    coords: np.ndarray,
    holdout_ratio: float,
    rng: np.random.Generator,
    id_to_traces: Optional[Dict[Any, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    random: 先按坐标排序，再随机采样holdout的ID
    返回: train_ids, test_ids, train_traces, test_traces
    """
    sort_indices = _sort_ids_by_coords(ids, coords)
    sorted_ids = ids[sort_indices]

    n_total = len(sorted_ids)
    n_holdout = int(n_total * holdout_ratio)

    if n_holdout <= 0:
        train_ids, test_ids = sorted_ids, np.array([], dtype=sorted_ids.dtype)
    else:
        holdout_indices = rng.choice(n_total, size=n_holdout, replace=False)
        train_indices = np.setdiff1d(np.arange(n_total), holdout_indices)
        train_ids, test_ids = sorted_ids[train_indices], sorted_ids[holdout_indices]

    if id_to_traces is None:
        return train_ids, test_ids, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    train_traces = _ids_to_traces(train_ids, id_to_traces)
    test_traces = _ids_to_traces(test_ids, id_to_traces)
    return train_ids, test_ids, train_traces, test_traces


def _split_cluster(
    ids: np.ndarray,
    coords: np.ndarray,
    holdout_ratio: float,
    rng: np.random.Generator,
    id_to_traces: Optional[Dict[Any, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    cluster: 在排序后的ID序列中生成若干不重叠连续区间（连续缺失模式）
    返回: train_ids, test_ids, train_traces, test_traces
    """
    sort_indices = _sort_ids_by_coords(ids, coords)
    sorted_ids = ids[sort_indices]

    n_total = len(sorted_ids)
    n_holdout = int(n_total * holdout_ratio)

    if n_holdout <= 0:
        train_ids, test_ids = sorted_ids, np.array([], dtype=sorted_ids.dtype)
    elif n_holdout >= n_total:
        train_ids, test_ids = np.array([], dtype=sorted_ids.dtype), sorted_ids
    else:
        # 确定连续区间数量（随比例变化）
        if holdout_ratio < 0.2:
            n_clusters = max(1, int(np.sqrt(n_holdout)))
        elif holdout_ratio < 0.5:
            n_clusters = max(2, int(np.sqrt(n_holdout) * 1.5))
        else:
            n_clusters = max(3, int(np.sqrt(n_holdout) * 2))

        n_clusters = min(n_clusters, max(1, n_holdout // 2), max(1, n_total // 10))
        n_clusters = max(1, n_clusters)

        avg_cluster_size = max(1, n_holdout // n_clusters)
        min_cluster_size = max(1, avg_cluster_size // 2)
        max_cluster_size = max(1, avg_cluster_size * 2)

        intervals = []
        used_indices = set()
        remaining = n_holdout

        max_attempts = n_clusters * 20
        attempts = 0

        while len(intervals) < n_clusters and remaining > 0 and attempts < max_attempts:
            attempts += 1

            if len(intervals) == n_clusters - 1:
                cluster_size = remaining
            else:
                upper = min(max_cluster_size, remaining - (n_clusters - len(intervals) - 1) * min_cluster_size)
                upper = max(upper, min_cluster_size)
                cluster_size = int(rng.integers(min_cluster_size, upper + 1))
                cluster_size = min(cluster_size, remaining)

            if cluster_size <= 0:
                break

            valid_starts = []
            for start in range(n_total - cluster_size + 1):
                end = start + cluster_size
                overlap = False
                for (ps, pe) in intervals:
                    if not (end <= ps or start >= pe):
                        overlap = True
                        break
                if not overlap:
                    valid_starts.append(start)

            if len(valid_starts) == 0:
                cluster_size = max(1, cluster_size - 1)
                valid_starts = []
                for start in range(n_total - cluster_size + 1):
                    end = start + cluster_size
                    if all(end <= ps or start >= pe for (ps, pe) in intervals):
                        valid_starts.append(start)

            if len(valid_starts) > 0:
                start = int(rng.choice(valid_starts))
                end = start + cluster_size
                intervals.append((start, end))
                used_indices.update(range(start, end))
                remaining = n_holdout - len(used_indices)
            else:
                break

        # 在区间间隙补充
        if remaining > 0 and len(intervals) > 0:
            intervals.sort(key=lambda x: x[0])
            for i in range(len(intervals)):
                if remaining <= 0:
                    break
                gap_start = 0 if i == 0 else intervals[i - 1][1]
                gap_end = intervals[i][0]
                gap_size = gap_end - gap_start
                if gap_size > 0:
                    insert_size = min(remaining, gap_size, max_cluster_size)
                    if insert_size > 0:
                        max_start = gap_end - insert_size
                        if max_start >= gap_start:
                            start = int(rng.integers(gap_start, max_start + 1))
                            intervals.append((start, start + insert_size))
                            used_indices.update(range(start, start + insert_size))
                            remaining = n_holdout - len(used_indices)

            if remaining > 0 and len(intervals) > 0:
                last_end = max(e for (_, e) in intervals)
                gap_size = n_total - last_end
                if gap_size > 0:
                    insert_size = min(remaining, gap_size, max_cluster_size)
                    if insert_size > 0:
                        max_start = n_total - insert_size
                        if max_start >= last_end:
                            start = int(rng.integers(last_end, max_start + 1))
                            intervals.append((start, start + insert_size))
                            used_indices.update(range(start, start + insert_size))
                            remaining = n_holdout - len(used_indices)

        # 最后兜底：还缺就从剩余里补（尽量保持连续）
        if remaining > 0:
            available = sorted(list(set(range(n_total)) - used_indices))
            if len(available) >= remaining:
                additional = []
                i = 0
                while len(additional) < remaining and i < len(available):
                    if len(additional) == 0 or available[i] == additional[-1] + 1:
                        additional.append(available[i])
                        i += 1
                    else:
                        if rng.random() < 0.3:
                            additional.append(available[i])
                            i += 1
                        else:
                            i += 1
                            if i < len(available):
                                additional.append(available[i])
                                i += 1

                if len(additional) < remaining:
                    rest = [idx for idx in available if idx not in additional]
                    need = remaining - len(additional)
                    if len(rest) >= need:
                        additional.extend(rng.choice(rest, size=need, replace=False).tolist())

                used_indices.update(additional[:remaining])

        holdout_indices = np.array(sorted(list(used_indices))[:n_holdout], dtype=np.int64)
        train_indices = np.setdiff1d(np.arange(n_total), holdout_indices)
        train_ids, test_ids = sorted_ids[train_indices], sorted_ids[holdout_indices]

    if id_to_traces is None:
        return train_ids, test_ids, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    train_traces = _ids_to_traces(train_ids, id_to_traces)
    test_traces = _ids_to_traces(test_ids, id_to_traces)
    return train_ids, test_ids, train_traces, test_traces


# -----------------------------
# Visualization (observation system only, ID-based)
# -----------------------------
def _visualize_split(
    df: pd.DataFrame,
    file_name: str,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    val_ids: np.ndarray,
    id_col: str,
    out_dir: Path,
    domain: str,
    mode: str,
    holdout_ratio: Optional[float] = None,
) -> None:
    """
    可视化 split 结果：清晰展示观测系统的空间分布
    - 划分域：展示每个唯一观测点位置（每个ID一个代表点）
    - 另一域：展示该split涉及的所有唯一观测点位置
    
    布局：2x2 (划分域/另一域 × train/test)
    """
    # 根据 domain 选择坐标列
    if domain == "shot":
        split_x_col, split_y_col = "sx", "sy"
        split_x_label, split_y_label = "Source X (m)", "Source Y (m)"
        split_title_prefix = "Source"
        other_x_col, other_y_col = "gx", "gy"
        other_x_label, other_y_label = "Receiver X (m)", "Receiver Y (m)"
        other_title_prefix = "Receiver"
        other_id_col = "receiver_id"
    else:
        split_x_col, split_y_col = "gx", "gy"
        split_x_label, split_y_label = "Receiver X (m)", "Receiver Y (m)"
        split_title_prefix = "Receiver"
        other_x_col, other_y_col = "sx", "sy"
        other_x_label, other_y_label = "Source X (m)", "Source Y (m)"
        other_title_prefix = "Source"
        other_id_col = "shot_id"
    
    # 将 ID 集合转换为 set 用于快速查询
    train_id_set = set(train_ids.tolist())
    test_id_set = set(test_ids.tolist())
    val_id_set = set(val_ids.tolist()) if val_ids is not None and len(val_ids) > 0 else set()
    
    # ========== 1. 划分域：计算每个ID的唯一观测点位置 ==========
    def get_split_domain_coords(ids: set) -> np.ndarray:
        """获取划分域中每个ID的代表坐标（平均值）"""
        coords_list = []
        for _id in ids:
            mask = df[id_col] == _id
            if mask.sum() > 0:
                x_coords = df.loc[mask, split_x_col].values
                y_coords = df.loc[mask, split_y_col].values
                coords_list.append([np.mean(x_coords), np.mean(y_coords)])
        return np.array(coords_list) if coords_list else np.empty((0, 2))
    
    train_split_coords = get_split_domain_coords(train_id_set)
    test_split_coords = get_split_domain_coords(test_id_set)
    val_split_coords = get_split_domain_coords(val_id_set)
    
    # ========== 2. 另一域：获取该split涉及的所有唯一观测点位置 ==========
    def get_other_domain_coords_with_ids(ids: set) -> Tuple[np.ndarray, set]:
        """获取另一域中的所有唯一观测点位置，同时返回另一域ID集合"""
        # 找到所有属于这些ID的道
        mask = df[id_col].isin(ids)
        if mask.sum() == 0:
            return np.empty((0, 2)), set()
        
        sub_df = df.loc[mask]
        # 为每个另一域ID计算代表坐标（唯一观测点）
        other_ids = sub_df[other_id_col].unique()
        coords_list = []
        other_id_set = set()
        for oid in other_ids:
            omask = sub_df[other_id_col] == oid
            if omask.sum() > 0:
                x_coords = sub_df.loc[omask, other_x_col].values
                y_coords = sub_df.loc[omask, other_y_col].values
                coords_list.append([np.mean(x_coords), np.mean(y_coords)])
                other_id_set.add(oid)
        return (np.array(coords_list) if coords_list else np.empty((0, 2))), other_id_set
    
    train_other_coords, train_other_id_set = get_other_domain_coords_with_ids(train_id_set)
    test_other_coords, test_other_id_set = get_other_domain_coords_with_ids(test_id_set)
    val_other_coords, val_other_id_set = get_other_domain_coords_with_ids(val_id_set)
    
    # ========== 3. 检查观测系统是否重叠 ==========
    def coords_to_set(coords: np.ndarray, precision: int = 0) -> set:
        """将坐标数组转换为集合（四舍五入到指定精度）"""
        if len(coords) == 0:
            return set()
        return {(round(x, precision), round(y, precision)) for x, y in coords}
    
    def set_to_coords(coord_set: set) -> np.ndarray:
        """将坐标集合转换为数组"""
        if len(coord_set) == 0:
            return np.empty((0, 2))
        return np.array(list(coord_set))
    
    # 划分域坐标集合
    train_split_set = coords_to_set(train_split_coords)
    test_split_set = coords_to_set(test_split_coords)
    val_split_set = coords_to_set(val_split_coords)
    
    # 另一域坐标集合
    train_other_set = coords_to_set(train_other_coords)
    test_other_set = coords_to_set(test_other_coords)
    val_other_set = coords_to_set(val_other_coords)
    
    # 检查划分域 train 和 test 的重叠（不应该存在）
    split_overlap_coords = set_to_coords(train_split_set & test_split_set)
    if len(split_overlap_coords) > 0:
        warnings.warn(
            f"⚠️ CRITICAL: Training and test observation systems overlap in {split_title_prefix} domain! "
            f"Found {len(split_overlap_coords)} overlapping coordinates. This indicates a split error!",
            UserWarning
        )
    
    # 检查另一域 train 和 test 的重叠（可能存在，是正常的）
    other_overlap_set = train_other_set & test_other_set
    other_overlap_coords = set_to_coords(other_overlap_set)
    n_other_overlap = len(other_overlap_coords)
    
    # 计算重叠统计
    other_train_only = train_other_set - test_other_set
    other_test_only = test_other_set - train_other_set
    
    print(f"  [Overlap Analysis] {other_title_prefix} domain:")
    print(f"    - Train only: {len(other_train_only)}")
    print(f"    - Test only: {len(other_test_only)}")
    print(f"    - Overlapping (shared): {n_other_overlap}")
    
    # 检查划分域 train 和 val 的重叠
    if val_split_set:
        train_val_split_overlap = train_split_set & val_split_set
        if train_val_split_overlap:
            warnings.warn(
                f"WARNING: Training and validation observation systems overlap in {split_title_prefix} domain! "
                f"Found {len(train_val_split_overlap)} overlapping coordinates.",
                UserWarning
            )
    
    # 同时检查ID是否有重合（作为额外检查）
    train_test_id_overlap = train_id_set & test_id_set
    if train_test_id_overlap:
        warnings.warn(
            f"WARNING: Training and test IDs overlap! "
            f"Found {len(train_test_id_overlap)} overlapping IDs: {sorted(list(train_test_id_overlap))[:10]}"
            f"{'...' if len(train_test_id_overlap) > 10 else ''}",
            UserWarning
        )
    
    # ========== 4. 绘图（2x1布局：划分域 + 另一域覆盖）==========
    # 颜色配置（论文友好，高对比度）
    train_color = "#1f77b4"    # 蓝色
    test_color = "#d62728"     # 红色
    val_color = "#ff7f0e"      # 橙色
    overlap_color = "#2ca02c"  # 绿色 - 另一域共享点
    error_color = "#9467bd"    # 紫色 - 划分域重叠（错误）
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    def _scatter_coords(ax, coords: np.ndarray, color: str, marker: str, alpha: float, 
                        size: int, label: str = None, edgecolor: str = "none", zorder: int = 1):
        """绘制散点图"""
        if len(coords) > 0:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=color, alpha=alpha, s=size, marker=marker,
                edgecolors=edgecolor, linewidths=0.5, label=label, zorder=zorder
            )
    
    # ===== 左图：划分域（Train + Test 在同一张图上）=====
    ax = axes[0]
    # 先画 Train，再画 Test，清晰展示不重叠
    _scatter_coords(ax, train_split_coords, train_color, "o", 0.8, 1, 
                   label=f"Train (n={len(train_split_coords)})", zorder=2)
    _scatter_coords(ax, test_split_coords, test_color, "s", 0.8, 1, 
                   label=f"Test (n={len(test_split_coords)})", zorder=3)
    if len(val_split_coords) > 0:
        _scatter_coords(ax, val_split_coords, val_color, "^", 0.7, 35, 
                       label=f"Val (n={len(val_split_coords)})", zorder=4)
    # 标记划分域重叠点（错误情况，用醒目标记）
    if len(split_overlap_coords) > 0:
        _scatter_coords(ax, split_overlap_coords, error_color, "X", 1.0, 150, 
                       label=f"⚠️ Overlap (n={len(split_overlap_coords)})", edgecolor="black", zorder=10)
    
    ax.set_xlabel(split_x_label, fontsize=11)
    ax.set_ylabel(split_y_label, fontsize=11)
    ax.set_title(f"{split_title_prefix} Domain Split", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # ===== 右图：另一域覆盖（Train + Test 在同一张图上）=====
    ax = axes[1]
    # 先画 Train only，再画 Test only，最后画共享点
    train_only_coords = set_to_coords(other_train_only)
    test_only_coords = set_to_coords(other_test_only)
    
    _scatter_coords(ax, train_only_coords, train_color, "o", 0.7, 20, 
                   label=f"Train only (n={len(train_only_coords)})", zorder=2)
    _scatter_coords(ax, test_only_coords, test_color, "s", 0.7, 25, 
                   label=f"Test only (n={len(test_only_coords)})", zorder=3)
    # 共享点用特殊标记
    if len(other_overlap_coords) > 0:
        _scatter_coords(ax, other_overlap_coords, overlap_color, "*", 0.9, 60, 
                       label=f"Shared (n={n_other_overlap})", edgecolor="white", zorder=5)
    
    ax.set_xlabel(other_x_label, fontsize=11)
    ax.set_ylabel(other_y_label, fontsize=11)
    ax.set_title(f"{other_title_prefix} Coverage", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    
    # ===== 总标题 =====
    domain_name = "Shot" if domain == "shot" else "Receiver"
    hr_txt = f" (holdout={holdout_ratio:.0%})" if isinstance(holdout_ratio, (float, int)) else ""
    
    # 简洁的标题
    title_text = f"Geometry Split by {split_title_prefix}{hr_txt} - {mode.capitalize()} Mode"
    if len(split_overlap_coords) > 0:
        title_text += f" | ⚠️ {len(split_overlap_coords)} overlaps in split domain!"
    
    plt.suptitle(title_text, fontsize=13, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"{file_name}_split_visualization_{mode}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] {out_path.name} saved")
    
    # ========== 5. 打印统计信息 ==========
    print(f"  Split domain ({split_title_prefix}): train={len(train_split_coords)}, test={len(test_split_coords)}, val={len(val_split_coords)}")
    print(f"  Other domain ({other_title_prefix}): train_only={len(train_only_coords)}, test_only={len(test_only_coords)}, shared={n_other_overlap}")
    if len(split_overlap_coords) > 0:
        print(f"  ⚠️ CRITICAL: {len(split_overlap_coords)} overlapping points in split domain!")
    if n_other_overlap > 0:
        overlap_ratio = n_other_overlap / max(1, len(train_other_set | test_other_set)) * 100
        print(f"  ℹ️ Shared {other_title_prefix.lower()}s: {n_other_overlap} ({overlap_ratio:.1f}% of total)")


# -----------------------------
# Main split maker
# -----------------------------
def make_geometry_split(
    index_parquet: str,
    out_dir: str,
    domain: str,
    holdout_ratio: float,
    mode: str,
    seed: int,
    val_ratio: float = 0.0,
    max_traces: Optional[int] = None,
) -> None:
    """
    几何 holdout split：按 shot 或 receiver 域进行划分
    - 先对观测系统ID做holdout（uniform/random/cluster）
    - 再映射到 trace_no（道序号），输出 trace 列表
    """
    if domain not in ["shot", "receiver"]:
        raise ValueError(f"domain must be 'shot' or 'receiver', got {domain}")

    if mode not in ["uniform", "random", "cluster", "all"]:
        raise ValueError(f"mode must be one of ['uniform', 'random', 'cluster', 'all'], got {mode}")

    df = pd.read_parquet(index_parquet)
    df = _ensure_trace_no(df)
    file_name = Path(index_parquet).stem

    # max_traces：抽样但不破坏 trace_no 的语义（trace_no 保留原值）
    if max_traces is not None and len(df) > max_traces:
        df = df.sample(n=max_traces, random_state=seed).copy()

    id_col = "shot_id" if domain == "shot" else "receiver_id"
    x_col, y_col = ("sx", "sy") if domain == "shot" else ("gx", "gy")

    # 建立：ID -> 代表坐标（均值）；ID -> trace_no 列表
    id_coords_list: Dict[Any, list] = {}
    id_to_traces: Dict[Any, list] = {}
    for _, row in df.iterrows():
        _id = row[id_col]
        id_coords_list.setdefault(_id, []).append((row[x_col], row[y_col]))
        id_to_traces.setdefault(_id, []).append(int(row["trace_no"]))

    id_coords_dict: Dict[Any, Tuple[float, float]] = {}
    for _id, pts in id_coords_list.items():
        arr = np.asarray(pts, dtype=np.float64)
        id_coords_dict[_id] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    unique_ids = np.array(list(id_coords_dict.keys()))
    coords = np.array([id_coords_dict[_id] for _id in unique_ids], dtype=np.float64)
    id_to_traces_np = {k: np.array(v, dtype=np.int64) for k, v in id_to_traces.items()}

    rng = np.random.Generator(np.random.PCG64(seed))
    modes_to_process = ["uniform", "random", "cluster"] if mode == "all" else [mode]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_mode in modes_to_process:
        if split_mode == "uniform":
            train_ids, test_ids, train_traces, test_traces = _split_uniform(
                unique_ids, coords, holdout_ratio, rng, id_to_traces=id_to_traces_np
            )
        elif split_mode == "random":
            train_ids, test_ids, train_traces, test_traces = _split_random(
                unique_ids, coords, holdout_ratio, rng, id_to_traces=id_to_traces_np
            )
        elif split_mode == "cluster":
            train_ids, test_ids, train_traces, test_traces = _split_cluster(
                unique_ids, coords, holdout_ratio, rng, id_to_traces=id_to_traces_np
            )
        else:
            continue

        # ID 不相交
        tr_id_set = set(train_ids.tolist())
        te_id_set = set(test_ids.tolist())
        if tr_id_set & te_id_set:
            raise ValueError(f"{split_mode}: train_ids and test_ids overlap!")

        # trace 不相交（保证训练完全不用holdout traces）
        tr_trace_set = set(train_traces.tolist())
        te_trace_set = set(test_traces.tolist())
        if tr_trace_set & te_trace_set:
            raise ValueError(f"{split_mode}: train_traces and test_traces overlap! mapping is inconsistent.")

        # val：从 train_ids 中切；然后重新映射得到 train/val traces（避免错位）
        val_ids = np.array([], dtype=train_ids.dtype)
        val_traces = np.array([], dtype=np.int64)
        if val_ratio > 0 and len(train_ids) > 0:
            n_val = int(len(train_ids) * val_ratio)
            if n_val > 0:
                val_idx = rng.choice(len(train_ids), size=n_val, replace=False)
                val_ids = train_ids[val_idx]
                keep = np.setdiff1d(np.arange(len(train_ids)), val_idx)
                train_ids = train_ids[keep]
                train_traces = _ids_to_traces(train_ids, id_to_traces_np)
                val_traces = _ids_to_traces(val_ids, id_to_traces_np)

                # 再检查一遍（trace层）
                if set(train_traces.tolist()) & set(val_traces.tolist()):
                    raise ValueError(f"{split_mode}: train_traces and val_traces overlap!")

        # 输出目录（mode=all 时分子目录）
        split_dir = (out_dir / split_mode) if mode == "all" else out_dir
        split_dir.mkdir(parents=True, exist_ok=True)

        # 保存：ID split（给你的 Dataset 用）
        with open(split_dir / f"{file_name}_train_ids_{holdout_ratio}_{split_mode}.json", "w") as f:
            json.dump(train_ids.tolist(), f)
        with open(split_dir / f"{file_name}_test_ids_{holdout_ratio}_{split_mode}.json", "w") as f:
            json.dump(test_ids.tolist(), f)
        if len(val_ids) > 0:
            with open(split_dir / f"{file_name}_val_ids_{holdout_ratio}_{split_mode}.json", "w") as f:
                json.dump(val_ids.tolist(), f)
        else:
            # 可选：不写 val 文件也行；这里为了“直接替换”更稳，不强制写空文件
            pass

        # 保存：trace split（训练“严格不用holdout”用这个）
        with open(split_dir / f"{file_name}_train_traces.json", "w") as f:
            json.dump(train_traces.tolist(), f)
        with open(split_dir / f"{file_name}_test_traces.json", "w") as f:
            json.dump(test_traces.tolist(), f)
        if len(val_traces) > 0:
            with open(split_dir / f"{file_name}_val_traces.json", "w") as f:
                json.dump(val_traces.tolist(), f)

        # 保存配置
        config = {
            "domain": domain,
            "holdout_ratio": float(holdout_ratio),
            "mode": split_mode,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "max_traces": None if max_traces is None else int(max_traces),
            "n_ids_total": int(len(unique_ids)),
            "n_train_ids": int(len(train_ids)),
            "n_test_ids": int(len(test_ids)),
            "n_val_ids": int(len(val_ids)),
            "n_train_traces": int(len(train_traces)),
            "n_test_traces": int(len(test_traces)),
            "n_val_traces": int(len(val_traces)),
        }
        with open(split_dir / f"{file_name}_split_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 可视化：观测系统（ID级别）
        _visualize_split(
            df=df,
            file_name=file_name,
            train_ids=train_ids,
            test_ids=test_ids,
            val_ids=val_ids,
            id_col=id_col,
            out_dir=split_dir,
            domain=domain,
            mode=split_mode,
            holdout_ratio=holdout_ratio,
        )

    # mode=all：把 random 子目录结果复制到根目录作为默认
    if mode == "all":
        import shutil

        default_dir = out_dir / "random"
        for fname in [
            f"{file_name}_train_ids.json",
            f"{file_name}_test_ids.json",
            f"{file_name}_val_ids.json",
            f"{file_name}_train_traces.json",
            f"{file_name}_test_traces.json",
            f"{file_name}_val_traces.json",
            f"{file_name}_split_config.json",
            f"{file_name}_split_visualization_random.png",
        ]:
            src = default_dir / fname
            if src.exists():
                shutil.copy(src, out_dir / fname)


# -----------------------------
# Optional CLI usage
# -----------------------------
if __name__ == "__main__":
    # 示例（你可以删掉这一段）
    # make_geometry_split(
    #     index_parquet="path/to/index.parquet",
    #     out_dir="path/to/splits",
    #     domain="shot",
    #     holdout_ratio=0.2,
    #     mode="random",
    #     seed=0,
    #     val_ratio=0.1,
    #     max_traces=None,
    # )
    pass
