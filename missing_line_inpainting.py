#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全缺失测线补全模块

对完全缺失的测线（某line_id下所有trace缺失）进行patch级KNN条件推理补全。
"""

from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
from scipy.spatial import KDTree
from scipy.stats import median_abs_deviation
import warnings
import csv
import matplotlib.pyplot as plt


def compute_offset(sx: np.ndarray, sy: np.ndarray, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """
    计算offset（炮检距）
    
    Args:
        sx, sy: 炮点坐标
        rx, ry: 检波点坐标
        
    Returns:
        offset: (n_traces,)
    """
    return np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)


def robust_scale(x: np.ndarray, use_robust: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Robust scaling (median/IQR) 归一化
    
    Args:
        x: 输入数组
        use_robust: 是否使用robust scaling，否则使用min-max
        
    Returns:
        x_scaled: 归一化后的数组
        stats: 统计信息字典 {'median', 'iqr', 'min', 'max'} 或 {'min', 'max'}
    """
    if use_robust:
        median = np.median(x)
        q25 = np.percentile(x, 25)
        q75 = np.percentile(x, 75)
        iqr = q75 - q25
        if iqr < 1e-10:
            iqr = np.std(x) if np.std(x) > 1e-10 else 1.0
        x_scaled = (x - median) / iqr
        stats = {'median': median, 'iqr': iqr, 'min': np.min(x), 'max': np.max(x)}
    else:
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-10:
            x_scaled = np.zeros_like(x)
        else:
            x_scaled = (x - x_min) / (x_max - x_min) * 2 - 1  # 映射到[-1, 1]
        stats = {'min': x_min, 'max': x_max}
    return x_scaled, stats


def robust_unscale(x_scaled: np.ndarray, stats: Dict, use_robust: bool = True) -> np.ndarray:
    """反归一化"""
    if use_robust:
        return x_scaled * stats['iqr'] + stats['median']
    else:
        return (x_scaled + 1) / 2 * (stats['max'] - stats['min']) + stats['min']


def compute_patch_center(sx_patch: np.ndarray, sy_patch: np.ndarray, offset_patch: np.ndarray) -> np.ndarray:
    """
    计算patch的几何中心
    
    Args:
        sx_patch: (patch_trace_size,) 炮点x坐标
        sy_patch: (patch_trace_size,) 炮点y坐标
        offset_patch: (patch_trace_size,) offset
        
    Returns:
        center: (3,) [mean(sx), mean(sy), mean(offset)]
    """
    return np.array([
        np.mean(sx_patch),
        np.mean(sy_patch),
        np.mean(offset_patch)
    ])


def knn_distance(
    center_p: np.ndarray,
    centers_q: np.ndarray,
    sx_stats: Dict,
    sy_stats: Dict,
    offset_stats: Dict,
    use_robust: bool = True
) -> np.ndarray:
    """
    计算KNN距离：d = sqrt((sx_p-sx_q)^2+(sy_p-sy_q)^2) + |offset_p-offset_q|
    
    Args:
        center_p: (3,) 目标patch中心 [sx, sy, offset]
        centers_q: (N_candidates, 3) 候选patch中心
        sx_stats, sy_stats, offset_stats: 归一化统计信息
        use_robust: 是否使用robust scaling
        
    Returns:
        distances: (N_candidates,)
    """
    # 归一化
    sx_p_scaled, _ = robust_scale(np.array([center_p[0]]), use_robust)
    sy_p_scaled, _ = robust_scale(np.array([center_p[1]]), use_robust)
    offset_p_scaled, _ = robust_scale(np.array([center_p[2]]), use_robust)
    
    sx_q_scaled, _ = robust_scale(centers_q[:, 0], use_robust)
    sy_q_scaled, _ = robust_scale(centers_q[:, 1], use_robust)
    offset_q_scaled, _ = robust_scale(centers_q[:, 2], use_robust)
    
    # 使用全局统计进行归一化（更一致）
    if use_robust:
        sx_p_norm = (center_p[0] - sx_stats['median']) / sx_stats['iqr']
        sy_p_norm = (center_p[1] - sy_stats['median']) / sy_stats['iqr']
        offset_p_norm = (center_p[2] - offset_stats['median']) / offset_stats['iqr']
        
        sx_q_norm = (centers_q[:, 0] - sx_stats['median']) / sx_stats['iqr']
        sy_q_norm = (centers_q[:, 1] - sy_stats['median']) / sy_stats['iqr']
        offset_q_norm = (centers_q[:, 2] - offset_stats['median']) / offset_stats['iqr']
    else:
        sx_p_norm = 2 * (center_p[0] - sx_stats['min']) / (sx_stats['max'] - sx_stats['min']) - 1
        sy_p_norm = 2 * (center_p[1] - sy_stats['min']) / (sy_stats['max'] - sy_stats['min']) - 1
        offset_p_norm = 2 * (center_p[2] - offset_stats['min']) / (offset_stats['max'] - offset_stats['min']) - 1
        
        sx_q_norm = 2 * (centers_q[:, 0] - sx_stats['min']) / (sx_stats['max'] - sx_stats['min']) - 1
        sy_q_norm = 2 * (centers_q[:, 1] - sy_stats['min']) / (sy_stats['max'] - sy_stats['min']) - 1
        offset_q_norm = 2 * (centers_q[:, 2] - offset_stats['min']) / (offset_stats['max'] - offset_stats['min']) - 1
    
    # 计算距离
    spatial_dist = np.sqrt((sx_p_norm - sx_q_norm) ** 2 + (sy_p_norm - sy_q_norm) ** 2)
    offset_dist = np.abs(offset_p_norm - offset_q_norm)
    distances = spatial_dist + offset_dist
    
    return distances


def find_fully_missing_lines(
    mask: np.ndarray,
    line_id: np.ndarray
) -> List[int]:
    """
    找出完全缺失的测线（某line_id下所有trace的mask都为0）
    
    Args:
        mask: (n_traces, n_samples) mask数组，>0表示有数据
        line_id: (n_traces,) 每条trace的line_id
        
    Returns:
        missing_lines: 完全缺失的测线ID列表
    """
    unique_lines = np.unique(line_id)
    missing_lines = []
    
    for line in unique_lines:
        idx_line = np.where(line_id == line)[0]
        if len(idx_line) == 0:
            continue
        # 检查该测线所有trace是否都缺失
        mask_line = mask[idx_line, :]
        if np.all(mask_line == 0):
            missing_lines.append(int(line))
    
    return missing_lines

class MissingLineRegular:
    """For dongfang dataset - Missing line inpainting using KNN"""
    def __init__(self, time_ps, trace_ps, bin_size: int = 50, K: Optional[int] = None, sort_by: str = 'offset'):
         self.time_ps = time_ps
         self.trace_ps = trace_ps
         self.sort_by = sort_by
         self.K = K if K is not None else trace_ps  # K defaults to trace_ps
         self.raw_data_dir = Path("./dongfang/raw")
         self.label_cleaned_dir = Path("./dongfang/label/cleaned_data")
         self.mask_dir = Path(f'./dongfang/aligned_raw_data_{bin_size}m')
         pattern_label = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
         pattern_mask = re.compile(r"aligned_raw_data_recl_(\d+)_recn_(\d+).npy")
         label_files = []
         mask_files = []
         for file in self.label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            m = pattern_label.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                label_files.append((recv_line, recv_no, file))

         for file in self.mask_dir.glob("aligned_raw_data_recl_*_recn_*.npy"):
            m = pattern_mask.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                mask_files.append((recv_line, recv_no, file))

         print(f"Total label files: {len(label_files)}")
         print(f"Total mask files: {len(mask_files)}")
         self.stats = self._compute_coord_stats(mask_files, self.mask_dir, use_raw=False)
    def _compute_coord_stats(self, file_list, attrs_dir, use_raw=True):
        sx_all, sy_all, rx_all, ry_all = [], [], [], []
        for recv_line, recv_no, _ in file_list:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            else:
                attrs_fp = attrs_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            if not attrs_fp.exists():
                print(f"[WARN] Attributes file not found: {attrs_fp}")
                continue

            attrs = np.load(attrs_fp, allow_pickle=True)
            if isinstance(attrs, np.ndarray) and attrs.dtype == object:
                attrs = attrs.item()  # dict
            # 如果你的 attrs 不是 dict，请按实际格式调整
            sx_all.append(attrs["shot_x"])
            sy_all.append(attrs["shot_y"])
            rx_all.append(attrs["rec_x"])
            ry_all.append(attrs["rec_y"])

        sx_all = np.concatenate(sx_all)
        sy_all = np.concatenate(sy_all)
        rx_all = np.concatenate(rx_all)
        ry_all = np.concatenate(ry_all)

        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)

        # 用同一套范围（这里用 unique 的 min/max；也可以换成分位数）
        sx_min, sx_max = float(sx_u.min()), float(sx_u.max())
        sy_min, sy_max = float(sy_u.min()), float(sy_u.max())
        rx_min, rx_max = float(rx_u.min()), float(rx_u.max())
        ry_min, ry_max = float(ry_u.min()), float(ry_u.max())
        #print('sx_min, sx_max, sy_min, sy_max, rx_min, rx_max, ry_min, ry_max:',sx_min, sx_max, sy_min, sy_max, rx_min, rx_max, ry_min, ry_max)
        #print('dsx, dsy, drx, dry:',dsx, dsy, drx, dry)
        deltas = {}
        if dsx is not None and (sx_max - sx_min) > 0:
            deltas["sx"] = float((sx_max - sx_min)/(2*dsx))
        if dsy is not None and (sy_max - sy_min) > 0:
            deltas["sy"] = float((sy_max - sy_min)/(2*dsy))
        if drx is not None and (rx_max - rx_min) > 0:
            deltas["rx"] = float((rx_max - rx_min)/(2*drx))
        if dry is not None and (ry_max - ry_min) > 0:
            deltas["ry"] = float((ry_max - ry_min)/(2*dry))

        # fallback：如果某些轴估不出来，就别让它把整体搞崩
        if len(deltas.keys()) == 0:
            self.space_scale = 1.0
        else:
            self.space_scale = deltas 
            
        stats = {
            "sx_min": sx_min, "sx_max": sx_max,
            "sy_min": sy_min, "sy_max": sy_max,
            "rx_min": rx_min, "rx_max": rx_max,
            "ry_min": ry_min, "ry_max": ry_max,
        }
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])
        
        # Compute offset range for normalization
        # Offset = sqrt((rx-sx)^2 + (ry-sy)^2), so max offset is approximately diagonal
        max_offset = np.sqrt((stats["rx_max"] - stats["sx_min"])**2 + (stats["ry_max"] - stats["sy_min"])**2)
        min_offset = 0.0  # Minimum offset is 0
        stats["offset_min"] = min_offset
        stats["offset_max"] = max_offset
        
        return stats
    
    def typical_grid_step(self, arr, eps=1e-9):
        """Estimate typical grid step from array"""
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u
        d = np.diff(u)
        d = d[d > eps]
        if d.size == 0:
            return None, u
        return float(np.median(d)), u
    
    def normalize_patch_data(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Normalize patch data using 99.5 percentile (consistent with patched_dataset5d.py)
        
        Args:
            data: (trace_ps, time_ps) patch data
            
        Returns:
            normalized_data: normalized patch
            threshold: normalization threshold used
        """
        threshold = np.percentile(np.abs(data), 99.5)
        if threshold == 0:
            threshold = 1e-6
        normalized = np.clip(data, -threshold, threshold) / threshold
        return normalized, threshold
    
    def normalize_coords(self, sx: np.ndarray, sy: np.ndarray, 
                        rx: np.ndarray, ry: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize coordinates to [-1, 1] using global stats (consistent with patched_dataset5d.py)
        
        Args:
            sx, sy, rx, ry: coordinate arrays
            
        Returns:
            sx_norm, sy_norm, rx_norm, ry_norm: normalized coordinates
        """
        stats = self.stats
        sx_norm = 2 * (sx - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
        sy_norm = 2 * (sy - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
        rx_norm = 2 * (rx - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
        ry_norm = 2 * (ry - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1
        return sx_norm, sy_norm, rx_norm, ry_norm
    
    def find_knn_traces(self, target_traces_coords: np.ndarray,
                        candidate_traces_coords: np.ndarray,
                        candidate_trace_to_line: np.ndarray,
                        K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find K nearest neighbor traces using KDTree on all traces from candidate lines
        
        Args:
            target_traces_coords: (trace_ps, 3) target patch trace coordinates [sx, sy, offset] (normalized)
            candidate_traces_coords: (N_total_traces, 3) all candidate trace coordinates (normalized)
            candidate_trace_to_line: (N_total_traces,) mapping from trace index to line_id
            K: number of neighbors to find per target trace
            
        Returns:
            knn_trace_indices: (trace_ps, K) indices of K nearest neighbor traces for each target trace
            knn_line_ids: (trace_ps, K) line IDs corresponding to KNN traces
        """
        if len(candidate_traces_coords) == 0:
            return np.array([]), np.array([])
        
        K = min(K, len(candidate_traces_coords))
        tree = KDTree(candidate_traces_coords)
        
        # Query for each target trace
        n_target = len(target_traces_coords)
        knn_trace_indices = np.zeros((n_target, K), dtype=np.int64)
        knn_line_ids = np.zeros((n_target, K), dtype=np.int64)
        
        for i, target_coord in enumerate(target_traces_coords):
            distances, indices = tree.query(target_coord.reshape(1, -1), k=K)
            if K == 1:
                indices = np.array([indices])
            knn_trace_indices[i] = indices[0]
            knn_line_ids[i] = candidate_trace_to_line[indices[0]]
        
        return knn_trace_indices, knn_line_ids
    
    def concatenate_traces(self, knn_traces: List[np.ndarray],
                                     knn_offsets: np.ndarray,
                                     missing_traces: np.ndarray,
                                     missing_offsets: np.ndarray,
                                     knn_sx: Optional[np.ndarray] = None,
                                     knn_sy: Optional[np.ndarray] = None,
                                     knn_rx: Optional[np.ndarray] = None,
                                     knn_ry: Optional[np.ndarray] = None,
                                     missing_sx: Optional[np.ndarray] = None,
                                     missing_sy: Optional[np.ndarray] = None,
                                     missing_rx: Optional[np.ndarray] = None,
                                     missing_ry: Optional[np.ndarray] = None,
                                     sort_by: str = 'offset') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Concatenate KNN traces and missing traces, sorted by offset or shot coordinates
        
        Args:
            knn_traces: List of KNN trace data, each (time_ps,)
            knn_offsets: (n_knn_traces,) offset array for KNN traces
            missing_traces: (trace_ps, time_ps) missing patch data (zeros or placeholder)
            missing_offsets: (trace_ps,) offset array for missing patch
            knn_sx: (n_knn_traces,) shot x coordinates for KNN traces
            knn_sy: (n_knn_traces,) shot y coordinates for KNN traces
            knn_rx: (n_knn_traces,) receiver x coordinates for KNN traces
            knn_ry: (n_knn_traces,) receiver y coordinates for KNN traces
            missing_sx: (trace_ps,) shot x coordinates for missing patch
            missing_sy: (trace_ps,) shot y coordinates for missing patch
            missing_rx: (trace_ps,) receiver x coordinates for missing patch
            missing_ry: (trace_ps,) receiver y coordinates for missing patch
            sort_by: 'offset' or 'shot' - sorting method
            
        Returns:
            concatenated_data: (total_traces, time_ps) concatenated data sorted by offset or shot coordinates
            concatenated_offsets: (total_traces,) sorted offsets
            original_indices: (total_traces,) original indices for restoring order
            concatenated_coords: Dict with keys 'sx', 'sy', 'rx', 'ry' - sorted coordinates
        """
        # Convert missing patch to list of traces
        missing_traces_list = [missing_traces[i] for i in range(missing_traces.shape[0])]
        
        # Combine all traces
        all_traces = knn_traces + missing_traces_list
        all_offsets = np.concatenate([knn_offsets, missing_offsets])
        
        # Combine coordinates if provided
        if knn_sx is not None and missing_sx is not None:
            all_sx = np.concatenate([knn_sx, missing_sx])
            all_sy = np.concatenate([knn_sy, missing_sy])
            all_rx = np.concatenate([knn_rx, missing_rx])
            all_ry = np.concatenate([knn_ry, missing_ry])
        else:
            all_sx = None
            all_sy = None
            all_rx = None
            all_ry = None
        
        # Stack traces into array
        concatenated_data = np.vstack(all_traces)  # (total_traces, time_ps)
        
        # Sort by offset or shot coordinates
        if sort_by == 'shot' and all_sx is not None:
            # Sort by shot x coordinate first, then by shot y coordinate
            sort_indices = np.lexsort((all_sy, all_sx))
        else:
            # Sort by offset (default)
            sort_indices = np.argsort(all_offsets)
        
        sorted_data = concatenated_data[sort_indices]
        sorted_offsets = all_offsets[sort_indices]
        
        # Track original indices for restoration
        original_indices = np.arange(len(all_offsets))[sort_indices]
        
        # Prepare concatenated coordinates
        concatenated_coords = {}
        if all_sx is not None:
            concatenated_coords['sx'] = all_sx[sort_indices]
            concatenated_coords['sy'] = all_sy[sort_indices]
            concatenated_coords['rx'] = all_rx[sort_indices]
            concatenated_coords['ry'] = all_ry[sort_indices]
        
        return sorted_data, sorted_offsets, original_indices, concatenated_coords
    
    def restore_patch_order(self, interpolated_data: np.ndarray,
                           original_indices: np.ndarray,
                           missing_start_idx: int,
                           missing_end_idx: int) -> np.ndarray:
        """
        Restore interpolated missing patch to original order
        
        Args:
            interpolated_data: (total_traces, time_ps) interpolated data
            original_indices: (total_traces,) original sort indices
            missing_start_idx: start index of missing patch in original concatenated array
            missing_end_idx: end index of missing patch in original concatenated array
            
        Returns:
            restored_patch: (trace_ps, time_ps) restored missing patch in original order
        """
        # Restore original order
        restored_data = np.zeros_like(interpolated_data)
        restored_data[original_indices] = interpolated_data
        
        # Extract missing patch
        missing_patch = restored_data[missing_start_idx:missing_end_idx]
        return missing_patch
    
    def build_candidate_traces(self, recv_line: int, recv_no: int) -> Dict:
        """
        Build candidate traces from the entire receiver gather (not divided by lines)
        Use all non-missing traces from the whole receiver for KDTree search
        
        Args:
            recv_line: receiver line number
            recv_no: receiver number
            
        Returns:
            candidate_dict: Dictionary with keys:
                - 'trace_coords': (N_total_traces, 3) all trace coordinates [sx, sy, offset] (normalized)
                - 'trace_to_line': (N_total_traces,) mapping from trace index to line_id
                - 'trace_to_global_idx': (N_total_traces,) mapping from trace index to global trace index
                - 'all_data': (N_total_traces, n_samples) all trace data
                - 'all_offsets': (N_total_traces,) all trace offsets
        """
        rdata_fp = self.raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
        rmask_fp = self.mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
        rattrs_fp = self.mask_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
        
        if not all(f.exists() for f in [rdata_fp, rmask_fp, rattrs_fp]):
            return {
                'trace_coords': np.array([]),
                'trace_to_line': np.array([]),
                'trace_to_global_idx': np.array([]),
                'all_data': np.array([]),
                'all_offsets': np.array([]),
                'all_sx': np.array([]),
                'all_sy': np.array([]),
                'all_rx': np.array([]),
                'all_ry': np.array([])
            }
        
        rdata = np.load(rdata_fp)
        rmask = np.load(rmask_fp)
        rattrs = np.load(rattrs_fp, allow_pickle=True)
        if isinstance(rattrs, np.ndarray) and rattrs.dtype == object:
            rattrs = rattrs.item()
        
        sx_old = rattrs["shot_x"]
        sy_old = rattrs["shot_y"]
        rx_old = rattrs["rec_x"]
        ry_old = rattrs["rec_y"]
        line_id = rattrs["shot_line"]
        
        # Use all traces from the entire receiver (not divided by lines)
        # Only filter out traces that are completely missing
        n_traces = rdata.shape[0]
        all_trace_coords = []
        trace_to_line = []
        trace_to_global_idx = []
        all_data_list = []
        all_offsets_list = []
        
        for i in range(n_traces):
            # Get coordinates for this trace
            sx = sx_old[i]
            sy = sy_old[i]
            rx = rx_old[i]
            ry = ry_old[i]
            
            # Compute offset
            offset = compute_offset(np.array([sx]), np.array([sy]), np.array([rx]), np.array([ry]))[0]
            
            # Normalize coordinates
            sx_norm = 2 * (sx - self.stats["sx_min"]) / (self.stats["sx_max"] - self.stats["sx_min"]) - 1
            sy_norm = 2 * (sy - self.stats["sy_min"]) / (self.stats["sy_max"] - self.stats["sy_min"]) - 1
            offset_range = self.stats["offset_max"] - self.stats["offset_min"]
            if offset_range > 0:
                offset_norm = 2 * (offset - self.stats["offset_min"]) / offset_range - 1
            else:
                offset_norm = 0.0
            
            trace_coord = np.array([sx_norm, sy_norm, offset_norm])
            all_trace_coords.append(trace_coord)
            trace_to_line.append(line_id[i])
            trace_to_global_idx.append(i)
            all_data_list.append(rdata[i])
            all_offsets_list.append(offset)
        
        return {
            'trace_coords': np.array(all_trace_coords) if all_trace_coords else np.array([]),
            'trace_to_line': np.array(trace_to_line) if trace_to_line else np.array([]),
            'trace_to_global_idx': np.array(trace_to_global_idx) if trace_to_global_idx else np.array([]),
            'all_data': np.array(all_data_list) if all_data_list else np.array([]),
            'all_offsets': np.array(all_offsets_list) if all_offsets_list else np.array([]),
            'all_sx': sx_old,
            'all_sy': sy_old,
            'all_rx': rx_old,
            'all_ry': ry_old
        }
    def get_patches(self, recv_line: int, recv_no: int):
        """
        Get patches for missing lines and find KNN patches for inpainting
        
        Args:
            recv_line: receiver line number
            recv_no: receiver number
            
        Returns:
            results: List of dicts containing patch information and KNN indices
        """
        rdata_fp = self.raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
        rmask_fp = self.mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
        rattrs_fp = self.mask_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
        
        if not all(f.exists() for f in [rdata_fp, rmask_fp, rattrs_fp]):
            print(f"[WARN] Files not found for recl_{recv_line}_recn_{recv_no}")
            return []
        
        rdata = np.load(rdata_fp)
        rmask = np.load(rmask_fp)
        rattrs = np.load(rattrs_fp, allow_pickle=True)
        if isinstance(rattrs, np.ndarray) and rattrs.dtype == object:
            rattrs = rattrs.item()
        
        sx_old = rattrs["shot_x"]
        sy_old = rattrs["shot_y"]
        rx_old = rattrs["rec_x"]
        ry_old = rattrs["rec_y"]
        line_id = rattrs["shot_line"]
        
        # Normalize coordinates
        sx_norm, sy_norm, rx_norm, ry_norm = self.normalize_coords(sx_old, sy_old, rx_old, ry_old)
        
        # Get missing lines
        missing_lines = find_fully_missing_lines(rmask, line_id)
        print(f"Missing lines: {missing_lines}")
        
        if len(missing_lines) == 0:
            return []
        
        # Build candidate traces (all traces from non-missing lines)
        print("Building candidate traces...")
        candidate_dict = self.build_candidate_traces(recv_line, recv_no)
        
        if len(candidate_dict['trace_coords']) == 0:
            print("[WARN] No candidate traces found")
            return []
        
        results = []
        print(f"Processing {len(missing_lines)} missing lines...")
        
        for line in missing_lines:
            line_indices = np.where(line_id == line)[0]
            if len(line_indices) < self.trace_ps:
                continue
            
            # Process patches for this missing line
            for t_start in range(0, rmask.shape[1] - self.time_ps + 1, self.time_ps):
                for tr_start in range(0, len(line_indices) - self.trace_ps + 1, self.trace_ps):
                    tr_end = tr_start + self.trace_ps
                    tr_idx = line_indices[tr_start:tr_end]
                    
                    if len(tr_idx) < self.trace_ps:
                        continue
                    
                    # Get coordinates for missing patch
                    patch_sx = sx_norm[tr_idx]
                    patch_sy = sy_norm[tr_idx]
                    patch_rx = rx_norm[tr_idx]
                    patch_ry = ry_norm[tr_idx]
                    
                    patch_sx_ori = sx_old[tr_idx]
                    patch_sy_ori = sy_old[tr_idx]
                    patch_rx_ori = rx_old[tr_idx]
                    patch_ry_ori = ry_old[tr_idx]
                    
                    # Compute offset
                    patch_offset = compute_offset(patch_sx_ori, patch_sy_ori, patch_rx_ori, patch_ry_ori)
                    
                    # Normalize coordinates for each trace in the patch
                    patch_trace_coords = []
                    for sx, sy, rx, ry, offset in zip(patch_sx, patch_sy, patch_rx, patch_ry, patch_offset):
                        offset_range = self.stats["offset_max"] - self.stats["offset_min"]
                        if offset_range > 0:
                            offset_norm = 2 * (offset - self.stats["offset_min"]) / offset_range - 1
                        else:
                            offset_norm = 0.0
                        patch_trace_coords.append([sx, sy, offset_norm])
                    patch_trace_coords = np.array(patch_trace_coords)  # (trace_ps, 3)
                    
                    # Find KNN traces using KDTree on all candidate traces
                    knn_trace_indices, knn_line_ids = self.find_knn_traces(
                        patch_trace_coords.mean(axis=0).reshape(1, -1),
                        candidate_dict['trace_coords'],
                        candidate_dict['trace_to_line'],
                        self.K
                    )
                    
                    # Extract KNN trace data
                    # For each target trace, get K nearest neighbor traces
                    knn_traces_data = []
                    knn_traces_offsets = []
                    knn_traces_sx = []
                    knn_traces_sy = []
                    knn_traces_rx = []
                    knn_traces_ry = []
                    
                    #for i in range(len(patch_trace_coords)):
                        # Get KNN traces for this target trace
                    for k in range(self.K):
                        if k < len(knn_trace_indices[0]):
                            knn_idx = knn_trace_indices[0, k]
                            # Extract trace data for the time window
                            knn_trace_data = candidate_dict['all_data'][knn_idx, t_start:t_start + self.time_ps]
                            knn_trace_offset = candidate_dict['all_offsets'][knn_idx]
                            knn_traces_data.append(knn_trace_data)
                            knn_traces_offsets.append(knn_trace_offset)
                            # Extract coordinates
                            knn_traces_sx.append(candidate_dict['all_sx'][knn_idx])
                            knn_traces_sy.append(candidate_dict['all_sy'][knn_idx])
                            knn_traces_rx.append(candidate_dict['all_rx'][knn_idx])
                            knn_traces_ry.append(candidate_dict['all_ry'][knn_idx])
                    
                    # Create missing patch placeholder (zeros)
                    missing_patch = np.zeros((self.trace_ps, self.time_ps), dtype=np.float32)
                    
                    # Concatenate KNN traces and missing patch, sorted by offset or shot coordinates
                    knn_offsets_array = np.array(knn_traces_offsets) if knn_traces_offsets else np.array([])
                    knn_sx_array = np.array(knn_traces_sx) if knn_traces_sx else None
                    knn_sy_array = np.array(knn_traces_sy) if knn_traces_sy else None
                    knn_rx_array = np.array(knn_traces_rx) if knn_traces_rx else None
                    knn_ry_array = np.array(knn_traces_ry) if knn_traces_ry else None
                    
                    concatenated_patch, concatenated_offsets, original_indices, concatenated_coords = self.concatenate_traces(
                        knn_traces_data,
                        knn_offsets_array,
                        missing_patch,
                        patch_offset,
                        knn_sx=knn_sx_array,
                        knn_sy=knn_sy_array,
                        knn_rx=knn_rx_array,
                        knn_ry=knn_ry_array,
                        missing_sx=patch_sx_ori,
                        missing_sy=patch_sy_ori,
                        missing_rx=patch_rx_ori,
                        missing_ry=patch_ry_ori,
                        sort_by=self.sort_by  # Can be changed to 'shot' for shot coordinate sorting
                    )
                    
                    # Store results
                    results.append({
                        'line_id': line,
                        'time_start': t_start,
                        'trace_indices': tr_idx,
                        'patch_sx': patch_sx_ori,
                        'patch_sy': patch_sy_ori,
                        'patch_rx': patch_rx_ori,
                        'patch_ry': patch_ry_ori,
                        'patch_offset': patch_offset,
                        'concatenated_patch': concatenated_patch,  # (total_traces, time_ps) sorted by offset or shot
                        'concatenated_offsets': concatenated_offsets,
                        'original_indices': original_indices,
                        'concatenated_sx': concatenated_coords.get('sx', None),  # Sorted shot x coordinates
                        'concatenated_sy': concatenated_coords.get('sy', None),  # Sorted shot y coordinates
                        'concatenated_rx': concatenated_coords.get('rx', None),  # Sorted receiver x coordinates
                        'concatenated_ry': concatenated_coords.get('ry', None),  # Sorted receiver y coordinates
                    })
        
        return results


def load_receivers_with_d_missing(csv_path: str) -> List[Tuple[int, int]]:
    """
    Load receivers with D-type missing lines (n_D > 0) from CSV file
    
    Args:
        csv_path: Path to per_receiver CSV file
        
    Returns:
        receivers: List of (recl, recn) tuples
    """
    receivers = []
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"[WARN] CSV file not found: {csv_path}")
        return receivers
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_D = int(float(row.get('n_D', 0) or 0))
                if n_D > 0:
                    recl = int(float(row.get('recl', 0)))
                    recn = int(float(row.get('recn', 0)))
                    receivers.append((recl, recn))
            except (ValueError, KeyError) as e:
                print(f"[WARN] Error parsing row: {e}")
                continue
    
    print(f"Found {len(receivers)} receivers with D-type missing lines")
    return receivers


def test_missing_line_inpainting(
    csv_path: str = "./mask_reports/per_receiver_20260121_231141.csv",
    time_ps: int = 1248,
    trace_ps: int = 64,
    bin_size: int = 50,
    K: Optional[int] = None,
    max_receivers: int = 3,
    max_patches_per_receiver: int = 2
):
    """
    Test missing line inpainting with visualization
    
    Args:
        csv_path: Path to per_receiver CSV file
        time_ps: Time patch size
        trace_ps: Trace patch size
        bin_size: Bin size for data directory
        K: Number of nearest neighbors (default: trace_ps)
        max_receivers: Maximum number of receivers to test
        max_patches_per_receiver: Maximum number of patches per receiver to visualize
    """
    # Load receivers with D-type missing
    receivers = load_receivers_with_d_missing(csv_path)
    
    if len(receivers) == 0:
        print("No receivers with D-type missing lines found")
        return
    
    # Limit number of receivers
    receivers = receivers[:max_receivers]
    
    # Initialize inpainting class
    inpainter = MissingLineRegular(time_ps, trace_ps, bin_size=bin_size, K=K,sort_by='shot')
    
    # Process each receiver
    for idx, (recl, recn) in enumerate(receivers):
        print(f"\n{'='*60}")
        print(f"Processing receiver {idx+1}/{len(receivers)}: recl={recl}, recn={recn}")
        print(f"{'='*60}")
        
        # Get patches
        results = inpainter.get_patches(recl, recn)
        
        if len(results) == 0:
            print(f"No patches found for recl={recl}, recn={recn}")
            continue
        
        # Limit number of patches to visualize
        results = results[:max_patches_per_receiver]
        
        # Visualize each patch
        for patch_idx, result in enumerate(results):
            print(f"\nProcessing patch {patch_idx+1}/{len(results)}")
            
            # Get concatenated patch (already created in get_patches)
            concatenated_patch = result.get('concatenated_patch')
            concatenated_offsets = result.get('concatenated_offsets')
            original_indices = result.get('original_indices')
            
            if concatenated_patch is None or len(concatenated_patch) == 0:
                print("No concatenated patch found")
                continue
            
            # Get missing patch coordinates
            patch_sx = result['patch_sx']
            patch_sy = result['patch_sy']
            patch_rx = result['patch_rx']
            patch_ry = result['patch_ry']
            patch_offset = result['patch_offset']
            
            # Determine missing patch indices in sorted array
            # In concatenate_traces_by_offset, KNN traces come first, then missing traces
            # We need to find where missing traces are in the sorted array
            n_knn = len(concatenated_patch) - trace_ps  # Number of KNN traces
            missing_start_idx_original = n_knn  # Original index where missing patch starts
            missing_end_idx_original = n_knn + trace_ps  # Original index where missing patch ends
            
            # Find sorted positions of missing patch
            missing_mask = (original_indices >= missing_start_idx_original) & (original_indices < missing_end_idx_original)
            missing_sorted_indices = np.where(missing_mask)[0]
            
            if len(missing_sorted_indices) > 0:
                missing_start_idx_sorted = missing_sorted_indices.min()
                missing_end_idx_sorted = missing_sorted_indices.max() + 1
            else:
                missing_start_idx_sorted = 0
                missing_end_idx_sorted = trace_ps
            
            # Use restore_patch_order to separate KNN and missing parts
            # First, restore the full concatenated patch to original order
            # original_indices[i] tells us which original position the sorted position i corresponds to
            restored_full = np.zeros_like(concatenated_patch)
            restored_full[original_indices] = concatenated_patch
            
            # Extract KNN part (first n_knn traces in original order)
            knn_part = restored_full[:n_knn] if n_knn > 0 else np.array([])
            
            # Extract missing part using restore_patch_order
            # This function restores the interpolated data to original order and extracts missing patch
            missing_part = inpainter.restore_patch_order(
                concatenated_patch,
                original_indices,
                missing_start_idx_original,
                missing_end_idx_original
            )
            
            # Visualize
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Receiver: recl={recl}, recn={recn}, Line={result["line_id"]}, Patch={patch_idx+1}', 
                        fontsize=14, fontweight='bold')
            
            # Plot 1: Concatenated patch (sorted by offset)
            ax = axes[0, 0]
            std_val = np.std(concatenated_patch)
            vmin, vmax = -3 * std_val, 3 * std_val
            im1 = ax.imshow(concatenated_patch.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, origin='upper')
            # Mark missing patch region
            if len(missing_sorted_indices) > 0:
                ax.axvline(x=missing_start_idx_sorted, color='r', linestyle='--', linewidth=2, label='Missing Start')
                ax.axvline(x=missing_end_idx_sorted-1, color='r', linestyle='--', linewidth=2, label='Missing End')
            ax.set_title(f'Concatenated Patch (Sorted by Offset, std={std_val:.4f})', fontsize=12)
            ax.set_xlabel('Trace Index (sorted)')
            ax.set_ylabel('Time Sample')
            ax.legend()
            plt.colorbar(im1, ax=ax, label='Amplitude')
            
            # Plot 2: KNN traces part (restored to original order)
            ax = axes[0, 1]
            if len(knn_part) > 0:
                std_val = np.std(knn_part)
                vmin, vmax = -3 * std_val, 3 * std_val
                im2 = ax.imshow(knn_part.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, origin='upper')
                ax.set_title(f'KNN Traces (Restored, std={std_val:.4f})', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No KNN traces', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('KNN Traces (Restored)', fontsize=12)
            ax.set_xlabel('Trace Index')
            ax.set_ylabel('Time Sample')
            if len(knn_part) > 0:
                plt.colorbar(im2, ax=ax, label='Amplitude')
            
            # Plot 3: Missing patch (restored to original order)
            ax = axes[1, 0]
            std_val = np.std(missing_part) if np.std(missing_part) > 0 else 1.0
            vmin, vmax = -3 * std_val, 3 * std_val
            im3 = ax.imshow(missing_part.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, origin='upper')
            ax.set_title(f'Missing Patch (Restored, std={std_val:.4f})', fontsize=12)
            ax.set_xlabel('Trace Index')
            ax.set_ylabel('Time Sample')
            plt.colorbar(im3, ax=ax, label='Amplitude')
            
            # Plot 4: Offset distribution
            ax = axes[1, 1]
            ax.plot(concatenated_offsets, 'b-', linewidth=2, label='Sorted Offsets')
            if len(missing_sorted_indices) > 0:
                ax.axvspan(missing_start_idx_sorted, missing_end_idx_sorted-1, 
                          alpha=0.3, color='red', label='Missing Patch Region')
            ax.set_title('Offset Distribution (Sorted)', fontsize=12)
            ax.set_xlabel('Trace Index (sorted)')
            ax.set_ylabel('Offset')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_dir = Path("./missing_line_test_results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"recl_{recl}_recn_{recn}_line_{result['line_id']}_patch_{patch_idx}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
            plt.close()
            
            # Print statistics
            print(f"  - Line ID: {result['line_id']}")
            print(f"  - KNN traces found: {n_knn}")
            print(f"  - Concatenated patch shape: {concatenated_patch.shape}")
            print(f"  - Missing patch range (original): [{missing_start_idx_original}, {missing_end_idx_original})")
            print(f"  - Missing patch range (sorted): [{missing_start_idx_sorted}, {missing_end_idx_sorted})")
            print(f"  - Offset range: [{patch_offset.min():.2f}, {patch_offset.max():.2f}]")
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run test
    test_missing_line_inpainting(
        csv_path="./mask_reports/per_receiver_20260121_231141.csv",
        time_ps=1248,
        trace_ps=16,
        bin_size=50,
        K=32,  # Will default to trace_ps
        max_receivers=1,
        max_patches_per_receiver=2
    )


