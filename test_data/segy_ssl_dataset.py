from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import pandas as pd
import segyio
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _augment_coords(rx, ry, sx, sy, jitter=0.05, rot_scale=True, center_prob=0.5):
    """坐标增强：旋转+缩放+随机中心化 + 可选 jitter"""
    rx = rx.copy()
    ry = ry.copy()
    sx = sx.copy()
    sy = sy.copy()

    # 轻微 jitter
    rx += np.random.uniform(-jitter, jitter, size=rx.shape)
    ry += np.random.uniform(-jitter, jitter, size=ry.shape)
    sx += np.random.uniform(-jitter, jitter, size=sx.shape)
    sy += np.random.uniform(-jitter, jitter, size=sy.shape)

    if rot_scale:
        # 随机中心化
        if np.random.rand() < center_prob:
            if np.random.rand() < 0.5:
                dx, dy = np.random.choice(rx), np.random.choice(ry)
            else:
                dx, dy = np.random.choice(sx), np.random.choice(sy)
            rx -= dx
            ry -= dy
            sx -= dx
            sy -= dy

        # 旋转
        theta = np.random.rand() * 2.0 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        rx_, ry_ = rx*c - ry*s, rx*s + ry*c
        sx_, sy_ = sx*c - sy*s, sx*s + sy*c
        rx, ry, sx, sy = rx_, ry_, sx_, sy_

        # 缩放
        scale = np.random.uniform(0.8, 1.2)
        rx *= scale
        ry *= scale
        sx *= scale
        sy *= scale

    # clip
    rx = np.clip(rx, -1.5, 1.5)
    ry = np.clip(ry, -1.5, 1.5)
    sx = np.clip(sx, -1.5, 1.5)
    sy = np.clip(sy, -1.5, 1.5)
    return rx, ry, sx, sy


def normalize_clip(data):
    """归一化并 clip"""
    threshold = np.percentile(np.abs(data), 99.5)
    if threshold == 0:
        threshold = 1e-6
    data = np.clip(data, -threshold, threshold)
    data = data / threshold
    return data


def sample_missing_ratio(a=2.0, b=5.0, min_r=0.4, max_r=0.6):
    """采样缺失率"""
    r = np.random.beta(a, b)
    return min_r + (max_r - min_r) * r


def apply_random_missing(traces, missing_ratio):
    """随机道掩码"""
    n_traces, n_samples = traces.shape
    trace_mask = np.random.choice(
        [0, 1], size=(n_traces, 1),
        p=[missing_ratio, 1 - missing_ratio], replace=True
    )
    mask = np.ones((n_traces, n_samples), dtype=np.float32) * trace_mask
    return traces * mask, mask


def apply_block_missing(traces):
    """连续道掩码"""
    n_traces, n_samples = traces.shape
    missing_ratio = sample_missing_ratio(a=2.0, b=5.0, min_r=0.2, max_r=0.4)
    mask = np.ones((n_traces, n_samples), dtype=np.float32)
    n_missing = int(n_traces * missing_ratio)
    if n_missing > 0:
        start = np.random.randint(0, max(1, n_traces - n_missing))
        mask[start:start + n_missing, :] = 0.0
    return traces * mask, mask


def apply_mixed_mask(traces, missing_ratio, block_prob=0.4):
    """混合缺失模式"""
    if np.random.rand() < block_prob:
        return apply_block_missing(traces)
    else:
        return apply_random_missing(traces, missing_ratio)


class SegySSLConfig:
    """SEGY SSL 数据集配置类"""
    
    def __init__(
        self,
        segy_path: str,
        index_parquet: str,
        split_dir: str,
        domain: str = 'shot',
        split: str = 'train',
        patch_mode: str = 'patch',
        time_window: Optional[Tuple[int, int]] = None,
        spatial_window: Optional[Tuple[int, int]] = None,
        r_min: float = 0.5,
        r_max: float = 0.8,
        mask_type_probs: Optional[Dict[str, float]] = None,
        normalize: str = 'per_patch',
        seed: Optional[int] = None,
        test_mode: str = 'gather_based',
        dt_ms: Optional[float] = None,
        t0_ms: float = 0.0,
        coord_augment: bool = True,
        n_test_lines: Optional[int] = None,
        maxRD: Optional[float] = None,
        maxSD: Optional[float] = None,
        useX: bool = True,
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        time_skip: int = 0,
        time_bins:int = 2,
        **kwargs
    ):
        """
        Args:
            segy_path: SEGY 文件路径
            index_parquet: trace 索引 parquet 文件
            split_dir: split 文件目录（包含 train_ids.json, test_ids.json 等）
            domain: 'shot' 或 'receiver'
            split: 'train', 'val', 或 'test'
            patch_mode: 'patch' 或 'trace'
            time_window: (t_start, t_end) 时间窗口，None 表示使用全部
            spatial_window: (n_traces, ) 空间窗口大小（对于 patch 模式）
            r_min, r_max: 训练掩码缺失率范围
            mask_type_probs: 掩码类型概率，例如 {"random": 0.5, "continuous": 0.5}
            normalize: 'per_patch', 'global', 或 'none'
            seed: 随机种子
            test_mode: 'gather_based', 'independent_traces', 或 'line_based'
            dt_ms: 采样间隔（毫秒），None 表示从 SEGY 文件读取
            t0_ms: 起始时间（毫秒）
            coord_augment: 是否进行坐标增强（训练时）
            n_test_lines: 测试测线数量（仅用于 test_mode='line_based'）
            maxRD: receiver 距离阈值，用于判断测线变化（仅用于 test_mode='line_based'）
            maxSD: source 距离阈值，用于判断测线变化（仅用于 test_mode='line_based'）
            useX: 是否使用 xline 判断测线变化（仅用于 test_mode='line_based'）
        """
        self.segy_path = segy_path
        self.index_parquet = index_parquet
        self.split_dir = split_dir
        self.domain = domain
        self.split = split
        self.patch_mode = patch_mode
        self.time_window = time_window
        self.spatial_window = spatial_window
        self.r_min = r_min
        self.r_max = r_max
        self.mask_type_probs = mask_type_probs or {"random": 0.8, "continuous": 0.2}
        self.normalize = normalize
        self.seed = seed
        self.test_mode = test_mode
        self.dt_ms = dt_ms
        self.t0_ms = t0_ms
        self.coord_augment = coord_augment
        self.n_test_lines = n_test_lines
        self.maxRD = maxRD
        self.maxSD = maxSD
        self.useX = useX
        self.missing_mode = missing_mode
        self.missing_ratio = missing_ratio
        self.time_skip = time_skip
        self.time_bins = time_bins
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于传递给 Dataset"""
        return {
            'segy_path': self.segy_path,
            'index_parquet': self.index_parquet,
            'split_dir': self.split_dir,
            'domain': self.domain,
            'split': self.split,
            'patch_mode': self.patch_mode,
            'time_window': self.time_window,
            'spatial_window': self.spatial_window,
            'r_min': self.r_min,
            'r_max': self.r_max,
            'mask_type_probs': self.mask_type_probs,
            'normalize': self.normalize,
            'seed': self.seed,
            'test_mode': self.test_mode,
            'dt_ms': self.dt_ms,
            't0_ms': self.t0_ms,
            'coord_augment': self.coord_augment,
            'n_test_lines': self.n_test_lines,
            'maxRD': self.maxRD,
            'maxSD': self.maxSD,
            'useX': self.useX,
            'missing_mode': self.missing_mode,
            'missing_ratio': self.missing_ratio,
            'time_skip': self.time_skip,
            'time_bins': self.time_bins,
        }
    
    def create_dataset(self) -> 'SegyGeometrySSLDataset':
        """创建数据集实例"""
        return SegyGeometrySSLDataset(**self.to_dict())
    
    @classmethod
    def C3NA(
        cls,
        start:int,
        end:int,
        split: str = 'train',
        domain: str = 'receiver',
        spatial_window: Tuple[int, ...] = (128,),
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        time_skip: int = 1,
        time_bins:int = 1,
        **kwargs
    ) -> 'SegySSLConfig':
        """C3NA 数据集配置：SEG_C3NA_ffid_1201-2400.sgy"""
        base_dir = '/NAS/czt/mount/Seis_DiT/segy_ssl_parquet'
        return cls(
            segy_path=f"/home/czt/seismic_ddpm/Seis_DiT/data/C3/SEG_C3NA_ffid_{start}-{end}.sgy",
            #segy_path=f"/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_{start}-{end}.sgy",
            index_parquet=f'{base_dir}/SEG_C3NA_ffid_{start}-{end}.parquet',
            split_dir=f'{base_dir}/splits',
            domain=domain,
            split=split,
            spatial_window=spatial_window,
            missing_mode=missing_mode,
            missing_ratio=missing_ratio,
            time_skip=time_skip,
            time_bins=time_bins,
            **kwargs
        )
    
    @classmethod
    def xbfy_006_3a3_nucns_3a2_data_DX004_p2(
        cls,
        split: str = 'train',
        domain: str = 'receiver',
        spatial_window: Tuple[int, ...] = (128,),
        test_mode: str = 'line_based',
        n_test_lines: int = 1,
        maxRD: float = 1000.0,
        maxSD: float = 20.0,
        useX: bool = True,
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        time_skip: int = 3,
        time_bins:int = 2,
        **kwargs
    ) -> 'SegySSLConfig':    
        base_dir = '/NAS/czt/mount/Seis_DiT/segy_ssl_parquet'
        return cls(
            segy_path=f"/NAS/data/data/jiangyr/segy/006_3a3_nucns_3a2_data_DX004_p2.sgy",
            index_parquet=f'{base_dir}/006_3a3_nucns_3a2_data_DX004_p2.parquet',
            split_dir=f'{base_dir}/splits',
            domain=domain,
            split=split,
            spatial_window=spatial_window,
            test_mode=test_mode,
            n_test_lines=n_test_lines,
            maxRD=maxRD,
            maxSD=maxSD,
            useX=useX,
            missing_mode=missing_mode,
            missing_ratio=missing_ratio,
            time_skip=time_skip,
            time_bins=time_bins,
            **kwargs
        )
    
    @classmethod
    def compute_global_min_trace_count(
        cls,
        ranges: List[Tuple[int, int]],
        domain: str = 'receiver',
        splits: List[str] = ['train', 'val',]
    ) -> int:
        """
        计算所有数据范围、所有 split 的全局最小 trace 数量
        
        Args:
            ranges: 数据范围列表，例如 [(1201, 2400), (2401, 3600)]
            domain: 'shot' 或 'receiver'
            splits: 要检查的 split 列表，例如 ['train', 'val', 'test']
        
        Returns:
            全局最小 trace 数量
        """
        base_dir = '/NAS/czt/mount/Seis_DiT/segy_ssl_parquet'
        split_dir = Path(f'{base_dir}/splits')
        
        # 划分维度
        split_id_col = 'shot_id' if domain == 'shot' else 'receiver_id'
        # 组织 gather 的维度（与划分维度相反）
        gather_id_col = 'receiver_id' if domain == 'shot' else 'shot_id'
        
        all_trace_counts = []  # 收集所有 gather 的 trace 数量
        
        for start, end in ranges:
            index_parquet = f'{base_dir}/SEG_C3NA_ffid_{start}-{end}.parquet'
            if not Path(index_parquet).exists():
                print(f"Warning: {index_parquet} not found, skipping...")
                continue
            
            # 读取索引
            index_df = pd.read_parquet(index_parquet)
            file_name = index_parquet.split('/')[-1].split('.')[0]
            
            # 读取所有 split IDs
            split_ids_dict = {}
            for split in splits:
                split_file = split_dir / f'{file_name}_{split}_ids.json'
                if split_file.exists():
                    with open(split_file, 'r') as f:
                        split_ids_dict[split] = set(json.load(f))
            
            # 对于每个 split，统计所有 gather 的 trace 数量
            for split, split_ids in split_ids_dict.items():
                # 按 gather_id_col 组织 gather
                gather_dict = {}
                for trace_idx in index_df.index:
                    if index_df.loc[trace_idx, split_id_col] in split_ids:
                        gather_id = index_df.loc[trace_idx, gather_id_col]
                        if gather_id not in gather_dict:
                            gather_dict[gather_id] = []
                        gather_dict[gather_id].append(trace_idx)
                
                # 收集每个 gather 的 trace 数量
                for traces in gather_dict.values():
                    all_trace_counts.append(len(traces))
        
        if len(all_trace_counts) == 0:
            print("Warning: No valid data found, returning default min_trace_count=32")
            return 32
        
        global_min = min(all_trace_counts)
        print(f"Global min_trace_count computed: {global_min} (from {len(all_trace_counts)} gathers across all ranges and splits)")
        return global_min
    
    @classmethod
    def create_C3NA_datasets(
        cls,
        train_ranges: List[Tuple[int, int]],
        val_ranges: List[Tuple[int, int]],
        domain: str = 'receiver',
        spatial_window: Tuple[int, ...] = (128,),
        train_split: str = 'train',
        val_split: str = 'test',
        auto_patch_size: bool = True,
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        **kwargs
    ) -> Tuple[ConcatDataset, ConcatDataset]:
        """
        创建合并的训练集和测试集
        
        Args:
            train_ranges: 训练集数据范围列表，例如 [(1201, 2400), (2401, 3600)]
            val_ranges: 验证集数据范围列表，例如 [(3601, 4781)]
            domain: 'shot' 或 'receiver'
            spatial_window: 空间窗口大小（如果 auto_patch_size=True，会被自动调整）
            train_split: 训练集使用的 split ('train' 或 'val')
            val_split: 验证集使用的 split ('train', 'val' 或 'test')
            auto_patch_size: 是否自动根据所有数据计算全局最小 patch_size
            **kwargs: 其他配置参数
        
        Returns:
            (train_dataset, val_dataset): 合并后的训练集和验证集
        """
        # 如果启用自动 patch_size，计算全局最小值
        if auto_patch_size:
            all_ranges = train_ranges + val_ranges
            global_min_trace_count = cls.compute_global_min_trace_count(
                ranges=all_ranges,
                domain=domain,
                splits=['train', 'val', 'test']
            )
            
            # 如果用户指定的 patch_size 大于全局最小值，使用全局最小值
            if spatial_window[0] > global_min_trace_count:
                print(f"Auto-adjusting patch_size: {spatial_window[0]} -> {global_min_trace_count}")
                spatial_window = (global_min_trace_count,)
            else:
                print(f"Using specified patch_size: {spatial_window[0]} (global_min={global_min_trace_count})")
        
        train_datasets = []
        for start, end in train_ranges:
            config = cls.C3NA(
                start=start,
                end=end,
                split=train_split,
                domain=domain,
                spatial_window=spatial_window,
                missing_mode=missing_mode,
                missing_ratio=missing_ratio,
                **kwargs
            )
            train_datasets.append(config.create_dataset())
        
        val_datasets = []
        for start, end in val_ranges:
            config = cls.C3NA(
                start=start,
                end=end,
                split=val_split,
                domain=domain,
                spatial_window=spatial_window,
                missing_mode=missing_mode,
                missing_ratio=missing_ratio,
                **kwargs
            )
            val_datasets.append(config.create_dataset())
        
        # 合并数据集
        train_concat = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_concat = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

        # 兼容下游代码：很多训练脚本直接访问 dataset.p_scale
        # 当返回的是 ConcatDataset 时，手动把第一个子数据集的 p_scale 挂到 concat 上
        try:
            from torch.utils.data import ConcatDataset as _TorchConcatDataset
        except Exception:
            _TorchConcatDataset = ConcatDataset

        if isinstance(train_concat, _TorchConcatDataset) and len(train_datasets) > 0:
            base_ds = train_datasets[0]
            if hasattr(base_ds, "p_scale"):
                train_concat.p_scale = base_ds.p_scale

        if isinstance(val_concat, _TorchConcatDataset) and len(val_datasets) > 0:
            base_ds = val_datasets[0]
            if hasattr(base_ds, "p_scale"):
                val_concat.p_scale = base_ds.p_scale

        return train_concat, val_concat
    @classmethod
    def create_xbfy_datasets(
        cls,
        split: str = 'train',
        domain: str = 'receiver',
        spatial_window: Tuple[int, ...] = (128,),
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        time_skip: int = 0,
        time_bins:int = 2,
        **kwargs
    ) -> Tuple[ConcatDataset, ConcatDataset]:
        config = cls.xbfy_006_3a3_nucns_3a2_data_DX004_p2(
            split=split,
            domain=domain,
            spatial_window=spatial_window,
            test_mode='line_based',
            n_test_lines=2,
            maxRD=1000.0,
            maxSD=20.0,
            useX=True,
            missing_mode=missing_mode,
            missing_ratio=missing_ratio,
            time_skip=time_skip,
            time_bins=time_bins,
            **kwargs
        )
        return config.create_dataset()
        
    
    @staticmethod
    def plot_coverage_heatmap(
        index_parquet: str,
        split_dir: Optional[str] = None,
        split: Optional[str] = None,
        domain: str = 'shot',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        grid_size: int = 100,
        cmap: str = 'YlOrRd'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        统计并绘制每炮（或每个receiver）的覆盖热力图
        
        Args:
            index_parquet: 索引 parquet 文件路径
            split_dir: split 文件目录，如果提供则只统计指定 split 的数据
            split: split 名称 ('train', 'val', 'test')，如果提供则只统计该 split
            domain: 'shot' 或 'receiver'
                - 'shot': 统计每个 shot 覆盖了多少个 receiver（在空间上）
                - 'receiver': 统计每个 receiver 接收了多少个 shot（在空间上）
            output_path: 输出图片路径，如果为 None 则不保存
            figsize: 图片大小
            dpi: 图片分辨率
            grid_size: 网格大小（用于插值）
            cmap: 颜色映射
        
        Returns:
            (coverage_grid, x_grid, y_grid): 覆盖度网格、X坐标网格、Y坐标网格
        """
        # 读取索引
        index_df = pd.read_parquet(index_parquet)
        
        # 如果指定了 split，只使用该 split 的数据
        if split_dir is not None and split is not None:
            split_dir = Path(split_dir)
            file_name = Path(index_parquet).stem
            split_file = split_dir / f'{file_name}_{split}_ids.json'
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_ids = set(json.load(f))
                if domain == 'shot':
                    index_df = index_df[index_df['shot_id'].isin(split_ids)]
                else:  # receiver
                    index_df = index_df[index_df['receiver_id'].isin(split_ids)]
        
        # 检查数据是否为空
        if len(index_df) == 0:
            raise ValueError(
                f"No data found for domain='{domain}', split='{split}'. "
                f"Please check if the split file exists and contains valid IDs."
            )
        
        # 根据 domain 选择统计维度
        if domain == 'shot':
            # 统计每个 shot 覆盖了多少个 receiver
            id_col = 'shot_id'
            x_col = 'sx'
            y_col = 'sy'
            title = f'Shot Coverage Heatmap ({split if split else "All"})'
            xlabel = 'Source X'
            ylabel = 'Source Y'
            # 统计每个 shot 的唯一 receiver 数量
            coverage_stats = index_df.groupby(id_col).agg({
                'receiver_id': 'nunique',  # 唯一 receiver 数量
                x_col: 'first',
                y_col: 'first'
            }).reset_index()
            coverage_stats.columns = [id_col, 'coverage', x_col, y_col]
        else:  # receiver
            # 统计每个 receiver 接收了多少个 shot
            id_col = 'receiver_id'
            x_col = 'gx'
            y_col = 'gy'
            title = f'Receiver Coverage Heatmap ({split if split else "All"})'
            xlabel = 'Receiver X'
            ylabel = 'Receiver Y'
            # 统计每个 receiver 的唯一 shot 数量
            coverage_stats = index_df.groupby(id_col).agg({
                'shot_id': 'nunique',  # 唯一 shot 数量
                x_col: 'first',
                y_col: 'first'
            }).reset_index()
            coverage_stats.columns = [id_col, 'coverage', x_col, y_col]
        
        # 检查统计结果是否为空
        if len(coverage_stats) == 0:
            raise ValueError(
                f"No coverage statistics found for domain='{domain}', split='{split}'. "
                f"This may happen if the split IDs don't match any data in the index."
            )
        
        # 提取坐标和覆盖度
        x_coords = coverage_stats[x_col].values
        y_coords = coverage_stats[y_col].values
        coverage_values = coverage_stats['coverage'].values
        
        # 检查坐标是否有效（非空且非NaN）
        valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
        if not np.any(valid_mask):
            raise ValueError(
                f"No valid coordinates found for domain='{domain}', split='{split}'. "
                f"All coordinates are NaN or invalid."
            )
        
        # 只使用有效坐标
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        coverage_values = coverage_values[valid_mask]
        
        # 创建网格用于插值
        if len(x_coords) == 0:
            raise ValueError(
                f"Empty coordinate arrays after filtering. "
                f"domain='{domain}', split='{split}'"
            )
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # 扩展边界（5%）
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.05
        x_max += x_range * 0.05
        y_min -= y_range * 0.05
        y_max += y_range * 0.05
        
        # 创建网格
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # 使用 griddata 进行插值
        from scipy.interpolate import griddata
        
        # 使用 griddata 进行插值
        coverage_grid = griddata(
            (x_coords, y_coords),
            coverage_values,
            (X_grid, Y_grid),
            method='linear',
            fill_value=0
        )
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 绘制热力图
        im = ax.imshow(
            coverage_grid,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear'
        )
        
        # 添加散点图显示实际位置
        scatter = ax.scatter(
            x_coords, y_coords,
            c=coverage_values,
            s=20,
            cmap=cmap,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.7,
            vmin=coverage_values.min(),
            vmax=coverage_values.max()
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Count', rotation=270, labelpad=20)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加统计信息
        stats_text = (
            f'Min: {coverage_values.min()}\n'
            f'Max: {coverage_values.max()}\n'
            f'Mean: {coverage_values.mean():.1f}\n'
            f'Median: {np.median(coverage_values):.1f}\n'
            f'Total {domain}s: {len(coverage_stats)}'
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # 保存图片
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Coverage heatmap saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return coverage_grid, X_grid, Y_grid
    
    def plot_coverage(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        便捷方法：使用当前配置绘制覆盖热力图
        
        Args:
            **kwargs: 传递给 plot_coverage_heatmap 的参数
        
        Returns:
            (coverage_grid, x_grid, y_grid): 覆盖度网格、X坐标网格、Y坐标网格
        """
        return self.plot_coverage_heatmap(
            index_parquet=self.index_parquet,
            split_dir=self.split_dir if 'split_dir' not in kwargs else kwargs.pop('split_dir'),
            split=self.split if 'split' not in kwargs else kwargs.pop('split'),
            domain=self.domain if 'domain' not in kwargs else kwargs.pop('domain'),
            **kwargs
        )


class SegyGeometrySSLDataset(Dataset):
    """
    自监督几何外推数据集
    
    训练模式：从 train_geometry_set 中采样，应用随机掩码 M_rand
    测试模式：输入来自 train_geometry_set，目标是重建 test_geometry_set
    """
    
    def __init__(
        self,
        segy_path: str,
        index_parquet: str,
        split_dir: str,
        domain: str = 'shot',
        split: str = 'train',
        patch_mode: str = 'patch',
        time_window: Optional[Tuple[int, int]] = None,
        spatial_window: Optional[Tuple[int, int]] = None,
        r_min: float = 0.5,
        r_max: float = 0.8,
        mask_type_probs: Optional[Dict[str, float]] = None,
        normalize: str = 'per_patch',
        seed: Optional[int] = None,
        test_mode: str = 'gather_based',
        dt_ms: float = 4.0,
        t0_ms: float = 0.0,
        coord_augment: bool = True,
        n_test_lines: Optional[int] = None,
        maxRD: Optional[float] = 1000.0,
        maxSD: Optional[float] = 20.0,
        useX: bool = True,
        missing_mode: str = 'random',
        missing_ratio: float = 0.5,
        time_skip: int = 0,
        time_bins:int = 2,
    ):
        """
        Args:
            segy_path: SEGY 文件路径
            index_parquet: trace 索引 parquet 文件
            split_dir: split 文件目录（包含 train_ids.json, test_ids.json 等）
            domain: 'shot' 或 'receiver'
            split: 'train', 'val', 或 'test'
            patch_mode: 'patch' 或 'trace'
            time_window: (t_start, t_end) 时间窗口，None 表示使用全部
            spatial_window: (n_traces, ) 空间窗口大小（对于 patch 模式）
            r_min, r_max: 训练掩码缺失率范围
            mask_type_probs: 掩码类型概率，例如 {"random": 0.5, "continuous": 0.5}
            normalize: 'per_patch', 'global', 或 'none'
            seed: 随机种子
            test_mode: 'gather_based', 'independent_traces', 或 'line_based'
            dt_ms: 采样间隔（毫秒）
            t0_ms: 起始时间（毫秒）
            coord_augment: 是否进行坐标增强（训练时）
            n_test_lines: 测试测线数量（仅用于 test_mode='line_based'）
            maxRD: receiver 距离阈值，用于判断测线变化（仅用于 test_mode='line_based'）
            maxSD: source 距离阈值，用于判断测线变化（仅用于 test_mode='line_based'）
            useX: 是否使用 xline 判断测线变化（仅用于 test_mode='line_based'）
            missing_mode: 缺失模式，例如 'random' 或 'cluster'
            missing_ratio: 缺失比例
            time_skip: 时间跳过步长
        """
        self.segy_path = segy_path
        self.domain = domain
        self.split = split
        self.patch_mode = patch_mode
        self.time_window = time_window
        self.spatial_window = spatial_window
        self.r_min = r_min
        self.r_max = r_max
        self.mask_type_probs = mask_type_probs or {"random": 0.8, "continuous": 0.2}
        self.normalize = normalize
        self.test_mode = test_mode
        self.dt_ms = dt_ms
        self.t0_ms = t0_ms
        self.coord_augment = coord_augment and (split in ['train', 'val'])
        self.n_test_lines = n_test_lines
        self.maxRD = maxRD
        self.maxSD = maxSD
        self.useX = useX
        self.time_skip = time_skip
        self.time_bins = time_bins
        self.p_scale = None
        # 读取索引
        self.index_df = pd.read_parquet(index_parquet)
        file_name = index_parquet.split('/')[-1].split('.')[0]
        #print(self.index_df)
        # 读取 split IDs
        split_dir = Path(split_dir)
        with open(split_dir / f'{file_name}_{split}_ids_{missing_ratio}_{missing_mode}.json', 'r') as f:
            self.split_ids = set(json.load(f))
        if split == 'test':
            with open(split_dir / f'{file_name}_train_ids_{missing_ratio}_{missing_mode}.json', 'r') as f:
                self.train_ids = set(json.load(f))
        else:
            self.train_ids = None
        self.coord_stats = self._compute_coord_stats()
        # 使用根据坐标统计计算得到的 space_scale 作为 RoPE 的 p_scale
        # 这样后续模型可以通过 dataset.p_scale 统一获取空间尺度
        if hasattr(self, "space_scale"):
            self.p_scale = self.space_scale
        with segyio.open(segy_path, ignore_geometry=True, mode='r') as f:
            self.ns = len(f.trace[0]) if f.tracecount > 0 else 0
            self.nt = self.ns
            header0 = f.header[0]
            dt_us = header0.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL, 0)
            if dt_us > 0:
                self.dt_ms = dt_us / 1000.0  # 微秒转毫秒
        if time_window is None:
            self.t_start, self.t_end = 0, self.nt
        else:
            self.t_start, self.t_end = time_window
        
        self.nt_patch = self.t_end - self.t_start

        if patch_mode == 'patch' and spatial_window is None:
            self.spatial_window = (32,)  # 默认 32 道
        self.rng = np.random.Generator(np.random.PCG64(seed)) if seed is not None else np.random.default_rng()
        
        # 训练模式：按 gather 组织
        if split in ['train', 'val']:
            self._setup_train_gathers()
        else:
            self._setup_test_gathers()
        
        # 预加载 SEGY 文件句柄（延迟打开）
        self._segy_file = None
    
    def typical_grid_step(self,arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u  # 无法估步长
        d = np.diff(u)
        d = d[d > eps]     # 去掉重复和数值噪声
        if d.size == 0:
            return None, u
        return float(np.median(d)), u

    def _compute_coord_stats(self):
        sx_all = self.index_df['sx'].values
        sy_all = self.index_df['sy'].values
        rx_all = self.index_df['gx'].values
        ry_all = self.index_df['gy'].values

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
        return stats
    
    def _normalize_coords(self, sx, sy, gx, gy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """归一化坐标到 [-1, 1]"""
        stats = self.coord_stats
        sx_n = 2 * (sx - stats['sx_min']) / (stats['sx_max'] - stats['sx_min']) - 1
        sy_n = 2 * (sy - stats['sy_min']) / (stats['sy_max'] - stats['sy_min']) - 1
        gx_n = 2 * (gx - stats['rx_min']) / (stats['rx_max'] - stats['rx_min']) - 1
        gy_n = 2 * (gy - stats['ry_min']) / (stats['ry_max'] - stats['ry_min']) - 1
        return sx_n, sy_n, gx_n, gy_n
    
    def _detect_lines_by_distance(
        self, 
        traces_df: pd.DataFrame, 
        trace_indices: List[int],
        maxRD: Optional[float] = None,
        maxSD: Optional[float] = None,
        useX: bool = True
    ) -> Dict[Any, List[int]]:
        """
        根据距离和xline变化判断测线
        
        Args:
            traces_df: 包含所有 traces 的 DataFrame
            trace_indices: trace 索引列表
            maxRD: receiver 距离阈值
            maxSD: source 距离阈值
            useX: 是否使用 xline 判断
        
        Returns:
            lines: 字典，key 是测线标识（可以是 line_id 或坐标），value 是该测线的 trace 索引列表
        """
        # 对 traces 排序（按 trace_idx 或坐标）
        sorted_df = traces_df.loc[trace_indices].sort_values(by='trace_idx')
        sorted_trace_indices = sorted_df.index.tolist()
        
        # 检查是否有 xline 字段
        has_xline = 'xline' in sorted_df.columns
        
        lines = {}  # {line_id: [trace_indices]}
        current_line_id = 0
        current_line_traces = []
        
        # 初始化上一道的坐标
        rx0 = None
        ry0 = None
        sx0 = None
        sy0 = None
        xline0 = None
        
        for trace_idx in sorted_trace_indices:
            row = sorted_df.loc[trace_idx]
            rx = row['gx']  # receiver x
            ry = row['gy']  # receiver y
            sx = row['sx']  # source x
            sy = row['sy']  # source y
            xline = row['xline'] if has_xline and useX else 0
            
            # 判断是否是新测线的开始
            is_new_line = False
            
            if rx0 is not None:  # 不是第一道
                # 计算距离
                dr = np.sqrt((rx - rx0)**2 + (ry - ry0)**2)
                ds = np.sqrt((sx - sx0)**2 + (sy - sy0)**2)
                #print(f"dr: {dr}, ds: {ds}")
                dxline = np.abs(xline - xline0) if useX and has_xline else 0
                #print(f"dxline: {dxline}")
                # 判断测线是否变化
                if (maxRD is not None and dr > maxRD) or \
                   (maxSD is not None and ds > maxSD) :
                    is_new_line = True
                    #print(f"is_new_line: {is_new_line}")
            
            if is_new_line:
                # 保存当前测线
                if len(current_line_traces) > 0:
                    lines[current_line_id] = current_line_traces
                # 开始新测线
                current_line_id += 1
                current_line_traces = [trace_idx]
            else:
                # 继续当前测线
                current_line_traces.append(trace_idx)
            
            # 更新上一道的坐标
            rx0, ry0, sx0, sy0, xline0 = rx, ry, sx, sy, xline
        
        # 保存最后一条测线
        if len(current_line_traces) > 0:
            lines[current_line_id] = current_line_traces
        
        return lines
    
    def _setup_train_gathers(self):
        """设置训练 gather 索引
        关键逻辑：按 domain 划分意味着按另一个维度组织 gather
        - domain='shot': 按 receiver_id 组织 gather（同一个 receiver 接收来自不同 shot 的数据）
        - domain='receiver': 按 shot_id 组织 gather（同一个 shot 向不同 receiver 发射）
        """
        # 划分维度
        split_id_col = 'shot_id' if self.domain == 'shot' else 'receiver_id'
        # 组织 gather 的维度（与划分维度相反）
        gather_id_col = 'receiver_id' if self.domain == 'shot' else 'shot_id'
        
        # 按 gather_id_col 组织 gather，但只包含属于 split_ids 的 traces
        gather_dict = {}
        for trace_idx in self.index_df.index:
            # 检查该 trace 是否属于当前 split
            if self.index_df.loc[trace_idx, split_id_col] in self.split_ids:
                gather_id = self.index_df.loc[trace_idx, gather_id_col]
                if gather_id not in gather_dict:
                    gather_dict[gather_id] = []
                gather_dict[gather_id].append(trace_idx)
        
        # 保存 gather 信息
        self.gather_ids = list(gather_dict.keys())
        self.gather_traces = list(gather_dict.values())
        self.gather_id_col = gather_id_col  # 保存用于后续使用
        self.split_id_col = split_id_col

        # 为每个 gather 计算可能的 patch 数量
        self.gather_info = []
        for i, (gather_id, trace_indices) in enumerate(zip(self.gather_ids, self.gather_traces)):
            n_traces = len(trace_indices)
            #print(f"n_traces: {n_traces}")
            if self.patch_mode == 'patch':
                n_patches_per_gather = max(1, (n_traces // self.spatial_window[0]) * 2)
            else:
                n_patches_per_gather = n_traces
            
            self.gather_info.append({
                'gather_id': gather_id,
                'gather_idx': i,
                'n_traces': n_traces,
                'num_patches': n_patches_per_gather
            })

        # 统计所有 gather 的 trace 数量的最小值，并调整 patch_size
        if self.patch_mode == 'patch' and len(self.gather_info) > 0 and self.spatial_window is not None:
            trace_counts = np.array([info['n_traces'] for info in self.gather_info])
            min_trace_count = int(trace_counts.min())
            current_patch_size = self.spatial_window[0]
            
            if current_patch_size > min_trace_count:
                # 需要调整 patch_size
                old_patch_size = current_patch_size
                self.spatial_window = (min_trace_count,)
                print(f"[{self.split}] Warning: patch_size ({old_patch_size}) > min_trace_count ({min_trace_count}). "
                      f"Setting patch_size to {min_trace_count}")
                
                # 重新计算每个 gather 的 patch 数量
                for info in self.gather_info:
                    n_traces = info['n_traces']
                    info['num_patches'] = max(1, (n_traces // self.spatial_window[0]) * 2)
            else:
                print(f"[{self.split}] patch_size check: patch_size ({current_patch_size}) <= min_trace_count ({min_trace_count}). OK.")

        self.patch_to_gather = []
        for info in self.gather_info:
            for patch_id in range(info['num_patches']):
                self.patch_to_gather.append((info['gather_idx'], patch_id))
    
    def _setup_test_gathers(self):
        """设置测试 gather
        关键逻辑：按 domain 划分意味着按另一个维度组织 gather
        - domain='shot': 按 receiver_id 组织 gather
        - domain='receiver': 按 shot_id 组织 gather
        """
        # 划分维度
        split_id_col = 'shot_id' if self.domain == 'shot' else 'receiver_id'
        # 组织 gather 的维度（与划分维度相反）
        gather_id_col = 'receiver_id' if self.domain == 'shot' else 'shot_id'
        
        if self.test_mode == 'line_based':
            # 按测线划分模式
            if self.n_test_lines is None:
                raise ValueError("n_test_lines must be specified when test_mode='line_based'")
            # 按 gather_id_col 组织 gather，包含所有 traces（train 和 test）
            test_gather_dict = {}
            for trace_idx in self.index_df.index:
                gather_id = self.index_df.loc[trace_idx, gather_id_col]
                # 只包含至少有一个 test trace 的 gather
                if self.index_df.loc[trace_idx, split_id_col] not in self.train_ids:
                    if gather_id not in test_gather_dict:
                        test_gather_dict[gather_id] = []
                    test_gather_dict[gather_id].append(trace_idx)
            
            # 对于每个 gather，按测线划分，每条测试测线作为一个独立的样本
            self.test_line_info = []  # 存储每条测试测线的信息
            
            for gather_id, _ in test_gather_dict.items():
                # 获取该 gather 的所有 traces（包括 train 和 test）
                all_traces_df = self.index_df[self.index_df[gather_id_col] == gather_id].copy()
                all_trace_indices = all_traces_df.index.tolist()
                
                # 根据距离和xline判断测线
                if self.maxRD is not None or self.maxSD is not None:
                    # 使用基于距离的测线判断
                    lines = self._detect_lines_by_distance(
                        all_traces_df, all_trace_indices, 
                        maxRD=self.maxRD, maxSD=self.maxSD, useX=self.useX
                    )
                else:
                    raise ValueError("maxRD and maxSD must be specified when test_mode='line_based'")

                unique_lines = list(lines.keys())
                n_lines = len(unique_lines)
                #print(n_lines)
                
                if n_lines < self.n_test_lines:
                    print(f"Warning: gather {gather_id} has only {n_lines} lines, "
                          f"but n_test_lines={self.n_test_lines}. Using all lines as test lines.")
                    test_line_keys = set(unique_lines)
                else:
                    # 随机选择 n_test_lines 条测线作为测试测线
                    test_line_indices = self.rng.choice(
                        n_lines, 
                        size=self.n_test_lines, 
                        replace=False
                    )
                    test_line_keys = set([unique_lines[i] for i in test_line_indices])
                
                # 为每条测试测线创建一个独立的条目
                for line_key in test_line_keys:
                    # 该测线的 traces
                    line_trace_indices_all = lines[line_key]
                    line_trace_indices_train = []
                    for trace_idx in line_trace_indices_all:
                        # 检查该 trace 的 split_id_col 是否在 train_ids 中
                        if self.index_df.loc[trace_idx, split_id_col] in self.train_ids:
                            line_trace_indices_train.append(trace_idx)
                    self.test_line_info.append({
                        'line_key': line_key,
                        'all_trace_indices': line_trace_indices_all,
                        'train_trace_indices': line_trace_indices_train,
                    })
            
            self.gather_id_col = gather_id_col
            self.split_id_col = split_id_col

        elif self.test_mode == 'gather_based':
            # 按 gather_id_col 组织 gather，包含所有 traces（train 和 test）
            test_gather_dict = {}
            for trace_idx in self.index_df.index:
                gather_id = self.index_df.loc[trace_idx, gather_id_col]
                # 只包含至少有一个 test trace 的 gather
                if self.index_df.loc[trace_idx, split_id_col] not in self.train_ids:
                    if gather_id not in test_gather_dict:
                        test_gather_dict[gather_id] = []
                    test_gather_dict[gather_id].append(trace_idx)
            
            self.test_gather_ids = list(test_gather_dict.keys())
            self.test_gather_traces = list(test_gather_dict.values())
            self.gather_id_col = gather_id_col
            self.split_id_col = split_id_col
        else:
            test_trace_indices = self.index_df[
                ~self.index_df[split_id_col].isin(self.train_ids)
            ].index.tolist()
            self.test_trace_indices = test_trace_indices
            self.gather_id_col = gather_id_col
            self.split_id_col = split_id_col
    
    def _get_segy_file(self):
        """延迟打开 SEGY 文件"""
        if self._segy_file is None:
            self._segy_file = segyio.open(self.segy_path, ignore_geometry=True, mode='r')
        return self._segy_file
    
    def __len__(self) -> int:
        if self.split in ['train', 'val']:
            return len(self.patch_to_gather)
        else:  # test
            if self.test_mode == 'line_based':
                return len(self.test_line_info)
            elif self.test_mode == 'gather_based':
                return len(self.test_gather_ids)
            else:
                return len(self.test_trace_indices)
    
    def __getitem__(self, idx: int):
        f = self._get_segy_file()
        
        if self.split in ['train', 'val']:
            return self._get_train_sample(f, idx)
        else:  # test
            return self._get_test_sample(f, idx)
    
    def _get_train_sample(self, f, idx: int):
        """训练样本：动态随机采样 + 应用随机掩码"""
        g_idx, _ = self.patch_to_gather[idx]
        gather_id = self.gather_ids[g_idx]
        trace_indices = self.gather_traces[g_idx]
        
        # 读取整个 gather 的数据和坐标
        gather_data = []
        gather_sx = []
        gather_sy = []
        gather_gx = []
        gather_gy = []
        
        for trace_idx in trace_indices:
            trace_data = f.trace[trace_idx][self.t_start:self.t_end].astype(np.float32)
            if np.isnan(trace_data).any() or np.isinf(trace_data).any():
                print(f"trace_idx: {trace_idx} is nan or inf")
                continue
            else:
                gather_data.append(trace_data)
                row = self.index_df.iloc[trace_idx]
                gather_sx.append(row['sx'])
                gather_sy.append(row['sy'])
                gather_gx.append(row['gx'])
                gather_gy.append(row['gy'])
        
        gather_data = np.array(gather_data)  # (n_traces, nt)
        gather_sx = np.array(gather_sx)
        gather_sy = np.array(gather_sy)
        gather_gx = np.array(gather_gx)
        gather_gy = np.array(gather_gy)
        
        n_traces, n_samples = gather_data.shape
        #print(f"n_traces: {n_traces}, n_samples: {n_samples}")
        # 动态随机采样位置
        if self.patch_mode == 'patch':
            n_traces_patch = self.spatial_window[0]
            max_trace_start = n_traces - n_traces_patch
            ts = np.random.randint(0, max(1, max_trace_start + 1))
            te = ts + n_traces_patch
            data_patch = gather_data[ts:te, :].astype(np.float32)  # (nx, nt)
            sx_patch = gather_sx[ts:te]
            sy_patch = gather_sy[ts:te]
            gx_patch = gather_gx[ts:te]
            gy_patch = gather_gy[ts:te]
        else:  # trace 模式
            trace_idx_in_gather = np.random.randint(0, n_traces)
            data_patch = gather_data[trace_idx_in_gather:trace_idx_in_gather+1, :].astype(np.float32)  # (1, nt)
            sx_patch = gather_sx[trace_idx_in_gather:trace_idx_in_gather+1]
            sy_patch = gather_sy[trace_idx_in_gather:trace_idx_in_gather+1]
            gx_patch = gather_gx[trace_idx_in_gather:trace_idx_in_gather+1]
            gy_patch = gather_gy[trace_idx_in_gather:trace_idx_in_gather+1]
        
        # 归一化坐标
        sx_n, sy_n, gx_n, gy_n = self._normalize_coords(sx_patch, sy_patch, gx_patch, gy_patch)
        
        # 坐标增强（训练时）
        if self.coord_augment:
            rx_n, ry_n, sx_n, sy_n = _augment_coords(
                gx_n, gy_n, sx_n, sy_n,
                jitter=0.05, rot_scale=True
            )
            gx_n, gy_n = rx_n, ry_n
        else:
            gx_n, gy_n = gx_n, gy_n
        
        # 生成随机掩码
        missing_ratio = sample_missing_ratio()
        masked_patch, mask_patch = apply_mixed_mask(data_patch, missing_ratio, block_prob=0.0)
        
        # 归一化：使用 masked_patch 的观测点计算 std，然后 percentile clip
        # 修复：mask_patch == 1 表示观测位置（未掩码），mask_patch == 0 表示掩码位置
        obs = masked_patch[mask_patch == 1]  # 修复：使用观测位置，不是掩码位置
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if len(obs) > 0 else np.float32(1.0)
        std_val = np.float32(max(std_val, 1e-2))
        
        # 修复：只使用观测点计算percentile，不包括掩码位置（都是0）
        if len(obs) > 0:
            thres = np.percentile(np.abs(masked_patch), 99.5)
        else:
            thres = 1e-6
        if thres == 0:
            thres = 1e-6
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_patch, -thres, thres) / thres
        
        # 时间轴
        time_idx_1d = np.arange(self.time_skip, self.nt_patch, dtype=np.int32)
        time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
        time_axis_2d = np.tile(time_axis_1d[None, :], (data_patch.shape[0], 1))
        
        # 保持 numpy array 格式，不转换为 tensor
        # 先创建完整副本，再切片，确保返回的是独立数组
        x_gt_full = np.array(data_patch, dtype=np.float32, copy=True)  # (nx, nt)
        x_obs_full = np.array(masked_patch, dtype=np.float32, copy=True)
        m_obs_full = np.array(mask_patch, dtype=np.float32, copy=True)
        
        # 坐标：提取坐标向量
        coords_2d = np.stack([gx_n, gy_n, sx_n, sy_n], axis=1)  # (nx, 4) 或 (1, 4)
        coords_2d = np.array(coords_2d, dtype=np.float32, copy=True)
        
        # 使用 np.array() 确保创建新数组，避免返回视图导致 collate 失败
        return {
            'x_gt': np.array(x_gt_full[:, self.time_skip::self.time_bins], dtype=np.float32, copy=True),
            'x_obs': np.array(x_obs_full[:, self.time_skip::self.time_bins], dtype=np.float32, copy=True),
            'm_obs': np.array(m_obs_full[:, 0], dtype=np.float32, copy=True),
            'gx': np.array(coords_2d[:, 0], dtype=np.float32, copy=True),
            'gy': np.array(coords_2d[:, 1], dtype=np.float32, copy=True),
            'sx': np.array(coords_2d[:, 2], dtype=np.float32, copy=True),
            'sy': np.array(coords_2d[:, 3], dtype=np.float32, copy=True),
            'time_axis': np.array(time_axis_2d, dtype=np.float32, copy=True),
            'std_val': np.float32(std_val)
        }
    
    def _get_test_sample(self, f, idx: int):
        """测试样本：输入来自 train geometry，目标是 test geometry"""
        if self.test_mode == 'line_based':
            # 按测线划分模式：每次返回一条测线的数据
            line_info = self.test_line_info[idx]
            all_trace_indices = line_info['all_trace_indices']  # 该 gather 的所有 traces
            train_trace_indices = line_info['train_trace_indices']  # 其他测线的 traces（作为输入）
            #test_trace_indices = line_info['test_trace_indices']  # 当前测线的 traces（作为目标）
            
        elif self.test_mode == 'gather_based':
            # 按 gather 组织
            gather_id = self.test_gather_ids[idx]
            test_trace_indices_in_gather = self.test_gather_traces[idx]
            
            # 获取该 gather 的所有 traces（包括 train 和 test）
            # gather_id_col 是组织 gather 的维度（与划分维度相反）
            all_traces_df = self.index_df[self.index_df[self.gather_id_col] == gather_id]
            all_trace_indices = all_traces_df.index.tolist()
            
            # 分离 train 和 test traces
            # split_id_col 是划分的维度
            train_trace_indices = [tid for tid in all_trace_indices 
                                 if self.index_df.loc[tid, self.split_id_col] in self.train_ids]
            test_trace_indices = [tid for tid in all_trace_indices 
                                if tid in test_trace_indices_in_gather]
        else:
            # independent_traces 模式
            raise NotImplementedError("independent_traces mode not implemented in _get_test_sample")
        
        if self.test_mode in ['gather_based', 'line_based']:
            
            # 读取数据
            all_data = []
            all_sx = []
            all_sy = []
            all_gx = []
            all_gy = []
            m_geo_list = []
            
            for trace_idx in all_trace_indices:
                trace_data = f.trace[trace_idx][self.t_start:self.t_end].astype(np.float32)
                all_data.append(trace_data)
                
                row = self.index_df.iloc[trace_idx]
                all_sx.append(row['sx'])
                all_sy.append(row['sy'])
                all_gx.append(row['gx'])
                all_gy.append(row['gy'])
                m_geo_list.append(1 if trace_idx in train_trace_indices else 0)
            
            all_data = np.array(all_data)  # (n_traces, nt)
            all_sx = np.array(all_sx)
            all_sy = np.array(all_sy)
            all_gx = np.array(all_gx)
            all_gy = np.array(all_gy)
            m_geo = np.array(m_geo_list, dtype=np.float32)  # (n_traces,)
            
            x_gt = all_data.copy()
            x_obs = all_data.copy()
            x_obs[m_geo == 1, :] = 0.0  

            # 归一化坐标
            sx_n, sy_n, gx_n, gy_n = self._normalize_coords(all_sx, all_sy, all_gx, all_gy)            
            # 时间轴
            time_idx_1d = np.arange(0, self.nt_patch, dtype=np.int32)
            time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
            time_axis_2d = np.tile(time_axis_1d[None, :], (all_data.shape[0], 1))
            x_gt_full = np.array(x_gt, dtype=np.float32, copy=True)  # (nx, nt)
            x_obs_full = np.array(x_obs, dtype=np.float32, copy=True)
            m_obs_full = np.array(m_geo[:, np.newaxis], dtype=np.float32, copy=True)  # (nx, 1)
            coords_2d = np.stack([gx_n, gy_n, sx_n, sy_n], axis=1)  # (nx, 4)
            coords_2d = np.array(coords_2d, dtype=np.float32, copy=True)
            
            # 使用 np.array() 确保创建新数组，避免返回视图导致 collate 失败
            return {
                'x_gt': np.array(x_gt_full[:, self.time_skip::self.time_bins], dtype=np.float32, copy=True),
                'x_obs': np.array(x_obs_full[:, self.time_skip::self.time_bins], dtype=np.float32, copy=True),
                'm_obs': np.array(m_obs_full[:, 0], dtype=np.float32, copy=True),
                'gx': np.array(coords_2d[:, 0], dtype=np.float32, copy=True),
                'gy': np.array(coords_2d[:, 1], dtype=np.float32, copy=True),
                'sx': np.array(coords_2d[:, 2], dtype=np.float32, copy=True),
                'sy': np.array(coords_2d[:, 3], dtype=np.float32, copy=True),
                'time_axis': np.array(time_axis_2d, dtype=np.float32, copy=True),
                'std_val': np.float32(1.0)
            }
    
    def __del__(self):
        """关闭 SEGY 文件"""
        if self._segy_file is not None:
            self._segy_file.close()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    
    # 方式1：使用默认配置创建训练集和验证集
    '''train_dataset, val_dataset = SegySSLConfig.create_C3NA_datasets(
        train_ranges=[(2401, 3600)],
        val_ranges=[(2401, 3600)],
        domain='receiver',
        spatial_window=(32,),
        missing_mode='cluster',
        missing_ratio=0.35,
    )'''
    
    train_dataset = SegySSLConfig.create_xbfy_datasets(
        split='train',
        domain='receiver',
        spatial_window=(128,),
        time_skip=14,
        time_bins=2,
    )
    val_dataset = SegySSLConfig.create_xbfy_datasets(
        split='test',
        domain='receiver',
        spatial_window=(128,),
        time_skip=10,
        time_bins=2,
    )
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 方式2：自定义数据范围
    # train_dataset, val_dataset = SegySSLConfig.create_train_val_datasets(
    #     train_ranges=[(1201, 2400), (2401, 3600)],
    #     val_ranges=[(3601, 4781)],
    #     domain='receiver',
    #     spatial_window=(68*2,),
    #     train_split='train',
    #     val_split='val'
    # )
    
    # 方式3：单个数据集
    # config = SegySSLConfig.C3NA(
    #     start=3601,
    #     end=4781,
    #     split='train',
    #     domain='receiver',
    #     spatial_window=(68*2,)
    # )
    # dataset = config.create_dataset()
    
    # 测试训练集
    sample = val_dataset[1]
    print(f"Sample keys: {sample.keys()}")
    print(f"x_gt shape: {sample['x_gt'].shape}")
    print(f"x_obs shape: {sample['x_obs'].shape}")
    print(f"m_obs shape: {sample['m_obs'].shape}")
    print(f"gx shape: {sample['gx'].shape}")
    print(f"gy shape: {sample['gy'].shape}")
    print(f"sx shape: {sample['sx'].shape}")
    print(f"sy shape: {sample['sy'].shape}")
    
    plt.imshow(sample['x_gt'].T, cmap='seismic', aspect='auto', 
               vmin=-sample['x_gt'].std(), vmax=sample['x_gt'].std())
    plt.colorbar()
    plt.savefig('./test_train.png')
    plt.close()
    plt.imshow(sample['x_obs'].T, cmap='seismic', aspect='auto', 
               vmin=-sample['x_obs'].std(), vmax=sample['x_obs'].std())
    plt.colorbar()
    plt.savefig('./test_train_mask.png')
    plt.close()
    
    # 测试 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    #
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(next(iter(train_loader))['x_obs'].shape)
    print(f"训练集 batch 数量: {len(train_loader)}")
    print(f"验证集 batch 数量: {len(val_loader)}")


    # 方式1：使用配置类的便捷方法
    '''config.plot_coverage(
        output_path='./coverage_heatmap_train.png',
        split='train',
        domain='shot'
    )
    config.plot_coverage(
        output_path='./coverage_heatmap_test.png',
        split='test',
        domain='shot'
    )'''
    
    # 方式2：使用静态方法（更灵活）
    '''SegySSLConfig.plot_coverage_heatmap(
        index_parquet='/NAS/czt/mount/Seis_DiT/segy_ssl_parquet/SEG_C3NA_ffid_2401-3600.parquet',
        split_dir='/NAS/czt/mount/Seis_DiT/segy_ssl_parquet/splits',
        split='train',
        domain='receiver',
        output_path='./coverage_heatmap_receiver_train.png',
        cmap='viridis'
    )
    SegySSLConfig.plot_coverage_heatmap(
        index_parquet='/NAS/czt/mount/Seis_DiT/segy_ssl_parquet/SEG_C3NA_ffid_2401-3600.parquet',
        split_dir='/NAS/czt/mount/Seis_DiT/segy_ssl_parquet/splits',
        split='test',
        domain='receiver',
        output_path='./coverage_heatmap_receiver_test.png',
        cmap='viridis'
    )'''
    

##C3NA 数据集路径:
#"/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_1201-2400.sgy"
#'/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_2-1200.sgy.1'
#'/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_2401-3600.sgy'
#'/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_3601-4781.sgy'
##xbfy
