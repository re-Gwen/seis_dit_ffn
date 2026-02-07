import numpy as np
from typing import Tuple, Dict, Optional


def sample_train_mask(
    shape: Tuple[int, ...],
    r_min: float,
    r_max: float,
    mask_type_probs: Dict[str, float],
    seed: Optional[int] = None,
    trace_dim: int = 0
) -> Tuple[np.ndarray, float]:
    """
    生成训练时的随机掩码（按道掩码）
    
    Args:
        shape: 掩码形状，第0维是道（trace），例如 (n_traces, n_time) 或 (n_traces, n_time, n_x)
        r_min: 最小缺失率
        r_max: 最大缺失率
        mask_type_probs: 掩码类型概率，例如 {"random": 0.5, "continuous": 0.5} 或混合模式
        seed: 随机种子
        trace_dim: 道所在的维度，默认为0
    
    Returns:
        M_rand: 掩码数组 (uint8, 1 表示缺失/掩码)，与 shape 相同
        r: 实际缺失率
    """
    if seed is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        rng = np.random.default_rng()
    
    # 采样缺失率
    r = rng.uniform(r_min, r_max)
    
    # 获取道的数量（第0维）
    n_traces = shape[trace_dim]
    n_masked_traces = int(n_traces * r)
    
    if n_masked_traces == 0:
        mask = np.zeros(shape, dtype=np.uint8)
        return mask, 0.0
    
    # 采样掩码类型
    mask_types = list(mask_type_probs.keys())
    probs = list(mask_type_probs.values())
    probs = np.array(probs) / np.sum(probs)  # 归一化
    
    # 生成道级别的掩码（1D，长度为 n_traces）
    trace_mask = np.zeros(n_traces, dtype=bool)
    
    # 如果只有一种类型，直接使用
    if len(mask_types) == 1:
        mask_type = mask_types[0]
        if mask_type == 'random':
            trace_mask = _sample_trace_random_mask(n_traces, n_masked_traces, rng)
        elif mask_type == 'continuous':
            trace_mask = _sample_trace_continuous_mask(n_traces, n_masked_traces, rng)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}, must be 'random' or 'continuous'")
    else:
        # 混合模式：按比例分配掩码的道数
        remaining_traces = n_masked_traces
        remaining_mask = np.zeros(n_traces, dtype=bool)
        
        for i, mask_type in enumerate(mask_types):
            if remaining_traces <= 0:
                break
            
            # 计算这种类型应该掩码的道数
            if i == len(mask_types) - 1:
                # 最后一种类型，使用剩余的所有
                n_this_type = remaining_traces
            else:
                n_this_type = int(n_masked_traces * probs[i])
                n_this_type = min(n_this_type, remaining_traces)
            
            if n_this_type > 0:
                # 从未掩码的道中选择
                available_traces = np.where(~remaining_mask)[0]
                if len(available_traces) >= n_this_type:
                    if mask_type == 'random':
                        selected = rng.choice(available_traces, size=n_this_type, replace=False)
                        remaining_mask[selected] = True
                    elif mask_type == 'continuous':
                        # 连续掩码：选择一个起始位置
                        start_idx = rng.integers(0, max(1, len(available_traces) - n_this_type + 1))
                        selected = available_traces[start_idx:start_idx + n_this_type]
                        remaining_mask[selected] = True
                    else:
                        raise ValueError(f"Unknown mask_type: {mask_type}")
                    remaining_traces -= n_this_type
        
        trace_mask = remaining_mask
    
    # 将道级别掩码广播到完整形状
    # trace_mask 是 (n_traces,) 形状，需要广播到 shape
    trace_mask_uint8 = trace_mask.astype(np.uint8)
    
    # 构建广播形状：在 trace_dim 维度保持原样，其他维度为 1
    broadcast_shape = [1] * len(shape)
    broadcast_shape[trace_dim] = shape[trace_dim]
    trace_mask_broadcast = trace_mask_uint8.reshape(broadcast_shape)
    
    # 广播到完整形状
    mask = np.broadcast_to(trace_mask_broadcast, shape).copy()
    
    # 计算实际缺失率
    actual_r = np.mean(mask)
    
    return mask, float(actual_r)


def _sample_trace_random_mask(n_traces: int, n_masked_traces: int, rng: np.random.Generator) -> np.ndarray:
    """随机道掩码：随机选择若干道"""
    mask = np.zeros(n_traces, dtype=bool)
    if n_masked_traces > 0:
        indices = rng.choice(n_traces, size=n_masked_traces, replace=False)
        mask[indices] = True
    return mask


def _sample_trace_continuous_mask(n_traces: int, n_masked_traces: int, rng: np.random.Generator) -> np.ndarray:
    """连续道掩码：选择连续的若干道"""
    mask = np.zeros(n_traces, dtype=bool)
    if n_masked_traces > 0:
        # 选择起始位置
        start_idx = rng.integers(0, n_traces - n_masked_traces + 1)
        mask[start_idx:start_idx + n_masked_traces] = True
    return mask
