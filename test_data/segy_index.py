import segyio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import trange

def build_trace_index(
    segy_path: str,
    out_parquet: str,
    field_map: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    构建 trace-level 索引，提取道头信息并保存为 parquet
    
    Args:
        segy_path: SEGY 文件路径
        out_parquet: 输出 parquet 文件路径
        field_map: 字段映射字典，用于覆盖默认字段名
    
    Returns:
        DataFrame 包含所有 trace 的索引信息
    """
    # 默认字段映射
    default_fields = {
        'shot_id': segyio.TraceField.FieldRecord,  # fldr/ffid
        'trace_in_file': segyio.TraceField.TraceNumber,
        'source_x': segyio.TraceField.SourceX,
        'source_y': segyio.TraceField.SourceY,
        'group_x': segyio.TraceField.GroupX,
        'group_y': segyio.TraceField.GroupY,
        'trace_sample_interval': segyio.TraceField.TRACE_SAMPLE_INTERVAL,
        'samples': segyio.TraceField.TRACE_SAMPLE_COUNT,
        'xline': segyio.TraceField.TRACE_SEQUENCE_LINE,
    }
    
    if field_map:
        default_fields.update(field_map)
    
    records = []
    
    with segyio.open(segy_path, ignore_geometry=True, mode='r') as f:
        trace_count = min(f.tracecount, 1000000)
        ns = len(f.trace[0]) if trace_count > 0 else 0
        
        # 读取第一条 trace 获取采样间隔
        first_header = f.header[0]
        dt_us = first_header.get(default_fields['trace_sample_interval'], 0)
        dt_s = dt_us / 1e6 if dt_us > 0 else None
        
        for trace_idx in trange(trace_count):
            header = f.header[trace_idx]
            
            # 提取坐标
            sx_raw = header.get(default_fields['source_x'], 0)
            sy_raw = header.get(default_fields['source_y'], 0)
            gx_raw = header.get(default_fields['group_x'], 0)
            gy_raw = header.get(default_fields['group_y'], 0)
            xline_raw = header.get(default_fields['xline'], 0)
            
            sx = float(sx_raw)
            sy = float(sy_raw)
            gx = float(gx_raw)
            gy = float(gy_raw)
            xline = float(xline_raw)

            # 提取 shot_id 和 receiver_id
            shot_id = header.get(default_fields['shot_id'], trace_idx)
            receiver_group = -1
            #trace_in_file = header.get(default_fields['trace_in_file'], trace_idx)
            if receiver_group > 0:
                receiver_id = int(receiver_group)
            else:
                # 使用 (gx, gy) 的 hash 作为 receiver_id
                receiver_id = hash((int(gx), int(gy))) % (2**31)
            
            # 计算 offset 和 azimuth
            dx = gx - sx
            dy = gy - sy
            offset = np.sqrt(dx**2 + dy**2)
            azimuth = np.arctan2(dy, dx)  # 弧度
            
            # 提取样本信息
            trace_ns = header.get(default_fields['samples'], ns)
            if trace_ns <= 0:
                trace_ns = ns
            
            record = {
                'trace_idx': trace_idx,
                'shot_id': int(shot_id),
                'receiver_id': int(receiver_id),
                'sx': float(sx),
                'sy': float(sy),
                'gx': float(gx),
                'gy': float(gy),
                'offset': float(offset),
                'azimuth': float(azimuth),
                'ns': int(trace_ns),
                'dt_us': int(dt_us),
                'dt_s': float(dt_s) if dt_s else None,
                'xline': float(xline),
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet(out_parquet, index=False)
    
    return df


def test_build_trace_index(segy_path, out_parquet):
    df = build_trace_index(segy_path, out_parquet)
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.isnull().sum().sum())
    print(df.isnull().sum().sum().sum())


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        build_trace_index(sys.argv[1])
    else:
        print("Usage: python segy_index.py <segy_path> <out_parquet>")
        print("Example: python segy_index.py /home/jiangyr/data/data_12000_18_5d.sgy /home/jiangyr/data/data_12000_18_5d.parquet")
