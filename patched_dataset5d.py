import os
from pathlib import Path
import re
import sys
import csv
from typing import List, Optional, Tuple
import segyio
import datetime
from torch.utils.data import Dataset
import torch
import numpy as np
from progressbar import *
import matplotlib.pyplot as plt
import random
from scipy import signal
from collections import defaultdict
from tqdm import tqdm, trange
from scipy.stats import f, mode

def normalize(data):
    for k in range(data.shape[0]):
        aux =  data[k,:]
        std = data[k,:].std()
        if std<1e-7:
            std = 1e-7
        data[k,:] = (data[k,:]-data[k,:].mean())/std
        amax = max(abs(aux))
        if amax<1e-7:
            amax=1e-7
        data[k,:] = data[k,:]/amax
    return data 

def normalize_clip(data): 
    threshold = np.percentile(np.abs(data), 99.5) 
    if threshold == 0:
        threshold = 1e-6
    data =np.clip(data, -threshold, threshold) 
    data = data / threshold 
    return data

def normalize_sgn(x):
    return np.tanh(np.sign(x)*np.log1p(1+np.abs(x)))

def normalize_coords(arr, amin=None, amax=None):
    # linear map to [-1,1]; if amin/amax None, use arr min/max
    if amin is None: amin = torch.min(arr)
    if amax is None: amax = torch.max(arr)
    d = amax - amin if (amax - amin) != 0 else 1.0
    arr=(arr - amin) / d 
    return arr ,amin, amax

def robust_scale_pair(x_full, x_masked, mask, q=0.995, eps=1e-6):
    obs = x_masked[mask > 0.5];  obs = obs if obs.size else x_full
    tau = np.quantile(np.abs(obs), q) + eps
    x_full_n = np.clip(x_full,  -tau, tau) / tau
    x_mask_n = np.clip(x_masked,-tau, tau) / tau
    return x_full_n.astype(np.float32), x_mask_n.astype(np.float32), np.float32(np.log(tau))

EPS_AMP = 1e-3
def keep_noblank_patches(patches, coords_list, threshold=1e-2):
    """
    patches: (N_patches, H, W)
    coords_list: 坐标列表，每个 (N_patches, H)，例如 [rx_p, ry_p, sx_p, sy_p] 或 [cmp_x_p, cmp_y_p, offset_p, azimuth_p]
    返回:
        patches_f, coords_f, kept_idx
    """
    keep = []

    for k in range(patches.shape[0]):
        patch = patches[k]
        # 判定是否"非空 patch"：使用标准差判断
        if np.std(patch) > threshold:
            keep.append(k)

    if len(keep) == 0:
        return None, None, None

    keep = np.array(keep, dtype=np.int32)

    patches_f = patches[keep]
    coords_f = [coord_p[keep] for coord_p in coords_list]
    return patches_f, coords_f, keep

class DownsamplingDataset(Dataset):
    """Dataset wrapping tensors. 
    Arguments:
        xs (Tensor): clean data patches
        rate: data sampling rate when regular=False, e.g. 0.3
              data sampling interval when regular=True
    """

    def __init__(self, xs, rate, regular=False):
        super(DownsamplingDataset, self).__init__()
        self.xs = xs
        self.rate = rate
        self.regular = regular

    def __getitem__(self, index):
        batch_x = self.xs[index]
        # the type of the data must be tensor
        if self.regular:
            mask = regular_mask(batch_x, self.rate)
        else:
            mask = irregular_mask(batch_x, self.rate)
        batch_y = mask.mul(batch_x)

        return batch_y, batch_x, mask

    def __len__(self):
        return self.xs.size(0)


def irregular_mask(data, rate):
    """the mask matrix of random sampling
    Args:
        data: original data patches
        rate: sampling rate,range(0,1)
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(), dtype=torch.float64)  # tensor
    v = round(n * rate)
    TM = random.sample(range(n), v)
    mask[:, :, TM] = 1  # missing by column
    return mask


def regular_mask(data, a):
    """the mask matrix of regular sampling
    Args:
        data: original data patches
        a(int): sampling interval, e.g: a = 5, sampling like : 100001000010000
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(), dtype=torch.float64)
    for i in range(n):
        if (i + 1) % a == 1:
            mask[:, :, i] = 1
        else:
            mask[:, :, i] = 0
    return mask

def augment_coordinates(
        rx, ry, sx, sy,
        rotation=True,           # 是否旋转
        isotropic_scale=False,    # 是否等比例缩放
        anisotropic_scale=False,  # 是否各向异性缩放
        translate=True,          # 是否随机平移
        local_jitter=True,       # 是否局部扰动
        shear=False,             # 是否剪切
        flip=False,               # 是否镜像
        min_dist=None,           # 炮检最小距离约束（None 表示不检查）
        scale_range=(0.8, 1.2),  # 缩放范围
        jitter_std=0.05,         # 局部扰动强度（比例）
        shear_range=(-0.1, 0.1)  # 剪切范围
   ):
    rx, ry, sx, sy = rx.copy(), ry.copy(), sx.copy(), sy.copy()

    # 1. 随机旋转
    if rotation:
        theta = np.random.rand() * np.pi * 2
        #rx, ry = rx * np.cos(theta) - ry * np.sin(theta), rx * np.sin(theta) + ry * np.cos(theta)
        sx, sy = sx * np.cos(theta) - sy * np.sin(theta), sx * np.sin(theta) + sy * np.cos(theta)

    # 2. 等比例缩放
    if isotropic_scale:
        scale = np.random.uniform(*scale_range)
        rx *= scale; ry *= scale
        sx *= scale; sy *= scale

    # 3. 各向异性缩放
    if anisotropic_scale:
        scale_x = np.random.uniform(*scale_range)
        scale_y = np.random.uniform(*scale_range)
        rx *= scale_x; sx *= scale_x
        ry *= scale_y; sy *= scale_y

    # 4. 随机平移（中心化）
    if translate and np.random.rand() > 0.5:
        dx = np.random.choice(sx); dy = np.random.choice(sy)
        #rx -= dx; ry -= dy
        sx -= dx; sy -= dy

    # 5. 局部扰动
    if local_jitter:
        #rx += np.random.randn(*rx.shape) * jitter_std
        #ry += np.random.randn(*ry.shape) * jitter_std
        sx += np.random.randn(*sx.shape) * jitter_std
        sy += np.random.randn(*sy.shape) * jitter_std

    # 6. 剪切变换
    if shear:
        shear_x = np.random.uniform(*shear_range)
        shear_y = np.random.uniform(*shear_range)
        rx, ry = rx + shear_x * ry, ry + shear_y * rx
        sx, sy = sx + shear_x * sy, sy + shear_y * sx

    # 7. 随机镜像
    if flip:
        if np.random.rand() > 0.5:
            rx, sx = -rx, -sx
        if np.random.rand() > 0.5:
            ry, sy = -ry, -sy

    # 8. 最小炮检距约束
    if min_dist is not None:
        dist = np.sqrt((rx - sx)**2 + (ry - sy)**2)
        mask = dist < min_dist
        if np.any(mask):
            correction = min_dist - dist[mask]
            rx[mask] += correction / 2
            sx[mask] -= correction / 2

    return rx, ry, sx, sy

def patch_show(train_data, save=False, root=""):
    """
    show some sampels of train data
    save: save or not save the showed sample
    root(path)if save=True, the data will be saved to this path(as a .png picture)
    """
    samples = 4
    idxs = np.random.choice(len(train_data), samples, replace=True)
    print(idxs)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, idx in enumerate(idxs):
        plt_idx = i + 1
        data = train_data[idx]
        y, x = np.reshape(data[0], (data[0].shape[1], data[0].shape[2])), np.reshape(
            data[1], (data[1].shape[1], data[1].shape[2])
        )
        plt.subplot(2, samples, plt_idx)
        plt.imshow(x)
        plt.axis("off")
        plt.subplot(2, samples, plt_idx + samples)
        plt.imshow(y)
        plt.axis("off")
    plt.show()

    if save:
        path = os.path.join(root, "samples.png")
        plt.savefig(path)



def progress_bar(temp_size, total_size, patch_num, file, file_list):
    done = int(50 * temp_size / total_size)
    #    sys.stdout.write("\r[%s%s][%s%s] %d%% %s" % (i+1,len(file_list),'#' * done, ' ' * (50 - done), 100 * temp_size / total_size,patch_num))
    sys.stdout.write(
        "\r[%s/%s][%s%s] %d%% %s"
        % (
            file + 1,
            file_list,
            "#" * done,
            " " * (50 - done),
            100 * temp_size / total_size,
            patch_num,
        )
    )
    sys.stdout.flush()
    
def gen_patches(traces, coords_list, patch_size, stride):
    """
    traces:      (N_traces, N_samples)
    coords_list: 坐标列表，每个都是 (N_traces,)，例如 [rx, ry, sx, sy] 或 [cmp_x, cmp_y, offset, azimuth]
    patch_size:  (trace_ps, time_ps)
    stride:      (trace_sd, time_sd)
    """
    trace_ps, time_ps = patch_size
    trace_sd, time_sd = stride

    n_traces, n_samples = traces.shape

    patches = []
    coords_p_lists = [[] for _ in range(len(coords_list))]
    t_idx_list = []   
    for i in range(0, n_traces - trace_ps + 1, trace_sd):
        for j in range(0, n_samples - time_ps + 1, time_sd):
            patch = traces[i:i + trace_ps, j:j + time_ps]          # (trace_ps, time_ps)
            patches.append(patch)
            for coord_idx, coord in enumerate(coords_list):
                coords_p_lists[coord_idx].append(coord[i:i + trace_ps])
            t_idx_list.append(j)

    patches = np.stack(patches, axis=0)                 # (N_patches, trace_ps, time_ps)
    coords_p = [np.stack(coord_p_list, axis=0) for coord_p_list in coords_p_lists]  # 每个都是 (N_patches, trace_ps)
    t_idx = np.array(t_idx_list, dtype=np.int32)        # (N_patches,)

    return patches, *coords_p, t_idx

def gen_patches_random(traces, coords_list, patch_size, num_patches=None, seed=None):
    """
    随机采样 patches（用于训练）
    
    Args:
        traces: (N_traces, N_samples)
        coords_list: 坐标列表 [rx, ry, sx, sy]
        patch_size: (trace_ps, time_ps)
        num_patches: 要采样的 patch 数量。如果为 None，自动计算合理数量
        seed: 随机种子（可选）
    
    Returns:
        patches: (num_patches, trace_ps, time_ps)
        coords_p: list of (num_patches, trace_ps)
        t_idx: (num_patches,) - 时间起始索引
        trace_idx: (num_patches,) - 道起始索引
    """
    if seed is not None:
        np.random.seed(seed)
    
    trace_ps, time_ps = patch_size
    n_traces, n_samples = traces.shape
    
    # 确保有足够空间采样
    max_trace_start = n_traces - trace_ps
    max_time_start = n_samples - time_ps
    
    if max_trace_start < 0 or max_time_start < 0:
        raise ValueError(f"数据太小，无法切出 {patch_size} 的 patch")
    
    # 如果未指定 num_patches，计算合理数量
    # 目标：覆盖约 80% 的数据区域（有重叠）
    if num_patches is None:
        # 估计不重叠情况下的 patch 数
        non_overlap_patches = ((n_traces // trace_ps) * (n_samples // time_ps))
        # 增加 2-3 倍以获得更多样性和重叠
        num_patches = max(non_overlap_patches * 2, 10)
    
    patches = []
    coords_p_lists = [[] for _ in range(len(coords_list))]
    t_idx_list = []
    trace_idx_list = []
    
    for _ in range(num_patches):
        # 随机采样起始位置
        i = np.random.randint(0, max_trace_start + 1)
        j = np.random.randint(0, max_time_start + 1)
        
        patch = traces[i:i + trace_ps, j:j + time_ps]
        patches.append(patch)
        
        for coord_idx, coord in enumerate(coords_list):
            coords_p_lists[coord_idx].append(coord[i:i + trace_ps])
        
        t_idx_list.append(j)
        trace_idx_list.append(i)
    
    patches = np.stack(patches, axis=0)
    coords_p = [np.stack(coord_p_list, axis=0) for coord_p_list in coords_p_lists]
    t_idx = np.array(t_idx_list, dtype=np.int32)
    trace_idx = np.array(trace_idx_list, dtype=np.int32)
    
    return patches, *coords_p, t_idx, trace_idx

def patch_show(train_data, save=False, root=""):
    """
    show some sampels of train data
    save: save or not save the showed sample
    root(path)if save=True, the data will be saved to this path(as a .png picture)
    """
    samples = 4
    idxs = np.random.choice(len(train_data), samples, replace=True)
    print(idxs)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, idx in enumerate(idxs):
        plt_idx = i + 1
        data = train_data[idx]
        plt.imshow(data.T,cmap='seismic',vmin=-1,vmax=1)
        plt.colorbar()
        plt.show()
        #plt.axis("off")
        if save:
            path = os.path.join(root, "sample_{}.png".format(plt_idx))
            plt.savefig(path)
        break


def reconstruct_from_patches(
    patches,
    data_shape,
    patch_size=(128,128),
    stride=(64, 64),
):
    """ 
    Args:
        patches (list of ndarray)
        data_shape (tuple): (h, w)
    Returns:
        data_reconstructed (ndarray)
    """
    h, w = data_shape
    p_h, p_w = patch_size
    s_h, s_w = stride

    data_reconstructed = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    # 计算起始位置，保证与 gen_patches 一致
    h_starts = list(range(0, h - p_h + 1, s_h))
    if h_starts[-1] != h - p_h:
        h_starts.append(h - p_h)
    w_starts = list(range(0, w - p_w + 1, s_w))
    if w_starts[-1] != w - p_w:
        w_starts.append(w - p_w)

    patch_idx = 0
    for i in h_starts:
        for j in w_starts:
            data_reconstructed[i:i+p_h, j:j+p_w] += patches[patch_idx]
            weight[i:i+p_h, j:j+p_w] += 1.0
            patch_idx += 1
    weight[weight == 0] = 1.0
    data_reconstructed /= weight
    return data_reconstructed

def gen_patches_torch(
    data: torch.Tensor,
    pos: list,
    patch_size=(64, 64),
    stride=(64, 64),
    return_t:bool=False,
    verbose=None,
):
    """
    将 batch 数据和位置信息一起划分为 patch (支持 GPU Tensor)

    Args:
        data (torch.Tensor): [B, 1, H, W]
        pos (list of torch.Tensor): [rx, ry, sx, sy]，每个 shape=[W,] 或 [H,]
        patch_size (tuple): (p_h, p_w)
        stride (tuple): (s_h, s_w)

    Returns:
        data_patches (list of torch.Tensor): [N_patches, 1, p_h, p_w]
        rxs, rys, sxs, sys (list of torch.Tensor)
    """
    B, C, H, W = data.shape
    assert C == 1, "只支持单通道数据"
    assert len(pos) == 4, "pos should be [rx, ry, sx, sy]"

    rx, ry, sx, sy = [torch.as_tensor(x, device=data.device) for x in pos]
    data_patches = []
    rxs_list, rys_list, sxs_list, sys_list = [], [], [], []
    t_idx_list = []

    h_starts = list(range(0, H - patch_size[0] + 1, stride[0]))
    if h_starts[-1] != H - patch_size[0]:
        h_starts.append(H - patch_size[0])

    w_starts = list(range(0, W - patch_size[1] + 1, stride[1]))
    if w_starts[-1] != W - patch_size[1]:
        w_starts.append(W - patch_size[1])

    patch_num = 0
    for b in range(B):
        for i in h_starts:
            for j in w_starts:
                patch = data[b, :, i:i+patch_size[0], j:j+patch_size[1]]
                data_patches.append(patch)
                rxs_list.append(rx[:,i:i+patch_size[0]])
                rys_list.append(ry[:,i:i+patch_size[0]])
                sxs_list.append(sx[:,i:i+patch_size[0]])
                sys_list.append(sy[:,i:i+patch_size[0]])
                t_idx_list.append(j)
                patch_num += 1
                if verbose:
                    print(f"Batch {b} Patch {patch_num}: {patch.shape}, rx_patch {rxs_list[-1].shape}")
    if return_t:
        return data_patches, rxs_list, rys_list, sxs_list, sys_list, torch.from_numpy(np.array(t_idx_list, dtype=np.int32))
    else:
        return data_patches, rxs_list, rys_list, sxs_list, sys_list,None

'''def gen_patches_torch(
    data: torch.Tensor,
    pos: list,
    patch_size=(64, 64),
    stride=(64, 64),
    verbose: bool = False,
):
    """
    将 batch 数据和位置信息一起划分为 patch (支持 GPU Tensor)

    Args:
        data (torch.Tensor): [B, 1, H, W]
        pos (list of torch.Tensor/ndarray): [rx, ry, sx, sy]
            - 每个元素形状可以是:
                [B, H]  或  [H]  (会自动 broadcast 到 B)
        patch_size (tuple): (p_h, p_w)  => (trace_dim, time_dim)
        stride (tuple): (s_h, s_w)
        verbose (bool): 是否打印 patch 信息

    Returns:
        data_patches: torch.Tensor [N_patches, 1, p_h, p_w]
        rxs:          torch.Tensor [N_patches, p_h]
        rys:          torch.Tensor [N_patches, p_h]
        sxs:          torch.Tensor [N_patches, p_h]
        sys:          torch.Tensor [N_patches, p_h]
        t_idx:        torch.Tensor [N_patches]  # 每个 patch 在时间维上的起始 index（W 方向）
    """
    B, C, H, W = data.shape
    assert C == 1, "只支持单通道数据"
    assert len(pos) == 4, "pos should be [rx, ry, sx, sy]"

    # ---- 处理坐标形状 ----
    rx, ry, sx, sy = pos  # 可能是 numpy，也可能是 tensor
    rx = torch.as_tensor(rx, device=data.device)
    ry = torch.as_tensor(ry, device=data.device)
    sx = torch.as_tensor(sx, device=data.device)
    sy = torch.as_tensor(sy, device=data.device)

    # 允许传入 [H]，自动 broadcast 到 [B, H]
    if rx.ndim == 1:
        rx = rx.unsqueeze(0).expand(B, -1)  # [B, H]
        ry = ry.unsqueeze(0).expand(B, -1)
        sx = sx.unsqueeze(0).expand(B, -1)
        sy = sy.unsqueeze(0).expand(B, -1)
    else:
        # 这里假定是 [B, H]，如果不是可以再加断言
        assert rx.shape == (B, H), f"rx shape 应为 [B,H]，当前 {rx.shape}"

    p_h, p_w = patch_size
    s_h, s_w = stride

    data_patches = []
    rxs_list, rys_list, sxs_list, sys_list = [], [], [], []
    t_idx_list = []   # ★ 记录每个 patch 在时间维上的起始索引 j

    # ---- 起始点列表（trace 方向 H，time 方向 W）----
    h_starts = list(range(0, H - p_h + 1, s_h))
    if h_starts[-1] != H - p_h:
        h_starts.append(H - p_h)

    w_starts = list(range(0, W - p_w + 1, s_w))
    if w_starts[-1] != W - p_w:
        w_starts.append(W - p_w)

    patch_num = 0
    for b in range(B):
        for i in h_starts:
            for j in w_starts:
                # 振幅 patch: (1, p_h, p_w)
                patch = data[b, :, i:i+p_h, j:j+p_w]      # [1, p_h, p_w]
                data_patches.append(patch)

                # 坐标 patch 只沿 trace 维度切片: (p_h,)
                rxs_list.append(rx[b, i:i+p_h])           # [p_h]
                rys_list.append(ry[b, i:i+p_h])
                sxs_list.append(sx[b, i:i+p_h])
                sys_list.append(sy[b, i:i+p_h])

                # 时间起始 index（W 方向）
                t_idx_list.append(j)

                patch_num += 1
                if verbose:
                    print(
                        f"Batch {b} Patch {patch_num}: {patch.shape}, "
                        f"rx_patch {rxs_list[-1].shape}, t_start={j}"
                    )

    # ---- 堆叠成 Tensor ----
    if len(data_patches) == 0:
        # 返回空 tensor，占位
        return (
            torch.empty(0, 1, p_h, p_w, device=data.device),
            torch.empty(0, p_h, device=data.device),
            torch.empty(0, p_h, device=data.device),
            torch.empty(0, p_h, device=data.device),
            torch.empty(0, p_h, device=data.device),
            torch.empty(0, dtype=torch.long, device=data.device),
        )

    data_patches = torch.stack(data_patches, dim=0)   # [N, 1, p_h, p_w]
    rxs = torch.stack(rxs_list, dim=0)                # [N, p_h]
    rys = torch.stack(rys_list, dim=0)                # [N, p_h]
    sxs = torch.stack(sxs_list, dim=0)                # [N, p_h]
    sys = torch.stack(sys_list, dim=0)                # [N, p_h]
    t_idx = torch.tensor(t_idx_list, device=data.device, dtype=torch.long)  # [N]

    return data_patches, rxs, rys, sxs, sys, t_idx'''

'''def gen_patches_torch(
    data: torch.Tensor,
    pos: list,
    patch_size=(64, 64),
    stride=(64, 64),
    verbose=None,
):
    """
    将 batch 数据和位置信息一起划分为 patch (支持 GPU Tensor)

    Args:
        data (torch.Tensor): [B, 1, H, W]
        pos (list of torch.Tensor): [rx, ry, sx, sy]，每个 shape=[W,] 或 [H,]
        patch_size (tuple): (p_h, p_w)
        stride (tuple): (s_h, s_w)

    Returns:
        data_patches (list of torch.Tensor): [N_patches, 1, p_h, p_w]
        rxs, rys, sxs, sys (list of torch.Tensor)
    """
    B, C, H, W = data.shape
    assert C == 1, "只支持单通道数据"
    assert len(pos) == 4, "pos should be [rx, ry, sx, sy]"

    rx, ry, sx, sy = [torch.as_tensor(x, device=data.device) for x in pos]
    #print(rx.shape, ry.shape, sx.shape, sy.shape)
    data_patches = []
    rxs_list, rys_list, sxs_list, sys_list = [], [], [], []

    # 起始点
    h_starts = list(range(0, H - patch_size[0] + 1, stride[0]))
    if h_starts[-1] != H - patch_size[0]:
        h_starts.append(H - patch_size[0])

    w_starts = list(range(0, W - patch_size[1] + 1, stride[1]))
    if w_starts[-1] != W - patch_size[1]:
        w_starts.append(W - patch_size[1])

    patch_num = 0
    for b in range(B):
        for i in h_starts:
            for j in w_starts:
                patch = data[b, :, i:i+patch_size[0], j:j+patch_size[1]]
                data_patches.append(patch)
                rxs_list.append(rx[:,i:i+patch_size[0]])
                rys_list.append(ry[:,i:i+patch_size[0]])
                sxs_list.append(sx[:,i:i+patch_size[0]])
                sys_list.append(sy[:,i:i+patch_size[0]])

                patch_num += 1
                if verbose:
                    print(f"Batch {b} Patch {patch_num}: {patch.shape}, rx_patch {rxs_list[-1].shape}")

    return data_patches, rxs_list, rys_list, sxs_list, sys_list'''


def reconstruct_from_patches_torch(
    patches: list,
    batch_size: int,
    data_shape,
    patch_size=(64,64),
    stride=(64,64),
):
    """
    将 batch patch 列表重建为完整数据

    Args:
        patches (list of torch.Tensor): [1, p_h, p_w] 单通道 patch
        batch_size (int): batch 数量
        data_shape (tuple): (H, W)
        patch_size (tuple)
        stride (tuple)

    Returns:
        data_reconstructed (torch.Tensor): [B, 1, H, W]
    """
    B = batch_size
    H, W = data_shape
    p_h, p_w = patch_size
    s_h, s_w = stride

    device = patches[0].device
    dtype = patches[0].dtype

    data_reconstructed = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    weight = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    # 起始点
    h_starts = list(range(0, H - p_h + 1, s_h))
    if h_starts[-1] != H - p_h:
        h_starts.append(H - p_h)

    w_starts = list(range(0, W - p_w + 1, s_w))
    if w_starts[-1] != W - p_w:
        w_starts.append(W - p_w)

    patch_idx = 0
    for b in range(B):
        for i in h_starts:
            for j in w_starts:
                data_reconstructed[b, :, i:i+p_h, j:j+p_w] += patches[patch_idx][0,:,:]
                weight[b, :, i:i+p_h, j:j+p_w] += 1.0
                patch_idx += 1

    data_reconstructed /= weight
    return data_reconstructed


def shot_generator_stream(filename, i0=0, i1=999999999999, isData=True):
    """
    流式按炮集分组，不需要预读所有数据
    """
    with segyio.open(filename, ignore_geometry=True, mode='r') as segyfile:
        current_shot_key = None
        dataL, rxL, ryL, sxL, syL, iL = [], [], [], [], [], []
        delta = None

        for trace_index in range(i0, min(segyfile.tracecount, i1)):
            header = segyfile.header[trace_index]

            # 创建炮集标识符（使用SourceX, SourceY组合）
            shot_x = header[segyio.TraceField.SourceX]
            shot_y = header[segyio.TraceField.SourceY]
            shot_key = (shot_x, shot_y)

            # 如果是新的炮集，yield上一个炮集
            if current_shot_key is not None and shot_key != current_shot_key:
                if len(dataL) > 0:
                    yield np.array(dataL), np.array(rxL), np.array(ryL), \
                          np.array(sxL), np.array(syL), np.array(iL), delta
                dataL, rxL, ryL, sxL, syL, iL = [], [], [], [], [], []

            # 添加当前trace数据
            if isData:
                trace = segyfile.trace[trace_index]
                dataL.append(trace.data[:])
            else:
                dataL.append(0)

            rxL.append(header[segyio.TraceField.GroupX])
            ryL.append(header[segyio.TraceField.GroupY])
            sxL.append(shot_x)
            syL.append(shot_y)
            iL.append(trace_index)
            delta = header[segyio.TraceField.TRACE_SAMPLE_INTERVAL]/1e6

            current_shot_key = shot_key


        # yield最后一个炮集
        if len(dataL) > 0:
            yield np.array(dataL), np.array(rxL), np.array(ryL), \
                  np.array(sxL), np.array(syL), np.array(iL), delta
## SEG C3NA
class SEG_C3NA_patched(Dataset):
    def __init__(
        self,
        max_traces: int,
        root_List: List[str],
        time_step: int,
        missing_rate_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        add_missing: bool = True,
        trace_num: int = 544,
        ps: int = 128,
        stride: int = 64,
        contin_missing: bool = False,
    ) -> None:
        super().__init__()
        self.max_traces = max_traces
        self.missing_rate_list = missing_rate_list  # 缺失率列表
        self.add_missing = add_missing
        self.dataL = []
        self.mask_dataL = []
        self.rxL = []
        self.ryL = []
        self.sxL = []
        self.syL = []
        self.maskL = []

        # 统计用
        total_count = 0
        skipped_nan = 0
        skipped_std0 = 0
        skipped_shape = 0
        kept_count = 0

        for root in root_List:
            generator = shot_generator_stream(root, isData=True, i1=max_traces)
            for data, rx, ry, sx, sy, i, delta in generator:
                total_count += 1
                # augment coordinates
                rx, ry, sx, sy = augment_coordinates(rx, ry, sx, sy)
                if not np.isfinite(data).all():
                    print(1)
                    skipped_nan += 1
                    continue
                std = data.std()
                if std == 0 or np.isnan(std):
                    skipped_std0 += 1
                    continue

                resampled_data = data/std
                #resampled_data = normalize(data)
                resampled_data = resampled_data[:,:time_step]

                if resampled_data.shape[0] != trace_num:
                    skipped_shape += 1
                    continue

                if not np.isfinite(resampled_data).all():
                    skipped_nan += 1
                    continue
                ##get masked
                if self.add_missing:
                    selected_missing_rate = np.random.choice(self.missing_rate_list)
                    data_masked, mask = self.add_trace_missing(
                        resampled_data, selected_missing_rate,continuous=contin_missing
                    )
                else:
                    data_masked = data
                ##get patches
                mask_patches, rxs, rys, sxs, sys,_ = gen_patches(
                    data_masked,
                    [rx, ry, sx, sy],
                    patch_size=(ps, 512),
                    stride=(stride, 512),
                )
                data_patches, _, _, _, masks,_ = gen_patches(
                    resampled_data,
                    [rx, ry, sx, mask],
                    patch_size=(ps, 512),
                    stride=(stride, 512),
                )
                self.dataL.extend(data_patches)
                self.mask_dataL.extend(mask_patches)
                self.rxL.extend(rxs)
                self.ryL.extend(rys)
                self.sxL.extend(sxs)
                self.syL.extend(sys)
                self.maskL.extend(masks)
                kept_count += 1

        print(f"[SEG_C3NA] 总shot数: {total_count}")
        print(f"[SEG_C3NA] 保留: {kept_count}")
        print(f"[SEG_C3NA] 跳过 (NaN/Inf): {skipped_nan}")
        print(f"[SEG_C3NA] 跳过 (std=0): {skipped_std0}")
        print(f"[SEG_C3NA] 跳过 (shape错误): {skipped_shape}")

    def add_trace_missing(self, data, missing_rate, continuous=False):
        n_traces, n_samples = data.shape
        data_masked = data.copy()
        mask = np.ones_like(data, dtype=np.float32)
        
        # 假设8条测线，每条68个检波器
        n_lines = 8
        traces_per_line = 68
        
        # 验证数据形状是否匹配
        assert n_traces == n_lines * traces_per_line, f"Expected {n_lines * traces_per_line} traces, got {n_traces}"
        
        if continuous:  # 按整条测线缺失
            # 计算需要缺失的测线数量
            n_missing_lines = int(np.ceil(missing_rate * n_lines))
            n_missing_lines = min(n_missing_lines, n_lines)  # 防止超出
            
            if n_missing_lines <= 0:
                return data_masked, mask
            
            # 随机选择要缺失的测线索引
            missing_line_indices = np.random.choice(n_lines, size=n_missing_lines, replace=False)
            
            # 计算对应的道索引范围
            missing_indices = []
            for line_idx in missing_line_indices:
                start_trace = line_idx * traces_per_line
                end_trace = (line_idx + 1) * traces_per_line
                missing_indices.extend(range(start_trace, end_trace))
            
            missing_indices = np.array(missing_indices).astype(int)
            
        else:  # 随机缺失
            n_missing = int(n_traces * missing_rate)
            n_missing = min(n_missing, n_traces)
            
            if n_missing <= 0:
                return data_masked, mask
            missing_indices = np.random.choice(n_traces, size=n_missing, replace=False).astype(int)
        # 应用缺失
        mask[missing_indices, :] = 0.0
        data_masked[missing_indices, :] = 0.0
        return data_masked, mask

    def __len__(self):
        return len(self.dataL)

    def __getitem__(self, idx):
        data = self.dataL[idx]
        rx = self.rxL[idx]
        ry = self.ryL[idx]
        sx = self.sxL[idx]
        sy = self.syL[idx]
        data_masked = self.mask_dataL[idx]
        mask = self.maskL[idx]
        return (
            data.astype(np.float32),
            data_masked.astype(np.float32),
            rx.astype(np.float32),
            ry.astype(np.float32),
            sx.astype(np.float32),
            sy.astype(np.float32),
            mask.mean(axis=-1),
            mask,
        )


## marmousi 5d
class SlidingWindowDataset_patched(Dataset):
    def __init__(
        self,
        sx: np.ndarray,
        sy: np.ndarray,
        rx: np.ndarray,
        ry: np.ndarray,
        data: np.ndarray,
        time_size: int = 512,
        shot_win: Tuple[int, int] = (2, 2),
        rec_win: Tuple[int, int] = (5, 20),
        mask_value: float = 0.0,
        normalize: bool = False,
        missing_ratio: float = 0.0,
        missing_type: str = "none",  # 'shot' | 'receiver' | 'both' | 'none'
        missing_mode: str = "random",  # 'random' | 'block' | 'line' | 'center' | 'edge' | 'pattern'
        missing_seed: Optional[int] = None,
        user_shot_mask: Optional[np.ndarray] = None,
        user_rec_mask: Optional[np.ndarray] = None,
    ):
        self.sx, self.sy = sx, sy
        self.rx, self.ry = rx, ry

        # === 保留完整数据作为标签 ===
        self.data_full = data.copy()

        # === 输入数据：将缺失mask作用到data ===
        self.data = data.copy()
        self.Nsx, self.Nsy, self.Nrx, self.Nry, self.Nt = data.shape
        self.shot_win = shot_win
        self.rec_win = rec_win
        self.mask_value = mask_value
        self.missing_ratio = float(missing_ratio)
        self.missing_type = missing_type
        self.missing_mode = missing_mode
        self.missing_seed = missing_seed
        self.user_shot_mask = user_shot_mask
        self.user_rec_mask = user_rec_mask
        self.time_size = time_size

        rng = np.random.default_rng(missing_seed)

        # === Step 1: 生成缺失 mask（全局级别） ===
        # 语义更新：1 表示存在，0 表示缺失
        self.rec_missing_mask = np.ones((self.Nrx, self.Nry), dtype=np.float32)
        self.shot_missing_mask = np.ones((self.Nsx, self.Nsy), dtype=np.float32)

        ratio = np.clip(self.missing_ratio, 0.0, 1.0)

        if self.missing_mode == "pattern":
            if self.user_rec_mask is not None:
                m = self.user_rec_mask.copy()
                if m.dtype == bool:
                    m = (~m).astype(np.float32)  # 旧语义 True=缺失 -> 0
                else:
                    m = np.clip(m.astype(np.float32), 0.0, 1.0)
                self.rec_missing_mask = m
            if self.user_shot_mask is not None:
                m = self.user_shot_mask.copy()
                if m.dtype == bool:
                    m = (~m).astype(np.float32)
                else:
                    m = np.clip(m.astype(np.float32), 0.0, 1.0)
                self.shot_missing_mask = m

        elif self.missing_mode == "random":
            if self.missing_type in ("receiver", "both"):
                self.rec_missing_mask = (
                    rng.random((self.Nrx, self.Nry)) >= ratio
                ).astype(np.float32)
            if self.missing_type in ("shot", "both"):
                self.shot_missing_mask = (
                    rng.random((self.Nsx, self.Nsy)) >= ratio
                ).astype(np.float32)

        elif self.missing_mode == "line":
            # 按比例选择多条随机线（ratio=0 不缺失）
            if ratio > 0:
                if self.missing_type in ("receiver", "both"):
                    k = int(np.ceil(ratio * self.Nrx))
                    k = min(max(k, 0), self.Nrx)
                    if k > 0:
                        rows = rng.choice(self.Nrx, size=k, replace=False)
                        self.rec_missing_mask[rows, :] = 0.0
                if self.missing_type in ("shot", "both"):
                    k = int(np.ceil(ratio * self.Nsx))
                    k = min(max(k, 0), self.Nsx)
                    if k > 0:
                        rows = rng.choice(self.Nsx, size=k, replace=False)
                        self.shot_missing_mask[rows, :] = 0.0

        elif self.missing_mode == "block":
            # 按比例设置缺失块面积（ratio=0 不缺失）
            if ratio > 0:
                if self.missing_type in ("receiver", "both"):
                    sx = max(1, int(np.sqrt(ratio) * self.Nrx))
                    sy = max(1, int(np.sqrt(ratio) * self.Nry))
                    x0 = int(rng.integers(0, max(1, self.Nrx - sx + 1)))
                    y0 = int(rng.integers(0, max(1, self.Nry - sy + 1)))
                    self.rec_missing_mask[x0 : x0 + sx, y0 : y0 + sy] = 0.0
                if self.missing_type in ("shot", "both"):
                    sx_ = max(1, int(np.sqrt(ratio) * self.Nsx))
                    sy_ = max(1, int(np.sqrt(ratio) * self.Nsy))
                    x0_ = int(rng.integers(0, max(1, self.Nsx - sx_ + 1)))
                    y0_ = int(rng.integers(0, max(1, self.Nsy - sy_ + 1)))
                    self.shot_missing_mask[x0_ : x0_ + sx_, y0_ : y0_ + sy_] = 0.0

        elif self.missing_mode == "center":
            # 以中心为对称的缺失区域，边长按比例（ratio=0 不缺失）
            if ratio > 0:
                if self.missing_type in ("receiver", "both"):
                    sx_len = max(1, int(np.sqrt(ratio) * self.Nrx))
                    sy_len = max(1, int(np.sqrt(ratio) * self.Nry))
                    # 取奇数边长以保证中心对称
                    if sx_len % 2 == 0:
                        sx_len += 1
                    if sy_len % 2 == 0:
                        sy_len += 1
                    cx, cy = self.Nrx // 2, self.Nry // 2
                    x0 = max(0, cx - sx_len // 2)
                    y0 = max(0, cy - sy_len // 2)
                    x1 = min(self.Nrx, x0 + sx_len)
                    y1 = min(self.Nry, y0 + sy_len)
                    self.rec_missing_mask[x0:x1, y0:y1] = 0.0
                if self.missing_type in ("shot", "both"):
                    sx_len = max(1, int(np.sqrt(ratio) * self.Nsx))
                    sy_len = max(1, int(np.sqrt(ratio) * self.Nsy))
                    if sx_len % 2 == 0:
                        sx_len += 1
                    if sy_len % 2 == 0:
                        sy_len += 1
                    cx, cy = self.Nsx // 2, self.Nsy // 2
                    x0 = max(0, cx - sx_len // 2)
                    y0 = max(0, cy - sy_len // 2)
                    x1 = min(self.Nsx, x0 + sx_len)
                    y1 = min(self.Nsy, y0 + sy_len)
                    self.shot_missing_mask[x0:x1, y0:y1] = 0.0

        elif self.missing_mode == "edge":
            # 按比例扩展边缘厚度（ratio=0 不缺失）
            if self.missing_type in ("receiver", "both"):
                k = int(np.ceil(0.5 * ratio * self.Nrx))
                k = min(max(k, 0), self.Nrx // 2)
                if k > 0:
                    self.rec_missing_mask[:k, :] = 0.0
                    self.rec_missing_mask[-k:, :] = 0.0
            if self.missing_type in ("shot", "both"):
                k = int(np.ceil(0.5 * ratio * self.Nsy))
                k = min(max(k, 0), self.Nsy // 2)
                if k > 0:
                    self.shot_missing_mask[:, :k] = 0.0
                    self.shot_missing_mask[:, -k:] = 0.0
        ## 保存mask
        os.makedirs("./shot_masks", exist_ok=True)
        np.save(
            os.path.join("./shot_masks", f"shot_mask_{self.missing_mode}.npy"),
            self.shot_missing_mask,
        )
        np.save(
            os.path.join("./shot_masks", f"rec_mask_{self.missing_mode}.npy"),
            self.rec_missing_mask,
        )
        # === Step 2: 应用缺失 mask 到输入数据 ===
        for irx in range(self.Nrx):
            for iry in range(self.Nry):
                if self.rec_missing_mask[irx, iry] == 0.0:
                    self.data[:, :, irx, iry, :] = self.mask_value
        for isx in range(self.Nsx):
            for isy in range(self.Nsy):
                if self.shot_missing_mask[isx, isy] == 0.0:
                    self.data[isx, isy, :, :, :] = self.mask_value

        # === Step 3: 构建炮点+检波点窗口索引（滑窗） ===
        self.index = []
        for i in range(self.Nsx - self.shot_win[0] * 2):
            for j in range(self.Nsy - self.shot_win[1] * 2):
                shot_x0 = i
                shot_x1 = i + 2 * self.shot_win[0] + 1
                shot_y0 = j
                shot_y1 = j + 2 * self.shot_win[1] + 1

                for p in range(self.Nrx - self.rec_win[0] * 2):
                    for q in range(self.Nry - self.rec_win[1] * 2):
                        rec_x0 = p
                        rec_x1 = p + 2 * self.rec_win[0] + 1
                        rec_y0 = q
                        rec_y1 = q + 2 * self.rec_win[1] + 1

                        self.index.append(
                            (
                                shot_x0,
                                shot_x1,
                                shot_y0,
                                shot_y1,
                                rec_x0,
                                rec_x1,
                                rec_y0,
                                rec_y1,
                            )
                        )
        self.dataL = []
        self.mask_dataL = []
        self.rxL = []
        self.ryL = []
        self.sxL = []
        self.syL = []
        self.maskL = []
        for i in range(len(self.index)):
            shot_x0, shot_x1, shot_y0, shot_y1, rec_x0, rec_x1, rec_y0, rec_y1 = (
                self.index[i]
            )
            # === 输入（带缺失） ===
            gather_in = self.data[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1, :
            ].copy()
            # === 标签（完整数据） ===
            gather_label = self.data_full[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1, :
            ].copy()
            # 子 mask
            rec_missing_sub = self.rec_missing_mask[rec_x0:rec_x1, rec_y0:rec_y1]
            shot_missing_sub = self.shot_missing_mask[shot_x0:shot_x1, shot_y0:shot_y1]
            # 逐道展平：得到 [N_traces, Nt]
            n_sx = shot_x1 - shot_x0
            n_sy = shot_y1 - shot_y0
            n_rx = rec_x1 - rec_x0
            n_ry = rec_y1 - rec_y0

            traces_in = signal.resample(
                normalize(gather_in.reshape(-1, self.Nt)), self.time_size, axis=1
            )
            traces_label = signal.resample(
                normalize(gather_label.reshape(-1, self.Nt)), self.time_size, axis=1
            )

            # 生成与每一道对应的坐标（与展平顺序一致，采用C顺序 Sx->Sy->Rx->Ry）
            sx_sub = self.sx[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1
            ]
            sy_sub = self.sy[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1
            ]
            rx_sub = self.rx[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1
            ]
            ry_sub = self.ry[
                shot_x0:shot_x1, shot_y0:shot_y1, rec_x0:rec_x1, rec_y0:rec_y1
            ]

            sx_traces = sx_sub.reshape(-1)
            sy_traces = sy_sub.reshape(-1)
            rx_traces = rx_sub.reshape(-1)
            ry_traces = ry_sub.reshape(-1)

            # 每一道的有效性mask（Sx,Sy,Rx,Ry）-> 展平
            shot_mask = shot_missing_sub[:, :, None, None] == 1.0
            rec_mask = rec_missing_sub[None, None, :, :] == 1.0
            valid_mask_4d = np.logical_and(shot_mask, rec_mask)
            valid_mask = valid_mask_4d.reshape(-1)

            ##get patches
            data_patches, rx_patches, ry_patches, sx_patches, sy_patches, _ = gen_patches(
                traces_label,
                [rx_traces, ry_traces, sx_traces, sy_traces],
                (128, 128),
                (64, 64),
            )
            masked_data_patches, _, _, _, masked_patches, _ = gen_patches(
                traces_in,
                [rx_traces, ry_traces, sx_traces, valid_mask],
                (128, 128),
                (64, 64),
            )
            self.dataL.extend(data_patches)
            self.mask_dataL.extend(masked_data_patches)
            self.rxL.extend(rx_patches)
            self.ryL.extend(ry_patches)
            self.sxL.extend(sx_patches)
            self.syL.extend(sy_patches)
            self.maskL.extend(masked_patches)

    def __len__(self):
        return len(self.dataL)

    def __getitem__(self, idx):

        data_dict = {
            "traces_in": torch.tensor(
                self.mask_dataL[idx], dtype=torch.float32
            ),  # [N_traces, Nt]
            "traces_label": torch.tensor(
                self.dataL[idx], dtype=torch.float32
            ),  # [N_traces, Nt]
            "rx": torch.tensor(self.rxL[idx], dtype=torch.float32),  # [N_traces]
            "ry": torch.tensor(self.ryL[idx], dtype=torch.float32),  # [N_traces]
            "sx": torch.tensor(self.sxL[idx], dtype=torch.float32),  # [N_traces]
            "sy": torch.tensor(self.syL[idx], dtype=torch.float32),  # [N_traces]
            "valid_mask": torch.tensor(
                self.maskL[idx], dtype=torch.bool
            ).float(),  # [N_traces]
            "shot_missing_mask": torch.tensor(
                self.shot_missing_mask, dtype=torch.float32
            ),
            "rec_missing_mask": torch.tensor(
                self.rec_missing_mask, dtype=torch.float32
            ),
            "shape_info": torch.tensor([128, 128, 64, 64, self.Nt], dtype=torch.int32),
        }
        return (
            data_dict["traces_label"],
            data_dict["traces_in"],
            data_dict["rx"],
            data_dict["ry"],
            data_dict["sx"],
            data_dict["sy"],
            data_dict["valid_mask"],
            data_dict["valid_mask"],
            
        )
# field data
import  binning       
## dongfang
class RAW_DONGFANG(Dataset):
    def __init__(self,root,file_path,time_ps,trace_ps,time_sd,trace_sd,train:bool):
        super().__init__()
        print('load data...')
        self.train =train
        D,H = binning.load_segy_data(file_path)
        pos = np.vstack([H['sx'], H['sy']]).T   # [n_traces, 2]
        ind = np.vstack([H['tatyp'], H['cdpt']]).T
        rec =np.vstack([H['rx'], H['ry']]).T
        recv_dict = defaultdict(list)
        for i, (rx, ry) in enumerate(rec):
            recv_dict[tuple((rx.item(), ry.item()))].append(i)
        print(f"Found {len(recv_dict)} unique receivers")
        dataL=[]
        rxL=[]
        ryL=[]
        sxL=[]
        syL=[]
        maskL=[]
        if train:
            data_begin =0
            data_num =len(recv_dict.keys())*0.8
        else:
            data_begin =int(len(recv_dict.keys())*0.8)
            data_num =len(recv_dict.keys())
        data_num =int(data_num)
        if train:
            for key, value in tqdm(list(recv_dict.items())[data_begin:data_begin+data_num:1]):
                pos =np.load(os.path.join(root,f'position_info_rx{key[0]}_ry{key[1]}.npy'),allow_pickle=True).item()
                sx=pos['X_grid'].reshape(-1)
                sy=pos['Y_grid'].reshape(-1)
                L = sx.shape
                rx=key[0]+np.zeros(L)
                ry=key[1]+np.zeros(L)
                data =np.load(os.path.join(root,f'data_regular_rx{key[0]}_ry{key[1]}.npy'),allow_pickle=True)
                T,Tr_h,Tr_w = data.shape
                data =data.reshape(T,Tr_h*Tr_w)
                data =data.T
                data_patches,rx_patches,ry_patches,sx_patches,sy_patches, _ = gen_patches(data, [rx, ry, sx, sy], (trace_ps,trace_ps), (trace_sd,trace_sd))
                dataL.extend(data_patches)
                mask =np.load(os.path.join(root,f'data_missing_rx{key[0]}_ry{key[1]}.npy'),allow_pickle=True).reshape(T,Tr_h*Tr_w)
                mask=mask.T
                mask_patches,_,_,_,_, _ = gen_patches(mask, [rx, ry, sx, sy], (trace_ps,trace_ps), (trace_sd,trace_sd))
                maskL.extend(mask_patches)
                sxL.extend(sx_patches)
                syL.extend(sy_patches)
                rxL.extend(rx_patches)
                ryL.extend(ry_patches)
        else:
            # Recalculate recv_keys for test split
            recv_keys = list(recv_dict.keys())
            selected_recv_keys = recv_keys[data_begin:data_begin+data_num:1]
            for key in tqdm(selected_recv_keys, desc="Test loading"):
                pos = np.load(os.path.join(root, f'position_info_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True).item()
                sx_full = pos['X_grid']  # (H, W)
                sy_full = pos['Y_grid']  # (H, W)

                data_full = np.load(os.path.join(root, f'data_regular_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True)  # (T, H, W)
                mask_full = np.load(os.path.join(root, f'data_missing_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True) # (T, H, W)
                H, W = sx_full.shape  # H: 炮线数, W: 每条线上的检波点数
                # 随机选择 4 条炮线
                chosen_lines = np.random.choice(H, size=min(5, H), replace=False)
                # 提取选中的炮线数据
                sx_selected = sx_full[chosen_lines, :].flatten()  # (4 * W,)
                sy_selected = sy_full[chosen_lines, :].flatten()  # (4 * W,)
                data_selected = data_full[:, chosen_lines, :].reshape(data_full.shape[0], -1).T  # (4 * W, T)
                mask_selected = mask_full[:, chosen_lines, :].reshape(mask_full.shape[0], -1).T # (4 * W, T)
                dataL.append(data_selected)
                maskL.append(mask_selected)
                sxL.append(sx_selected)
                syL.append(sy_selected)
                rxL.append(key[0] + np.zeros_like(sx_selected))
                ryL.append(key[1] + np.zeros_like(sy_selected))
        self.dataL=dataL
        self.rxL=rxL
        self.ryL=ryL
        self.sxL=sxL
        self.syL=syL
        self.maskL=maskL
        print(f'load data done, number of data:{len(self.dataL)}')
    def __len__(self):
        return len(self.dataL)

    def __getitem__(self, idx):
        data = self.dataL[idx]
        data_masked = self.maskL[idx]
        rx = self.rxL[idx]
        ry = self.ryL[idx]
        sx = self.sxL[idx]
        sy = self.syL[idx]
        ##normalize
        data =normalize(data)
        data_masked =normalize(data_masked)
        if self.train:
            rx,ry,sx,sy =augment_coordinates(rx,ry,sx,sy)
        return (
            data.astype(np.float32),
            data_masked.astype(np.float32),
            rx.astype(np.float32),
            ry.astype(np.float32),
            sx.astype(np.float32),
            sy.astype(np.float32),
            rx.mean(axis=-1),
            rx,
        )

def gen_2d_spatial_patches(data_3d, pos_2d_dict=None, spatial_ps=(8, 8), spatial_sd=(4, 4)):
    """
    Generate 2D patches on the HxW grid (e.g., source grid).
    :param data_3d: (T, H, W) - 3D seismic data or mask
    :param pos_2d_dict: dict with keys like 'sx', 'sy', values are (H, W) arrays. Optional.
    :param spatial_ps: spatial patch size (h, w)
    :param spatial_sd: spatial stride (sh, sw)
    :return: list of (T, patch_h, patch_w), list of corresponding pos dicts (or list of Nones if pos_2d_dict is None)
    """
    T, H, W = data_3d.shape
    patch_h, patch_w = spatial_ps
    stride_h, stride_w = spatial_sd

    patches_data = []
    patches_pos = []

    for i in range(0, H - patch_h + 1, stride_h):
        for j in range(0, W - patch_w + 1, stride_w):
            patch_data = data_3d[:, i:i+patch_h, j:j+patch_w] # (T, patch_h, patch_w)
            patches_data.append(patch_data)
            if pos_2d_dict is not None:
                patch_pos = {k: v[i:i+patch_h, j:j+patch_w] for k, v in pos_2d_dict.items()} # {k: (patch_h, patch_w)}
                patches_pos.append(patch_pos)
            else:
                patches_pos.append(None)

    return patches_data, patches_pos

class RAW_DONGFANG_Sliding(Dataset):
    def __init__(self, root, file_path, time_ps, trace_ps, time_sd, trace_sd, train: bool):
        super().__init__()
        print('load data...')
        self.train = train
        self.time_ps = time_ps
        self.trace_ps = trace_ps  # Used for final 1D patching after flattening
        self.time_sd = time_sd
        self.trace_sd = trace_sd # Used for final 1D patching after flattening

        # --- Step 1: Load global SEGY and build recv_dict (same as before) ---
        D, H = binning.load_segy_data(file_path)
        pos = np.vstack([H['sx'], H['sy']]).T   # [n_traces, 2]
        ind = np.vstack([H['tatyp'], H['cdpt']]).T
        rec = np.vstack([H['rx'], H['ry']]).T
        recv_dict = defaultdict(list)
        for i, (rx, ry) in enumerate(rec):
            recv_dict[tuple((rx.item(), ry.item()))].append(i)
        print(f"Found {len(recv_dict)} unique receivers")

        # --- Step 2: Get unique receiver coordinates (same as your logic) ---
        rx_unique = sorted(list(set([k[0] for k in recv_dict.keys()])))
        ry_unique = sorted(list(set([k[1] for k in recv_dict.keys()])))
        Rx, Ry = len(rx_unique), len(ry_unique)
        print(f"Unique receiver grid: Rx={Rx}, Ry={Ry}")

        dataL = []
        rxL = []
        ryL = []
        sxL = []
        syL = []
        maskL = []

        # --- Step 3: Calculate train/val split indices based on receiver grid size ---
        if train:
            rx_begin, rx_end = 0, int(Rx * 0.8)
            ry_begin, ry_end = 0, int(Ry * 0.8)
        else:
            rx_begin, rx_end = int(Rx * 0.8), Rx
            ry_begin, ry_end = int(Ry * 0.8), Ry
        if train:
            # --- Step 4: Iterate through receiver grid windows (your logic) ---
            rx_ps, rx_sd = 4, 2 # Reusing for Rx window. Consider separate params.
            ry_ps, ry_sd = 4, 2 # Reusing for Ry window. Consider separate params.
            # NOTE: Consider adding rx_ps, rx_sd, ry_ps, ry_sd to __init__ for more control
            print(f"Receiver Rx window: {rx_begin}-{rx_end}, Ry window: {ry_begin}-{ry_end}")
            for rx_i in tqdm(range(rx_begin, max(rx_begin+1, rx_end - rx_ps + 1), rx_sd), desc="Receiver Rx sliding"):
                for ry_j in range(ry_begin, max(ry_begin+1, ry_end - ry_ps + 1), ry_sd):
                    # Current receiver window coordinates
                    rx_window = rx_unique[rx_i:rx_i + rx_ps]
                    ry_window = ry_unique[ry_j:ry_j + ry_ps]
                    recv_window_keys = [(rx, ry) for rx in rx_window for ry in ry_window if (rx, ry) in recv_dict]
                    # --- NEW: Iterate through each receiver in the window ---
                    all_flattened_patches_data = []
                    all_flattened_patches_mask = []
                    all_flattened_patches_sx = []
                    all_flattened_patches_sy = []
                    all_flattened_patches_rx = [] # These will be repeated for each spatial patch of this receiver
                    all_flattened_patches_ry = [] # These will be repeated for each spatial patch of this receiver

                    for (rx, ry) in recv_window_keys:
                        data_path = os.path.join(root, f'data_regular_rx{rx}_ry{ry}.npy')
                        mask_path = os.path.join(root, f'data_missing_rx{rx}_ry{ry}.npy')
                        pos_path  = os.path.join(root, f'position_info_rx{rx}_ry{ry}.npy')

                        if not os.path.exists(data_path):
                            print(f"Warning: Data file missing {data_path}")
                            continue
                        data = np.load(data_path, allow_pickle=True)  # (T, H, W)
                        mask = np.load(mask_path, allow_pickle=True) # (T, H, W)
                        pos  = np.load(pos_path, allow_pickle=True).item()
                        sx = pos['X_grid']  # (H, W) - source X grid
                        sy = pos['Y_grid']  # (H, W) - source Y grid

                        # --- Step 5: For THIS receiver, generate 2D spatial patches on (H, W) grid ---
                        spatial_ps = (8, 8) # Reusing trace_ps for spatial patch size. Consider separate params.
                        spatial_sd = (8, 8) # Reusing trace_sd for spatial stride. Consider separate params.
                        # NOTE: Consider adding src_ps, src_sd (or spatial_ps, spatial_sd) to __init__ for more control

                        pos_2d_dict = {'sx': sx, 'sy': sy}
                        data_spatial_patches, pos_spatial_patches = gen_2d_spatial_patches(
                            data, pos_2d_dict, spatial_ps, spatial_sd
                        )
                        mask_spatial_patches, _ = gen_2d_spatial_patches( # Now only returns data patches for mask
                            mask, None, spatial_ps, spatial_sd # Pass None for pos_2d_dict when processing mask
                        )

                        # --- Step 6: Flatten each spatial patch for this receiver ---
                        for data_sp, mask_sp, pos_sp in zip(data_spatial_patches, mask_spatial_patches, pos_spatial_patches):
                            T, patch_h, patch_w = data_sp.shape
                            # Flatten spatial dims: (T, patch_h, patch_w) -> (T, patch_h * patch_w) -> (patch_h * patch_w, T)
                            data_flat = data_sp.reshape(T, patch_h * patch_w).T # (N_spatial_flat, T)
                            mask_flat = mask_sp.reshape(T, patch_h * patch_w).T # (N_spatial_flat, T)

                            # Flatten coordinates for this spatial patch
                            sx_flat = pos_sp['sx'].flatten() # (N_spatial_flat,)
                            sy_flat = pos_sp['sy'].flatten() # (N_spatial_flat,)
                            rx_flat = np.full_like(sx_flat, rx) # (N_spatial_flat,)
                            ry_flat = np.full_like(sy_flat, ry) # (N_spatial_flat,)

                            all_flattened_patches_data.append(data_flat)
                            all_flattened_patches_mask.append(mask_flat)
                            all_flattened_patches_sx.append(sx_flat)
                            all_flattened_patches_sy.append(sy_flat)
                            all_flattened_patches_rx.append(rx_flat)
                            all_flattened_patches_ry.append(ry_flat)

                    # --- Step 7: Concatenate flattened patches from ALL receivers in the window ---
                    if not all_flattened_patches_data:
                        continue # Skip this receiver window if no data found

                    # Concatenate along the first axis (N_spatial_flat axis)
                    combined_data_flattened = np.concatenate(all_flattened_patches_data, axis=0) # (N_combined, T)
                    combined_mask_flattened = np.concatenate(all_flattened_patches_mask, axis=0) # (N_combined, T)
                    combined_sx_flattened = np.concatenate(all_flattened_patches_sx) # (N_combined,)
                    combined_sy_flattened = np.concatenate(all_flattened_patches_sy) # (N_combined,)
                    combined_rx_flattened = np.concatenate(all_flattened_patches_rx) # (N_combined,)
                    combined_ry_flattened = np.concatenate(all_flattened_patches_ry) # (N_combined,)

                    print(f"  Processing combined block from window rx_i={rx_i}, ry_j={ry_j}: shape {combined_data_flattened.shape}") # Debug print

                    # --- Step 8: Generate final 1D patches from the combined flattened sequence ---
                    try:
                        data_final_patches, rx_final_patches, ry_final_patches, sx_final_patches, sy_final_patches, _ = gen_patches(
                            combined_data_flattened, [combined_rx_flattened, combined_ry_flattened, combined_sx_flattened, combined_sy_flattened],
                            (self.trace_ps, self.trace_ps), (self.trace_sd, self.trace_sd)
                        )
                        mask_final_patches, _, _, _, _, _ = gen_patches(
                            combined_mask_flattened, [combined_rx_flattened, combined_ry_flattened, combined_sx_flattened, combined_sy_flattened],
                            (self.trace_ps, self.trace_ps), (self.trace_sd, self.trace_sd)
                        )
                    except Exception as e:
                        print(f"Error in gen_patches for receiver window rx_i={rx_i}, ry_j={ry_j}, combined shape {combined_data_flattened.shape}: {e}")
                        continue # Skip this combined block if gen_patches fails

                    dataL.extend(data_final_patches)
                    maskL.extend(mask_final_patches)
                    sxL.extend(sx_final_patches)
                    syL.extend(sy_final_patches)
                    rxL.extend(rx_final_patches)
                    ryL.extend(ry_final_patches)

        # --- Test Logic (unchanged from previous version) ---
        else:
            # Recalculate recv_keys for test split
            recv_keys = list(recv_dict.keys())
            selected_recv_keys = recv_keys[rx_begin:rx_end]

            for key in tqdm(selected_recv_keys, desc="Test loading"):
                pos = np.load(os.path.join(root, f'position_info_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True).item()
                sx_full = pos['X_grid']  # (H, W)
                sy_full = pos['Y_grid']  # (H, W)

                data_full = np.load(os.path.join(root, f'data_regular_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True)  # (T, H, W)
                mask_full = np.load(os.path.join(root, f'data_missing_rx{key[0]}_ry{key[1]}.npy'), allow_pickle=True) # (T, H, W)

                H, W = sx_full.shape  # H: 炮线数, W: 每条线上的检波点数

                # 随机选择 4 条炮线
                chosen_lines = np.random.choice(H, size=min(10, H), replace=False)

                # 提取选中的炮线数据
                sx_selected = sx_full[chosen_lines, :].flatten()  # (4 * W,)
                sy_selected = sy_full[chosen_lines, :].flatten()  # (4 * W,)
                data_selected = data_full[:, chosen_lines, :].reshape(data_full.shape[0], -1).T  # (4 * W, T)
                mask_selected = mask_full[:, chosen_lines, :].reshape(mask_full.shape[0], -1).T # (4 * W, T)
                dataL.append(data_selected)
                maskL.append(mask_selected)
                sxL.append(sx_selected)
                syL.append(sy_selected)
                rxL.append(key[0] + np.zeros_like(sx_selected))
                ryL.append(key[1] + np.zeros_like(sy_selected))
        self.dataL = dataL
        self.rxL = rxL
        self.ryL = ryL
        self.sxL = sxL
        self.syL = syL
        self.maskL = maskL
        print(f'load data done, number of items: {len(self.dataL)}')

    def __len__(self):
        return len(self.dataL)

    def __getitem__(self, idx):
        data = self.dataL[idx]
        data_masked = self.maskL[idx]
        rx = self.rxL[idx]
        ry = self.ryL[idx]
        sx = self.sxL[idx]
        sy = self.syL[idx]

        # normalize
        data = normalize(data)
        data_masked = normalize(data_masked)

        if self.train:
            rx, ry, sx, sy = augment_coordinates(rx, ry, sx, sy)

        return (
            data.astype(np.float32),
            data_masked.astype(np.float32),
            rx.astype(np.float32),
            ry.astype(np.float32),
            sx.astype(np.float32),
            sy.astype(np.float32),
            rx.mean(axis=-1), # 可选
            rx, # 可选
        )
def rotate_points(x, y):
    """
    将炮点旋转，使主要方向水平化
    """
    x_mean, y_mean = np.mean(x), np.mean(y)
    x0, y0 = x - x_mean, y - y_mean
    coords = np.vstack([x0, y0])
    cov = np.cov(coords)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]  # 主方向向量
    theta = np.arctan2(main_dir[1], main_dir[0])
    rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                    [np.sin(-theta),  np.cos(-theta)]])
    rotated = rot @ coords
    return rotated,theta

def sample_missing_ratio(a=2.0, b=5.0, min_r=0.4, max_r=0.6):
    r = np.random.beta(a, b)
    return min_r + (max_r - min_r) * r

def sample_missing_ratio_block(a=2.0, b=5.0, min_r=0.2, max_r=0.4):
    r = np.random.beta(a, b)
    return min_r + (max_r - min_r) * r

def apply_random_missing(traces, missing_ratio):
    n_traces = traces.shape[0]
    miss_idx = np.random.choice(n_traces, int(n_traces * missing_ratio), replace=False)
    mask = np.ones_like(traces, dtype=np.float32)
    mask[miss_idx, :] = 0.0
    return traces * mask, mask

def apply_random_missing(traces, missing_ratio):
    n_traces = traces.shape[0]
    n_samples = traces.shape[1]
    trace_mask = np.random.choice([0, 1], size=(n_traces, 1), 
                                 p=[missing_ratio, 1-missing_ratio], replace=True)
    mask = np.ones((n_traces, n_samples), dtype=np.float32) * trace_mask
    return traces * mask, mask


def apply_large_spacing_missing(traces, missing_ratio):
    n_traces = traces.shape[0]
    exact_missing_count = int(n_traces * missing_ratio)
    
    if exact_missing_count == 0:
        mask = np.ones_like(traces, dtype=np.float32)
        return traces * mask, mask
    spacing = max(2, n_traces // exact_missing_count)
    for s in range(spacing, 1, -1):
        for start in range(s):
            if len(range(start, n_traces, s)) == exact_missing_count:
                miss_idx = np.arange(start, n_traces, s)
                mask = np.ones_like(traces, dtype=np.float32)
                mask[miss_idx, :] = 0.0
                return traces * mask, mask
    start = np.random.randint(0, spacing)
    miss_idx = np.arange(start, n_traces, spacing)[:exact_missing_count]
    if len(miss_idx) < exact_missing_count:
        available = set(range(n_traces)) - set(miss_idx)
        miss_idx = np.append(miss_idx, list(available)[:exact_missing_count-len(miss_idx)])
    
    mask = np.ones_like(traces, dtype=np.float32)
    mask[miss_idx, :] = 0.0
    return traces * mask, mask

def apply_mixed_missing(traces, missing_ratio=0.5,
                        a_cont=2.0, b_cont=5.0,
                        a_block=2.0, b_block=5.0,
                        max_blocks=5):
    n = traces.shape[0]
    total_miss = int(n * missing_ratio)
    if total_miss <= 0:
        mask = np.ones_like(traces, dtype=np.float32)
        return traces, mask
    p_continuous = sample_missing_ratio(a_cont, b_cont, 0.15, 0.5)  
    num_blocks = int(1 + sample_missing_ratio(a_block, b_block, 0, 1) * (max_blocks - 1))  
    cont_miss = int(total_miss * p_continuous)
    rand_miss = total_miss - cont_miss
    mask = np.ones_like(traces, dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    if cont_miss > 0 and num_blocks > 0:
        avg_block = max(1, cont_miss // num_blocks)
        lengths = np.random.randint(max(1, avg_block // 2),
                                    avg_block * 3 // 2 + 1,
                                    size=num_blocks)
        total_len = min(np.sum(lengths), cont_miss)
        scale = cont_miss / total_len
        lengths = np.maximum(1, (lengths * scale).astype(int))
        for L in lengths:
            free = np.where(~used)[0]
            if len(free) < L:
                break
            start = np.random.choice(free[:-L+1])
            end = start + L
            mask[start:end] = 0
            used[start:end] = True
    remaining = np.where(~used)[0]
    if rand_miss > 0 and len(remaining) > 0:
        miss_idx = np.random.choice(remaining, min(rand_miss, len(remaining)), replace=False)
        mask[miss_idx] = 0
    return traces * mask, mask

class RAW_dongfang(Dataset):
    def __init__(self, root, file_path, time_ps, trace_ps, trace_sd, time_sd, train=True,
                 test_shot_lines: Optional[List[int]] = None,
                 test_num_lines: int = 4,
                 test_seed: Optional[int] = None):
        super().__init__()
        print('Using fixed missing ratio')
        self.root = root
        self.train = train
        self.time_ps, self.trace_ps, self.trace_sd, self.time_sd = time_ps, trace_ps, trace_sd, time_sd
        # 测试时固定选择的炮线设置
        self.test_num_lines = max(1, int(test_num_lines))
        self._rng = np.random.default_rng(test_seed) if test_seed is not None else np.random.default_rng()
        self.test_shot_lines = None if test_shot_lines is None else np.array(sorted(set(test_shot_lines)), dtype=int)

        # ===== Load basic SEG-Y =====
        D, H = binning.load_segy_data(file_path)
        shot = np.vstack([H['sx'], H['sy']]).T
        rec = np.vstack([H['rx'], H['ry']]).T
        shot_rot, _ = rotate_points(shot[:, 0] / 10.0, shot[:, 1] / 10.0)
        recv_dict = defaultdict(list)
        for i, (rx, ry) in enumerate(rec):
            recv_dict[(rx.item(), ry.item())].append(i)
        print(f"Found {len(recv_dict)} receiver gathers.")

        self.D = D
        self.shot_rot = shot_rot
        self.recv_dict = recv_dict
        self.recv_keys = list(recv_dict.keys())
        self.sample_map = []  
        if train:
            for rx, ry in tqdm(self.recv_keys, desc="Counting patches"):
                idx_list = np.array(recv_dict[(rx, ry)])
                traces = normalize_clip(D[:, idx_list].T)
                sx = shot_rot[0,idx_list]
                sy = shot_rot[1,idx_list]
                rx_arr = np.full_like(sx, rx)
                ry_arr = np.full_like(sy, ry)
                patches, *_ = gen_patches(
                    traces, [rx_arr, ry_arr, sx, sy],
                    (self.trace_ps, self.time_ps),
                    (self.trace_sd, self.time_sd)
                )
                for p_idx in range(len(patches)):
                    self.sample_map.append((rx, ry, p_idx))
        else:
            for rx, ry in tqdm(self.recv_keys, desc="Indexing test data"):
                pos_path = os.path.join(root, f'position_info_rx{rx}_ry{ry}.npy')
                if os.path.exists(pos_path):
                    self.sample_map.append((rx, ry, 0))  
        print(f"Total patches: {len(self.sample_map)}")

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        rx, ry, local_idx = self.sample_map[idx]
        if self.train:
            idx_list = np.array(self.recv_dict[(rx, ry)])
            traces = normalize_clip(self.D[:, idx_list].T)
            sx = self.shot_rot[0,idx_list]
            sy = self.shot_rot[1,idx_list]
            rx_arr = np.full_like(sx, rx)
            ry_arr = np.full_like(sy, ry)
            missing_ratio = sample_missing_ratio()
            # traces_masked, _ = apply_random_missing(traces,missing_ratio)
            patches, rx_p, ry_p, sx_p, sy_p, _ = gen_patches(
                traces, [rx_arr, ry_arr, sx, sy],
                (self.trace_ps, self.time_ps),
                (self.trace_sd, self.time_sd)
            )
            patch = patches[local_idx]  
            rx_p, ry_p, sx_p, sy_p = rx_p[local_idx], ry_p[local_idx], sx_p[local_idx], sy_p[local_idx]
            masked, _ = apply_random_missing(patch, missing_ratio)
            # augment
            if self.train:
                rx_p, ry_p, sx_p, sy_p = augment_coordinates(rx_p, ry_p, sx_p, sy_p)
            return (
                patch.astype(np.float32),
                masked.astype(np.float32),
                rx_p.astype(np.float32),
                ry_p.astype(np.float32),
                sx_p.astype(np.float32),
                sy_p.astype(np.float32),
                rx_p.mean().astype(np.float32),
                rx_p,
            )
        else:
            pos_path = os.path.join(self.root, f'position_info_rx{rx}_ry{ry}.npy')
            data_path = os.path.join(self.root, f'data_regular_rx{rx}_ry{ry}.npy')
            mask_path = os.path.join(self.root, f'data_missing_rx{rx}_ry{ry}.npy')

            pos = np.load(pos_path, allow_pickle=True).item()
            sx_full = pos['X_grid']
            sy_full = pos['Y_grid']
            data_full = np.load(data_path, allow_pickle=True)
            mask_full = np.load(mask_path, allow_pickle=True)

            data_full=normalize_clip(data_full)
            mask_full=normalize_clip(mask_full)

            H, W = sx_full.shape
            if self.test_shot_lines is not None:
                chosen_lines = self.test_shot_lines[(self.test_shot_lines >= 0) & (self.test_shot_lines < H)]
                if chosen_lines.size == 0:
                    chosen_lines = np.arange(min(self.test_num_lines, H), dtype=int)
            else:
                k = min(self.test_num_lines, H)
                chosen_lines = self._rng.choice(H, size=k, replace=False)
            sx_selected = sx_full[chosen_lines, :].flatten()
            sy_selected = sy_full[chosen_lines, :].flatten()
            data_selected = data_full[:, chosen_lines, :].reshape(data_full.shape[0], -1).T
            mask_selected = mask_full[:, chosen_lines, :].reshape(mask_full.shape[0], -1).T

            traces = data_selected
            # traces= gather_time_window(traces,target_len=1024)
            masked = mask_selected
            # masked = gather_time_window(masked,target_len=1024)
            rx_arr = np.full_like(sx_selected, rx, dtype=np.float32)
            ry_arr = np.full_like(sy_selected, ry, dtype=np.float32)
            return (
                traces.astype(np.float32),
                masked.astype(np.float32),
                rx_arr.astype(np.float32),
                ry_arr.astype(np.float32),
                sx_selected.astype(np.float32),
                sy_selected.astype(np.float32),
                np.mean(rx_arr).astype(np.float32),
                rx_arr,
            )

##for dongfang_dataV2
def apply_random_missing(traces, missing_ratio):
    n_traces, n_samples = traces.shape
    trace_mask = np.random.choice(
        [0, 1], size=(n_traces, 1),
        p=[missing_ratio, 1 - missing_ratio], replace=True
    )
    mask = np.ones((n_traces, n_samples), dtype=np.float32) * trace_mask
    return traces * mask, mask


def apply_block_missing(traces,):
    n_traces, n_samples = traces.shape
    missing_ratio = sample_missing_ratio_block()
    mask = np.ones((n_traces, n_samples), dtype=np.float32)
    n_missing = int(n_traces * missing_ratio)
    if n_missing > 0:
        start = np.random.randint(0, max(1, n_traces - n_missing))
        mask[start:start + n_missing, :] = 0.0
    return traces * mask, mask



def apply_mixed_mask(traces, missing_ratio, block_prob=0.4):
    """混合缺失模式"""
    if np.random.rand() < block_prob:
        return apply_block_missing(traces,)
    else:
        return apply_random_missing(traces, missing_ratio)

class RAW_dongfang_1031(Dataset):
    def __init__(self, label_file_path, time_ps, trace_ps, trace_sd, time_sd, train=True):
        super().__init__()
        print('Loading Dongfang dataset...')
        raw_data_dir = Path("./dongfang/raw")
        label_cleaned_dir = Path(
            "/home/czt/seismic_ddpm/Seis_DiT/dongfang/label/cleaned_data"
        )
        mask_dir = Path("./dongfang/aligned_raw_data")
        pattern = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        label_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            match = pattern.match(file.name)
            if match:
                recv_line = int(match.group(1))
                recv_no = int(match.group(2))
                label_files.append((recv_line, recv_no, file))

        dataL, rxL, ryL, sxL, syL, maskL = [], [], [], [], [], []
        self._rng = np.random.default_rng(123)
        self.train = train

        # ------------------------
        # 训练阶段：仅保存原始完整patch，不提前mask
        # ------------------------
        if train:
            stats =self.compute_coord_stats(label_files,raw_data_dir)
            for recv_line, recv_no, file in tqdm(label_files, desc="Loading training data"):
                raw_data_file = raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                raw_attrs_file = raw_data_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                if not raw_data_file.exists() or not raw_attrs_file.exists():
                    continue
                raw_data = np.load(raw_data_file)
                raw_attrs = np.load(raw_attrs_file)
                sx_old = raw_attrs["shot_x"]
                sy_old = raw_attrs["shot_y"]
                rx_old = raw_attrs["rec_x"]
                ry_old = raw_attrs["rec_y"]
                         
                #normalize  
                sx =2*(sx_old-stats["sx_min"])/(stats['sx_max']-stats['sx_min'])-1
                sy =2*(sy_old-stats["sy_min"])/(stats['sy_max']-stats['sy_min'])-1
                rx =2*(rx_old-stats["rx_min"])/(stats['rx_max']-stats['rx_min'])-1
                ry =2*(ry_old-stats["ry_min"])/(stats['ry_max']-stats['ry_min'])-1
                if False:
                    plt.hist(sx,bins=100,alpha=0.5,label='sx')
                    plt.hist(sy,bins=100,alpha=0.5,label='sy')
                    plt.legend()
                    plt.savefig('./dongfang_sx_sy_hist.png')
                    plt.close()
                    exit()
                #rx, ry, sx, sy = augment_coordinates(rx, ry, sx, sy)
                ##normalize
                traces = raw_data
                patches, rx_p, ry_p, sx_p, sy_p, _ = gen_patches(
                    traces, [rx, ry, sx, sy],
                    (trace_ps, time_ps),
                    (trace_sd, time_sd)
                )
                dataL.extend(patches)
                sxL.extend(sx_p)
                syL.extend(sy_p)
                rxL.extend(rx_p)
                ryL.extend(ry_p)
        else:
            stats =self.compute_coord_stats(label_files,label_cleaned_dir,use_raw=False)
            for recv_line, recv_no, file in tqdm(label_files, desc="Loading test data"):
                label_file = (
                    label_cleaned_dir
                    / f"label_cleaned_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                label_attrs_file = (
                    label_cleaned_dir
                    / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                )
                mask_file = (
                    mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                if not mask_file.exists():
                    continue
                label = np.load(label_file)
                label_attrs = np.load(label_attrs_file)
                mask = np.load(mask_file)
                sx_old = label_attrs["shot_x"]
                sy_old = label_attrs["shot_y"]
                rx_old = label_attrs["rec_x"]
                ry_old = label_attrs["rec_y"]
                shot_line = label_attrs["shot_line"]
                #normalize  
                sx =2*(sx_old-stats["sx_min"])/(stats['sx_max']-stats['sx_min'])-1
                sy =2*(sy_old-stats["sy_min"])/(stats['sy_max']-stats['sy_min'])-1
                rx =2*(rx_old-stats["rx_min"])/(stats['rx_max']-stats['rx_min'])-1
                ry =2*(ry_old-stats["ry_min"])/(stats['ry_max']-stats['ry_min'])-1
                               
                shot_line_unique = np.unique(shot_line)
                idx = self._rng.choice(shot_line_unique, size=4, replace=False)
                idx = np.sort(idx)
                indices = np.where(np.isin(shot_line, idx))[0]

                sx_selected = sx[indices]
                sy_selected = sy[indices]
                rx_arr = rx[indices]
                ry_arr = ry[indices]
                data_selected =(label[indices])
                mask_selected = (mask[indices])
                dataL.append(data_selected)
                sxL.append(sx_selected)
                syL.append(sy_selected)
                rxL.append(rx_arr)
                ryL.append(ry_arr)
                maskL.append(mask_selected)

        self.dataL = dataL
        self.rxL = rxL
        self.ryL = ryL
        self.sxL = sxL
        self.syL = syL
        self.maskL = maskL
        print(f"Total patches: {len(self.dataL)}")
        print(f'Begin Training at {datetime.datetime.now()}')

    def __len__(self):
        return len(self.dataL)
    
    def compute_coord_stats(self,label_files, attrs_dir, use_raw=True):
        sx_all, sy_all, rx_all, ry_all = [], [], [], []
        for recv_line, recv_no, _ in label_files:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            else:
                attrs_fp = attrs_dir / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            if not attrs_fp.exists():
                continue
            attrs = np.load(attrs_fp)
            sx_all.append(attrs["shot_x"]); sy_all.append(attrs["shot_y"])
            rx_all.append(attrs["rec_x"]);   ry_all.append(attrs["rec_y"])

        sx_all = np.concatenate(sx_all); sy_all = np.concatenate(sy_all)
        rx_all = np.concatenate(rx_all); ry_all = np.concatenate(ry_all)

        stats = {
            "sx_min": sx_all.min(), "sx_max": sx_all.max(),
            "sy_min": sy_all.min(), "sy_max": sy_all.max(),
            "rx_min": rx_all.min(), "rx_max": rx_all.max(),
            "ry_min": ry_all.min(), "ry_max": ry_all.max(),
        }
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])
        return stats

    def __getitem__(self, idx):
        data = self.dataL[idx]
        rx = self.rxL[idx]
        ry = self.ryL[idx]
        sx = self.sxL[idx]
        sy = self.syL[idx]

        if self.train:
            missing_ratio = sample_missing_ratio()
            masked, mask = apply_mixed_mask(data, missing_ratio, block_prob=0.5)
            data_n, masked_n, log_tau = robust_scale_pair(data, masked, mask, q=0.995)
            return (
                data_n.astype(np.float32),          
                masked_n.astype(np.float32),               
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                log_tau.astype(np.float32),
                mask.astype(np.float32),
            )
        else:
            data_masked = self.maskL[idx]
            mask =(np.abs(data_masked)> 0).astype(np.float32)
            return (
                data.astype(np.float32),
                data_masked.astype(np.float32),
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                np.mean(rx).astype(np.float32),
                mask.astype(np.float32),
            )
      
##use data normed by shotidx  
def _augment_coords(rx, ry, sx, sy, jitter=0.05, rot_scale=True, center_prob=0.5):
    """H5 那套：旋转+缩放+随机中心化 + 可选 jitter。输入输出都是 numpy 1D。"""
    rx = rx.copy(); ry = ry.copy(); sx = sx.copy(); sy = sy.copy()

    # 轻微 jitter
    rx += np.random.uniform(-jitter, jitter, size=rx.shape)
    ry += np.random.uniform(-jitter, jitter, size=ry.shape)
    sx += np.random.uniform(-jitter, jitter, size=sx.shape)
    sy += np.random.uniform(-jitter, jitter, size=sy.shape)

    if rot_scale:
        # 随机中心化（选 receiver 或 source 的某个点）
        if np.random.rand() < center_prob:
            if np.random.rand() < 0.5:
                dx, dy = np.random.choice(rx), np.random.choice(ry)
            else:
                dx, dy = np.random.choice(sx), np.random.choice(sy)
            rx -= dx; ry -= dy; sx -= dx; sy -= dy

        # 旋转
        theta = np.random.rand() * 2.0 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        rx_, ry_ = rx*c - ry*s, rx*s + ry*c
        sx_, sy_ = sx*c - sy*s, sx*s + sy*c
        rx, ry, sx, sy = rx_, ry_, sx_, sy_

        # 缩放（别太猛）
        scale = np.random.uniform(0.8, 1.2)
        rx *= scale; ry *= scale; sx *= scale; sy *= scale

    # 如果你坐标本来在 [-1,1]，增强后建议轻微 clip
    rx = np.clip(rx, -1.5, 1.5)
    ry = np.clip(ry, -1.5, 1.5)
    sx = np.clip(sx, -1.5, 1.5)
    sy = np.clip(sy, -1.5, 1.5)
    return rx, ry, sx, sy   

      
class RAW_dongfang_1031V2(Dataset):
    def __init__(self, time_ps, trace_ps,sample_num:int=1248,train=True,bin_size:int=50):
        super().__init__()
        print('Loading Dongfang dataset...')
        raw_data_dir = Path("./dongfang/raw")
        label_cleaned_dir = Path("./dongfang/label/cleaned_data")
        #mask_dir = Path("./dongfang/aligned_raw_data")
        mask_dir = Path(f'./dongfang/aligned_raw_data_{bin_size}m')

        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.train = train

        self._rng = np.random.default_rng(123)
        
        self.std_val = None

        self.dt_ms = 4
        self.t0_ms = 0

        pattern_label = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        pattern_mask = re.compile(r"aligned_raw_data_recl_(\d+)_recn_(\d+).npy")
        
        self.space_scale =None

        label_files = []
        mask_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            m = pattern_label.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                label_files.append((recv_line, recv_no, file))

        for file in mask_dir.glob("aligned_raw_data_recl_*_recn_*.npy"):
            m = pattern_mask.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                mask_files.append((recv_line, recv_no, file))

        print(f"Total label files: {len(label_files)}")
        print(f"Total mask files: {len(mask_files)}")
        print('use normalize_clip')

        # -------------------------------------------------
        # 统一全局坐标归一化：用 aligned_raw_attributes 计算 stats
        # -------------------------------------------------
        self.stats = self._compute_coord_stats(mask_files, mask_dir, use_raw=False)

        if train:
            # ================== 训练：整炮 + patch 索引 ==================
            self.gathers_data = []
            self.gathers_rx = []
            self.gathers_ry = []
            self.gathers_sx = []
            self.gathers_sy = []
            self.patch_index = []
            self.gather_info = []

            for (recv_line, recv_no, file) in tqdm(mask_files, desc="Loading training gathers"):
                raw_data_file = raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                raw_attrs_file = raw_data_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                if not raw_data_file.exists() or not raw_attrs_file.exists():
                    print(f"[WARN] File not found, skip: {raw_data_file}, {raw_attrs_file}")
                    continue

                raw_data = np.load(raw_data_file) 
                if sample_num is not None and raw_data.shape[-1]>sample_num:
                    diff = raw_data.shape[-1] - sample_num
                    if diff > 0:
                      raw_data= raw_data[:,diff:]
                raw_attrs = np.load(raw_attrs_file)

                sx_old = raw_attrs["shot_x"]
                sy_old = raw_attrs["shot_y"]
                rx_old = raw_attrs["rec_x"]
                ry_old = raw_attrs["rec_y"]
                shot_line = raw_attrs["shot_line"]
                
                #order = np.lexsort((ry_old, rx_old, sy_old, sx_old))
                #raw_data = raw_data[order]
                #sx_old = sx_old[order]
                #sy_old = sy_old[order]
                #rx_old = rx_old[order]
                #ry_old = ry_old[order]
                #shot_line = shot_line[order]

                # --- per-shot_line 幅度归一化 ---
                '''shot_line_unique = np.unique(shot_line)
                for line_id in shot_line_unique:
                    idx_line = np.where(shot_line == line_id)[0]
                    raw_data[idx_line] = normalize_clip(raw_data[idx_line])'''

                # --- 使用全局 stats 归一化坐标 ---
                stats = self.stats
                sx = 2 * (sx_old - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
                sy = 2 * (sy_old - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
                rx = 2 * (rx_old - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
                ry = 2 * (ry_old - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1

                # --- 保存整炮 ---
                self.gathers_data.append(raw_data.astype(np.float32))
                self.gathers_sx.append(sx.astype(np.float32))
                self.gathers_sy.append(sy.astype(np.float32))
                self.gathers_rx.append(rx.astype(np.float32))
                self.gathers_ry.append(ry.astype(np.float32))


                '''patches, rx_p, ry_p, sx_p, sy_p, t_idx = gen_patches_random(
                    raw_data,
                    [rx, ry, sx, sy],
                    patch_size=(trace_ps, time_ps),
                    #stride=(trace_sd, time_sd),
                )
                # 过滤全空 patch（比如全 0 或能量太低）
                patches_f, coords_f, kept_idx = keep_noblank_patches(
                    patches, [rx_p, ry_p, sx_p, sy_p]
                )
                if patches_f is None:
                    continue
                n_traces, n_samples = raw_data.shape
                if n_traces < trace_ps or n_samples < time_ps:
                    continue
                '''
                n_traces, n_samples = raw_data.shape
                non_overlap_patches = ((n_traces // trace_ps) * (n_samples // time_ps))
                num_patches_per_gather = max(non_overlap_patches * 2, 10)
                self.gather_info.append({
                    'gather_id': len(self.gathers_data) - 1,
                    'n_traces': n_traces,
                    'n_samples': n_samples,
                    'num_patches': num_patches_per_gather
                })

            self.patch_to_gather = []
            for info in self.gather_info:
                for patch_id in range(info['num_patches']):
                    self.patch_to_gather.append((info['gather_id'], patch_id))

            print(f"Total training patches (indexed): {len(self.patch_to_gather)}")
            print(f'Begin Training at {datetime.datetime.now()}')

        else:
            dataL, rxL, ryL, sxL, syL = [], [], [], [], []
            maskL = []
            mask_01L = []
            recL = []
            mode_countL=[]

            for recv_line, recv_no, file in tqdm(mask_files, desc="Loading test data"):
                recL.append((recv_line, recv_no))

                label_file = (
                    label_cleaned_dir
                    / f"label_cleaned_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                label_attrs_file = (
                    mask_dir
                    / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                )
                mask_file = (
                    mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                if not mask_file.exists() or not label_file.exists() or not label_attrs_file.exists():
                    print(f"[WARN] File not found, skip: {mask_file}, {label_file}, {label_attrs_file}")
                    continue

                label = np.load(label_file)
                mask = np.load(mask_file)
                if sample_num is not None and label.shape[-1]>sample_num:
                    diff = label.shape[-1] - sample_num
                    if diff > 0:
                        label = label[:,diff:]

                label_attrs = np.load(label_attrs_file)
            
                sx_old = label_attrs["shot_x"]
                sy_old = label_attrs["shot_y"]
                rx_old = label_attrs["rec_x"]
                ry_old = label_attrs["rec_y"]
                shot_line = label_attrs["shot_line"]
                
                if False:
                    unique_lines, counts = np.unique(shot_line, return_counts=True) 
                    mode_count = mode(counts, keepdims=True).mode[0] 
                    mode_countL.append(mode_count)
                    valid_lines = unique_lines[counts == mode_count] 
                    valid_indices_mask = np.isin(shot_line, valid_lines) 
                    label = label[valid_indices_mask]
                    mask = mask[valid_indices_mask] 
                    sx_old = sx_old[valid_indices_mask] 
                    sy_old = sy_old[valid_indices_mask] 
                    rx_old = rx_old[valid_indices_mask] 
                    ry_old = ry_old[valid_indices_mask] 
                    shot_line = shot_line[valid_indices_mask]

                # 排序
                order = np.lexsort((ry_old, rx_old, sy_old, sx_old))
                sx_old = sx_old[order]
                sy_old = sy_old[order]
                rx_old = rx_old[order]
                ry_old = ry_old[order]
                label = label[order]
                mask = mask[order]
                shot_line = shot_line[order]

                # 使用全局 stats 归一化坐标（跟 train 一致）
                stats = self.stats
                sx = 2 * (sx_old - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
                sy = 2 * (sy_old - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
                rx = 2 * (rx_old - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
                ry = 2 * (ry_old - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1

                # per-shot_line 幅度归一化
                '''shot_line_unique = np.unique(shot_line)
                for line_id in shot_line_unique:
                    idx_line = np.where(shot_line == line_id)[0]
                    label[idx_line] = normalize_clip(label[idx_line])'''
                
                indices = np.arange(label.shape[0])
                sx_selected = sx[indices]
                sy_selected = sy[indices]
                rx_arr = rx[indices]
                ry_arr = ry[indices]
                data_selected = label[indices]
                mask_selected = mask[indices]

                mask_01 = (np.any(mask_selected > 0, axis=1)).astype(np.float32)
                mask_selected = data_selected * mask_01[:, None]
                assert np.allclose(mask_selected[mask_01 == 0], 0.0)

                dataL.append(data_selected.astype(np.float32))
                sxL.append(sx_selected.astype(np.float32))
                syL.append(sy_selected.astype(np.float32))
                rxL.append(rx_arr.astype(np.float32))
                ryL.append(ry_arr.astype(np.float32))
                maskL.append(mask_selected.astype(np.float32))
                mask_01L.append(mask_01.astype(np.float32))

            self.recL = recL
            self.dataL = dataL
            self.rxL = rxL
            self.ryL = ryL
            self.sxL = sxL
            self.syL = syL
            self.maskL = maskL
            self.mask_01L = mask_01L
            self.modecountL = mode_countL

            print(f"Total test gathers: {len(self.dataL)}")
            print(f'Begin Testing at {datetime.datetime.now()}')

    def __len__(self):
        if self.train:
            return len(self.patch_to_gather)
        else:
            return len(self.dataL)

    def typical_grid_step(self,arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u  # 无法估步长
        d = np.diff(u)
        d = d[d > eps]     # 去掉重复和数值噪声
        if d.size == 0:
            return None, u
        return float(np.median(d)), u

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
        return stats
    
    def __getitem__(self, idx):
        if self.train:
            # 动态随机采样
            g_id, _ = self.patch_to_gather[idx]
            data_full = self.gathers_data[g_id]
            rx_full = self.gathers_rx[g_id]
            ry_full = self.gathers_ry[g_id]
            sx_full = self.gathers_sx[g_id]
            sy_full = self.gathers_sy[g_id]
            
            n_traces, n_samples = data_full.shape     
            # ========== 动态随机采样位置 ==========
            max_trace_start = n_traces - self.trace_ps
            # 每次调用都随机采样新的位置
            ts = np.random.randint(0, max(1, max_trace_start + 1))
            # ======================================
            te = ts + self.trace_ps
            
            data_patch = data_full[ts:te, :].astype(np.float32)
            missing_ratio = sample_missing_ratio()
            masked_patch, mask_patch = apply_mixed_mask(data_patch, missing_ratio, block_prob=0.0)
            
            # ========== 归一化：使用 masked_patch 的标准差（与推理一致）==========
            obs = masked_patch[mask_patch > 0]  # 只用观测点（mask=1表示观测点，mask=0表示缺失点）
            obs = obs[np.isfinite(obs)]
            std_val  = np.float32(np.std(obs))
            std_val  = np.float32(max(std_val, 1e-2))
            self.std_val = std_val
            thres = np.percentile(np.abs(masked_patch), 99.5)
            if thres == 0:
                thres = 1e-6
            masked_patch =np.clip(masked_patch, -thres, thres)
            masked_patch = masked_patch / thres
            data_patch = np.clip(data_patch, -thres, thres)
            data_patch = data_patch / thres
            # ====================================================================
            
            rx_patch = rx_full[ts:te]
            ry_patch = ry_full[ts:te]
            sx_patch = sx_full[ts:te]
            sy_patch = sy_full[ts:te]
            #rx_patch, ry_patch, sx_patch, sy_patch = _augment_coords(
            #    rx_patch, ry_patch, sx_patch, sy_patch,
            #    jitter=0.05, rot_scale=True
            #)
            
            # 确保坐标也是 float32
            rx_patch = rx_patch.astype(np.float32)
            ry_patch = ry_patch.astype(np.float32)
            sx_patch = sx_patch.astype(np.float32)
            sy_patch = sy_patch.astype(np.float32)
            
            # 时间轴
            time_idx_1d = np.arange(0, self.time_ps, dtype=np.int32)
            time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
            time_axis_2d = np.tile(time_axis_1d[None, :], (self.trace_ps, 1))
            
            return (
                (data_patch).astype(np.float32),
                (masked_patch).astype(np.float32),
                rx_patch,
                ry_patch,
                sx_patch,
                sy_patch,
                time_axis_2d.astype(np.float32),
                self.std_val,  # 返回归一化因子（std），用于推理时反归一化
            )
        else:
            data = self.dataL[idx]
            rx = self.rxL[idx]
            ry = self.ryL[idx]
            sx = self.sxL[idx]
            sy = self.syL[idx]
            data_masked = self.maskL[idx]
            mask_01 = self.mask_01L[idx]
            recv_line, recv_no = self.recL[idx]
            if len(self.modecountL) != 0:
                mode_count = self.modecountL[idx]
                print(f"Mode count: {mode_count}")
            return (
                data.astype(np.float32),
                data_masked.astype(np.float32),
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                recv_line,
                recv_no,
            )

class RAW_dongfang_1031V2_LineGradeTest(Dataset):
    """
    测线级测试数据集：每个样本是一个 (recl, recn, shot_line)。

    统计来源：`stat_missing_gaps.py` 输出的 per_line_*.csv（包含 grade=A/B/C/D）。
    输出/处理形式与 RAW_dongfang_1031V2(train=False) 对齐：
      (data, data_mask, rx, ry, sx, sy, recv_line, recv_no)

    说明：
    - 本数据集不会把 grade/shot_line 放进返回 tuple（避免改动下游签名）；
      但会在 `self.metaL[idx]` 中保存元信息，供采样脚本用于命名与统计。
    """

    def __init__(
        self,
        time_ps: int,
        trace_ps: int,
        sample_num: int = 1248,
        report_path: str = "./mask_reports",
        grades: str = "B,C,D",
        mask_subdir: str = "./dongfang/aligned_raw_data_50m",
        label_subdir: str = "./dongfang/label/cleaned_data",
    ):
        super().__init__()
        print("Loading Dongfang LINE-GRADE test dataset...")

        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.sample_num = sample_num
        self.train = False
        self.dt_ms = 4
        self.t0_ms = 0

        self.mask_dir = Path(mask_subdir)
        self.label_cleaned_dir = Path(label_subdir)

        # 计算全局坐标 stats（与 RAW_dongfang_1031V2 一致）
        pattern_mask = re.compile(r"aligned_raw_data_recl_(\d+)_recn_(\d+).npy")
        mask_files_all = []
        for file in self.mask_dir.glob("aligned_raw_data_recl_*_recn_*.npy"):
            m = pattern_mask.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                mask_files_all.append((recv_line, recv_no, file))
        self.stats = self.compute_coord_stats(mask_files_all, self.mask_dir, use_raw=False)

        allowed_grades = self._parse_grades(grades)
        per_line_csv = self._resolve_per_line_csv(Path(report_path))
        self.metaL = self._load_line_meta(per_line_csv, allowed_grades=allowed_grades)
        self.metaL.sort(key=lambda d: (d["recl"], d["recn"], d["shot_line"]))

        print(f"Line-grade meta loaded from: {per_line_csv}")
        print(f"Selected grades: {','.join(sorted(allowed_grades))}")
        print(f"Total LINE-GRADE test samples: {len(self.metaL)}")
        print(f"Begin Testing at {datetime.datetime.now()}")

        # receiver cache（减少同一 receiver 多测线重复 IO）
        self._cache_key = None
        self._cache_label = None
        self._cache_mask = None
        self._cache_shot_line = None
        self._cache_sx_old = None
        self._cache_sy_old = None
        self._cache_rx_old = None
        self._cache_ry_old = None
        # 缓存每个 receiver 的所有 shot_line 列表（用于找相邻测线）
        self._cache_shot_lines_unique = None

    @staticmethod
    def _parse_grades(grades) -> set:
        if grades is None:
            return {"B", "C", "D"}
        if isinstance(grades, str):
            s = grades.strip().upper()
            parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) == 1 and len(parts[0]) > 1 and all(c in "ABCD" for c in parts[0]):
                parts = list(parts[0])
            return set(parts)
        return set([str(g).strip().upper() for g in grades])

    @staticmethod
    def _resolve_per_line_csv(report_path: Path) -> Path:
        if report_path.is_file():
            return report_path
        if not report_path.exists():
            raise FileNotFoundError(f"report_path 不存在: {report_path}")
        candidates = sorted(report_path.glob("per_line_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"在目录中未找到 per_line_*.csv: {report_path}")
        return candidates[0]

    @staticmethod
    def _grade_from_row(row: dict) -> str:
        # 优先用 csv 中已写好的 grade
        g = (row.get("grade", "") or "").strip().upper()
        if g in {"A", "B", "C", "D"}:
            return g

        # 兼容旧表：没有 grade 列时，从数值字段推导
        n_traces = int(float(row.get("n_traces", "0") or 0))
        missing_traces = int(float(row.get("missing_traces", "0") or 0))
        missing_ratio = float(row.get("missing_ratio", "0") or 0.0)
        max_run_ratio = float(row.get("max_consecutive_missing_ratio", "0") or 0.0)
        gap_thr = float(row.get("gap_ratio_threshold", row.get("threshold", "0.2")) or 0.2)
        miss_thr = float(row.get("missing_ratio_threshold", "0.6") or 0.6)

        if n_traces <= 0:
            return "A"
        if missing_traces >= n_traces:
            return "D"
        if missing_ratio >= miss_thr:
            return "C"
        if max_run_ratio >= gap_thr:
            return "B"
        return "A"

    def _load_line_meta(self, per_line_csv: Path, allowed_grades: set) -> List[dict]:
        metaL: List[dict] = []
        receiver_exists_cache = {}

        with per_line_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if "recl" not in row or "recn" not in row or "shot_line" not in row:
                    continue
                recl = int(float(row["recl"]))
                recn = int(float(row["recn"]))
                shot_line = int(float(row["shot_line"]))
                grade = self._grade_from_row(row)
                if grade not in allowed_grades:
                    continue

                key = (recl, recn)
                ok = receiver_exists_cache.get(key)
                if ok is None:
                    label_file = self.label_cleaned_dir / f"label_cleaned_data_recl_{recl}_recn_{recn}.npy"
                    attrs_file = self.mask_dir / f"aligned_raw_attributes_recl_{recl}_recn_{recn}.npy"
                    mask_file = self.mask_dir / f"aligned_raw_data_recl_{recl}_recn_{recn}.npy"
                    ok = bool(label_file.exists() and attrs_file.exists() and mask_file.exists())
                    receiver_exists_cache[key] = ok
                if not ok:
                    continue

                metaL.append(
                    dict(
                        recl=recl,
                        recn=recn,
                        shot_line=shot_line,
                        grade=grade,
                        n_traces=int(float(row.get("n_traces", "0") or 0)),
                        missing_traces=int(float(row.get("missing_traces", "0") or 0)),
                        missing_ratio=float(row.get("missing_ratio", "0") or 0.0),
                        max_consecutive_missing_ratio=float(row.get("max_consecutive_missing_ratio", "0") or 0.0),
                    )
                )
        return metaL

    def __len__(self):
        return len(self.metaL)

    def _load_receiver_cache(self, recl: int, recn: int) -> None:
        label_file = self.label_cleaned_dir / f"label_cleaned_data_recl_{recl}_recn_{recn}.npy"
        attrs_file = self.mask_dir / f"aligned_raw_attributes_recl_{recl}_recn_{recn}.npy"
        mask_file = self.mask_dir / f"aligned_raw_data_recl_{recl}_recn_{recn}.npy"

        label = np.load(label_file)
        mask = np.load(mask_file)
        if self.sample_num is not None and label.shape[-1] > self.sample_num:
            diff = label.shape[-1] - self.sample_num
            if diff > 0:
                label = label[:, diff:]

        attrs = np.load(attrs_file)
        sx_old = attrs["shot_x"]
        sy_old = attrs["shot_y"]
        rx_old = attrs["rec_x"]
        ry_old = attrs["rec_y"]
        shot_line = attrs["shot_line"]

        # 与 RAW_dongfang_1031V2 test 一致的排序
        order = np.lexsort((ry_old, rx_old, sy_old, sx_old))
        self._cache_label = label[order]
        self._cache_mask = mask[order]
        self._cache_sx_old = sx_old[order]
        self._cache_sy_old = sy_old[order]
        self._cache_rx_old = rx_old[order]
        self._cache_ry_old = ry_old[order]
        self._cache_shot_line = shot_line[order]
        # 缓存该 receiver 的所有唯一 shot_line（排序后），用于找相邻测线
        self._cache_shot_lines_unique = np.unique(self._cache_shot_line)
        self._cache_key = (recl, recn)

    def _get_neighbor_lines(self, target_sl: int) -> List[int]:
        """
        获取目标测线的相邻两条测线（同一 receiver 下）。
        返回：[prev_line, target_sl, next_line]（如果存在）
        """
        if self._cache_shot_lines_unique is None or len(self._cache_shot_lines_unique) == 0:
            return [target_sl]
        
        sorted_lines = np.sort(self._cache_shot_lines_unique)
        idx_target = np.searchsorted(sorted_lines, target_sl)
        
        if idx_target < 0 or idx_target >= len(sorted_lines) or sorted_lines[idx_target] != target_sl:
            # 目标测线不在列表中（理论上不应该发生）
            return [target_sl]
        
        neighbor_lines = []
        # 前一条
        if idx_target > 0:
            neighbor_lines.append(int(sorted_lines[idx_target - 1]))
        # 目标测线
        neighbor_lines.append(target_sl)
        # 后一条
        if idx_target < len(sorted_lines) - 1:
            neighbor_lines.append(int(sorted_lines[idx_target + 1]))
        
        return neighbor_lines

    def __getitem__(self, idx):
        meta = self.metaL[idx]
        recl = meta["recl"]
        recn = meta["recn"]
        sl = meta["shot_line"]

        if self._cache_key != (recl, recn):
            self._load_receiver_cache(recl, recn)

        # 获取相邻测线（包括目标测线）
        neighbor_lines = self._get_neighbor_lines(sl)
        
        # 合并所有相邻测线的数据
        data_parts = []
        mask_parts = []
        sx_parts = []
        sy_parts = []
        rx_parts = []
        ry_parts = []
        
        for neighbor_sl in neighbor_lines:
            idx_line = np.where(self._cache_shot_line == neighbor_sl)[0]
            if idx_line.size == 0:
                continue
            
            data_parts.append(self._cache_label[idx_line])
            mask_parts.append(self._cache_mask[idx_line])
            sx_parts.append(self._cache_sx_old[idx_line])
            sy_parts.append(self._cache_sy_old[idx_line])
            rx_parts.append(self._cache_rx_old[idx_line])
            ry_parts.append(self._cache_ry_old[idx_line])
        
        if len(data_parts) == 0:
            raise IndexError(f"receiver ({recl},{recn}) 内不存在 shot_line={sl} 及其相邻测线")
        
        # 按测线顺序合并（保持空间连续性）
        data_selected = np.concatenate(data_parts, axis=0)
        mask_selected = np.concatenate(mask_parts, axis=0)
        sx_old = np.concatenate(sx_parts, axis=0)
        sy_old = np.concatenate(sy_parts, axis=0)
        rx_old = np.concatenate(rx_parts, axis=0)
        ry_old = np.concatenate(ry_parts, axis=0)

        stats = self.stats
        sx = 2 * (sx_old - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
        sy = 2 * (sy_old - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
        rx = 2 * (rx_old - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
        ry = 2 * (ry_old - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1

        mask_01 = (np.any(mask_selected > 0, axis=1)).astype(np.float32)
        data_masked = data_selected * mask_01[:, None]
        assert np.allclose(data_masked[mask_01 == 0], 0.0)

        return (
            data_selected.astype(np.float32),
            data_masked.astype(np.float32),
            rx.astype(np.float32),
            ry.astype(np.float32),
            sx.astype(np.float32),
            sy.astype(np.float32),
            recl,
            recn,
        )

    def compute_coord_stats(self, file_list, attrs_dir, use_raw=True):
        # 复制 RAW_dongfang_1031V2 的实现，保持归一化统计一致
        sx_all, sy_all, rx_all, ry_all = [], [], [], []
        for recv_line, recv_no, _ in file_list:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            else:
                attrs_fp = attrs_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            if not attrs_fp.exists():
                continue
            attrs = np.load(attrs_fp)
            sx_all.append(attrs["shot_x"])
            sy_all.append(attrs["shot_y"])
            rx_all.append(attrs["rec_x"])
            ry_all.append(attrs["rec_y"])

        sx_all = np.concatenate(sx_all)
        sy_all = np.concatenate(sy_all)
        rx_all = np.concatenate(rx_all)
        ry_all = np.concatenate(ry_all)

        stats = {
            "sx_min": np.unique(sx_all).min(),
            "sx_max": np.unique(sx_all).max(),
            "sy_min": np.unique(sy_all).min(),
            "sy_max": np.unique(sy_all).max(),
            "rx_min": np.unique(rx_all).min(),
            "rx_max": np.unique(rx_all).max(),
            "ry_min": np.unique(ry_all).min(),
            "ry_max": np.unique(ry_all).max(),
        }
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])
        return stats

class RAW_dongfang_cmp(Dataset):
    def __init__(
        self,
        time_ps,
        trace_ps,
        trace_sd,
        time_sd,
        sample_num: int = 1248,
        train=True,
        filter_by_offset: bool = False,
        filter_by_cmp_no: bool = False,
        filter_by_azimuth: bool = False,
        offset_choice_ratio: float= 0.3,
        azimuth_bucket_idx: int  =9,
        azimuth_bucket_split: int = 8,
        test_all_azimuth_buckets: bool = False,
    ):
        super().__init__()
        print('Loading Dongfang dataset...')
        raw_data_dir = Path("./dongfang/cmp_raw")
        label_cleaned_dir = Path("./dongfang/cmp_label/")
        mask_dir = Path('./dongfang/cmp_aligned_raw_data')

        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.time_sd = time_sd
        self.trace_sd = trace_sd
        self.train = train
        self.filter_by_offset = filter_by_offset
        self.filter_by_cmp_no = filter_by_cmp_no
        self.filter_by_azimuth = filter_by_azimuth
        self.offset_choice_ratio = offset_choice_ratio
        self.azimuth_bucket_idx = azimuth_bucket_idx
        self.azimuth_bucket_split = azimuth_bucket_split
        self.test_all_azimuth_buckets = test_all_azimuth_buckets

        self._rng = np.random.default_rng(123)
        self.dt_ms = 4
        self.t0_ms = 0

        pattern_label = re.compile(r"label_data_cmpline_(\d+).npy")
        pattern_mask = re.compile(r"aligned_raw_data_cmpline_(\d+).npy")
        pattern_raw = re.compile(r"raw_data_cmpline_(\d+).npy")

        label_files = []
        mask_files = []
        raw_files = []
        for file in label_cleaned_dir.glob("label_data_cmpline_*.npy"):
            m = pattern_label.match(file.name)
            if m:
                cmp_line = int(m.group(1))
                label_files.append(cmp_line)

        for file in mask_dir.glob("aligned_raw_data_cmpline_*.npy"):
            m = pattern_mask.match(file.name)
            if m:
                cmp_line = int(m.group(1))
                mask_files.append(cmp_line)
        
        for file in raw_data_dir.glob("raw_data_cmpline_*.npy"):
            m = pattern_raw.match(file.name)
            if m:
                cmp_line = int(m.group(1))
                raw_files.append(cmp_line)

        print(f"Total label files: {len(label_files)}")
        print(f"Total mask files: {len(mask_files)}")
        print('use normalize_clip')
        self.stats = self.compute_coord_stats(mask_files, mask_dir, use_raw=False)

        if train:
            self.gathers_data = []
            self.gathers_cmp_x = []
            self.gathers_cmp_y = []
            self.gathers_offset = []
            self.gathers_azimuth = []
            self.patch_index = []

            for cmp_line in tqdm(raw_files, desc="Loading training gathers"):
                raw_data = np.load(raw_data_dir / f"raw_data_cmpline_{cmp_line}.npy")
                raw_attrs = np.load(raw_data_dir / f"raw_attributes_cmpline_{cmp_line}.npy")

                sx_old = raw_attrs["shot_x"]
                sy_old = raw_attrs["shot_y"]
                rx_old = raw_attrs["rec_x"]
                ry_old = raw_attrs["rec_y"]
                offset_old = raw_attrs["offset"]
                azimuth_old = raw_attrs["azimuth"]
                cmp_line = raw_attrs["cmp_line"]
                shot_line = raw_attrs["shot_line"]
                cmp_no = raw_attrs["cmp"]
                
                cmp_line_unique = np.unique(cmp_line)
                for line_id in cmp_line_unique:
                    idx_line = np.where(cmp_line == line_id)[0]
                    raw_data[idx_line] = normalize_clip(raw_data[idx_line])

                # 计算 CMP 坐标
                cmp_x_old = (sx_old + rx_old) / 2.0
                cmp_y_old = (sy_old + ry_old) / 2.0

                stats = self.stats
                cmp_x = 2 * (cmp_x_old - stats["cmp_x_min"]) / (stats["cmp_x_max"] - stats["cmp_x_min"]) - 1
                cmp_y = 2 * (cmp_y_old - stats["cmp_y_min"]) / (stats["cmp_y_max"] - stats["cmp_y_min"]) - 1
                offset = 2 * (offset_old - stats["offset_min"]) / (stats["offset_max"] - stats["offset_min"]) - 1
                azimuth = 2 * (azimuth_old - stats["azimuth_min"]) / (stats["azimuth_max"] - stats["azimuth_min"]) - 1

                self.gathers_data.append(raw_data.astype(np.float32))
                self.gathers_cmp_x.append(cmp_x.astype(np.float32))
                self.gathers_cmp_y.append(cmp_y.astype(np.float32))
                self.gathers_offset.append(offset.astype(np.float32))
                self.gathers_azimuth.append(azimuth.astype(np.float32))

                g_id = len(self.gathers_data) - 1

                trace_ps, time_ps = self.trace_ps, self.time_ps
                trace_sd, time_sd = self.trace_sd, self.time_sd

                n_traces, n_samples = raw_data.shape
                if n_traces < trace_ps or n_samples < time_ps:
                    continue

                patches, cmp_x_p, cmp_y_p, offset_p, azimuth_p, t_idx = gen_patches(
                    raw_data,
                    [cmp_x, cmp_y, offset, azimuth],
                    patch_size=(trace_ps, time_ps),
                    stride=(trace_sd, time_sd),
                )

                patches_f, coords_f, kept_idx = keep_noblank_patches(
                    patches, [cmp_x_p, cmp_y_p, offset_p, azimuth_p]
                )
                if patches_f is None:
                    continue

                n_j = (n_samples - time_ps) // time_sd + 1

                for k in kept_idx:
                    trace_step = k // n_j
                    time_step = k % n_j
                    i_start = trace_step * trace_sd
                    j_start = time_step * time_sd
                    self.patch_index.append((g_id, i_start, j_start))

            print(f"Total training patches (indexed): {len(self.patch_index)}")
            print(f'Begin Training at {datetime.datetime.now()}')

        else:
            dataL, cmp_xL, cmp_yL, offsetL, azimuthL = [], [], [], [], []
            maskL = []
            mask_01L = []
            cmpL = []
            mode_countL=[]
            orderL = []
            bucket_idxL = []

            for cmp_line in tqdm(mask_files, desc="Loading test data"):
                cmpL.append(cmp_line)
                label_file = (
                    label_cleaned_dir
                    / f"label_data_cmpline_{cmp_line}.npy"
                )
                label_attrs_file = (
                    label_cleaned_dir
                    / f"label_attributes_cmpline_{cmp_line}.npy"
                )
                mask_file = (
                    mask_dir
                    / f"aligned_raw_data_cmpline_{cmp_line}.npy"
                )
                label = np.load(label_file)
                cmp_attrs = np.load(label_attrs_file)
                mask = np.load(mask_file)
                if sample_num is not None and label.shape[-1]>sample_num:
                    diff = label.shape[-1] - sample_num
                    if diff > 0:
                        label = label[:,diff:]
                        
                sx_old = cmp_attrs["shot_x"]
                sy_old = cmp_attrs["shot_y"]
                rx_old = cmp_attrs["rec_x"]
                ry_old = cmp_attrs["rec_y"]
                offset_old = cmp_attrs["offset"]
                azimuth_old = cmp_attrs["azimuth"]
                cmp_line = cmp_attrs["cmp_line"]
                cmp_no = cmp_attrs["cmp"]
                shot_line = cmp_attrs["shot_line"]                
                if False:
                    unique_lines, counts = np.unique(shot_line, return_counts=True) 
                    mode_count = mode(counts, keepdims=True).mode[0] 
                    mode_countL.append(mode_count)
                    valid_lines = unique_lines[counts == mode_count] 
                    valid_indices_mask = np.isin(shot_line, valid_lines) 
                    label = label[valid_indices_mask]
                    mask = mask[valid_indices_mask] 
                    sx_old = sx_old[valid_indices_mask] 
                    sy_old = sy_old[valid_indices_mask] 
                    rx_old = rx_old[valid_indices_mask] 
                    ry_old = ry_old[valid_indices_mask] 
                    offset_old = offset_old[valid_indices_mask]
                    azimuth_old = azimuth_old[valid_indices_mask]
                    shot_line = shot_line[valid_indices_mask]
                order = np.lexsort((azimuth_old, offset_old))
                sx_old = sx_old[order]
                sy_old = sy_old[order]
                rx_old = rx_old[order]
                ry_old = ry_old[order]
                label = label[order]
                mask = mask[order]
                shot_line = shot_line[order]
                offset_old = offset_old[order]
                azimuth_old = azimuth_old[order]
                cmp_no = cmp_no[:]
                
                # 计算 CMP 坐标
                cmp_x_old = (sx_old + rx_old) / 2.0
                cmp_y_old = (sy_old + ry_old) / 2.0
               
                stats = self.stats
                cmp_x = 2 * (cmp_x_old - stats["cmp_x_min"]) / (stats["cmp_x_max"] - stats["cmp_x_min"]) - 1
                cmp_y = 2 * (cmp_y_old - stats["cmp_y_min"]) / (stats["cmp_y_max"] - stats["cmp_y_min"]) - 1
                offset = 2 * (offset_old - stats["offset_min"]) / (stats["offset_max"] - stats["offset_min"]) - 1
                azimuth = 2 * (azimuth_old - stats["azimuth_min"]) / (stats["azimuth_max"] - stats["azimuth_min"]) - 1

                cmp_line_unique = np.unique(cmp_line)
                for line_id in cmp_line_unique:
                    idx_line = np.where(cmp_line == line_id)[0]
                    label[idx_line] = normalize_clip(label[idx_line])
                if self.test_all_azimuth_buckets:
                    # 测试模式：同一条测线上遍历所有方位角桶
                    az_min, az_max = azimuth_old.min(), azimuth_old.max()
                    bucket_total = max(1, self.azimuth_bucket_split)
                    edges = np.linspace(az_min, az_max, bucket_total + 1)
                    for b in range(bucket_total):
                        indices = np.arange(label.shape[0])
                        lower, upper = edges[b], edges[b + 1]
                        in_bucket = (azimuth_old >= lower) & (
                            azimuth_old <= upper if b == bucket_total - 1 else azimuth_old < upper
                        )
                        bucket_idx = np.where(in_bucket)[0]
                        indices = bucket_idx
                        if len(indices) == 0:
                            continue

                        if self.offset_choice_ratio is not None and len(indices) > 0:
                            offset_sub = offset_old[indices]
                            offset_min, offset_max = offset_sub.min(), offset_sub.max()
                            offset_target = offset_min + self.offset_choice_ratio * (offset_max - offset_min)
                            nearest_offset = offset_sub[np.argmin(np.abs(offset_sub - offset_target))]
                            indices = indices[np.where(np.isclose(offset_sub, nearest_offset))[0]]
                            print(
                                f"[bucket {b}/{bucket_total-1}] Offset range "
                                f"[{offset_min:.3f},{offset_max:.3f}] target={offset_target:.3f} "
                                f"picked={nearest_offset:.3f}, traces={len(indices)}"
                            )
                        elif self.filter_by_offset and len(indices) > 0:
                            offset_sub = offset_old[indices]
                            unique_offsets, offset_counts = np.unique(offset_sub, return_counts=True)
                            most_common_offset = unique_offsets[np.argmax(offset_counts)]
                            indices = indices[np.where(offset_sub == most_common_offset)[0]]
                            print(
                                f"[bucket {b}/{bucket_total-1}] "
                                f"Filtering by offset: selected offset={most_common_offset}, {len(indices)} traces"
                            )
                        if self.filter_by_cmp_no and len(indices) > 0:
                            cmp_sub = cmp_no[indices]
                            unique_cmp_nos, cmp_no_counts = np.unique(cmp_sub, return_counts=True)
                            most_common_cmp_no = unique_cmp_nos[np.argmax(cmp_no_counts)]
                            indices = indices[np.where(cmp_sub == most_common_cmp_no)[0]]
                            print(
                                f"[bucket {b}/{bucket_total-1}] "
                                f"Filtering by cmp_no: selected cmp_no={most_common_cmp_no}, {len(indices)} traces"
                            )

                        if len(indices) == 0:
                            continue

                        cmp_x_selected = cmp_x[indices]
                        cmp_y_selected = cmp_y[indices]
                        offset_selected = offset[indices]
                        azimuth_selected = azimuth[indices]
                        data_selected = label[indices]
                        mask_selected = mask[indices]
                        offset_old_selected = offset_old[indices]
                        azimuth_old_selected = azimuth_old[indices]
                        order_offset = np.lexsort((azimuth_old_selected, offset_old_selected))

                        mask_01 = (np.any(mask_selected > 0, axis=1)).astype(np.float32)
                        mask_selected = data_selected * mask_01[:, None]
                        assert np.allclose(mask_selected[mask_01 == 0], 0.0)

                        cmpL.append(cmp_line)
                        dataL.append(data_selected.astype(np.float32))
                        cmp_xL.append(cmp_x_selected.astype(np.float32))
                        cmp_yL.append(cmp_y_selected.astype(np.float32))
                        offsetL.append(offset_selected.astype(np.float32))
                        azimuthL.append(azimuth_selected.astype(np.float32))
                        maskL.append(mask_selected.astype(np.float32))
                        mask_01L.append(mask_01.astype(np.float32))
                        orderL.append(order_offset)
                        bucket_idxL.append(b)
                else:
                    indices = np.arange(label.shape[0])
                    if self.azimuth_bucket_idx is not None:
                        az_min, az_max = azimuth_old.min(), azimuth_old.max()
                        bucket_total = max(1, self.azimuth_bucket_split)
                        bucket = int(np.clip(self.azimuth_bucket_idx, 0, bucket_total - 1))
                        edges = np.linspace(az_min, az_max, bucket_total + 1)
                        lower, upper = edges[bucket], edges[bucket + 1]
                        in_bucket = (azimuth_old >= lower) & (
                            azimuth_old <= upper if bucket == bucket_total - 1 else azimuth_old < upper
                        )
                        bucket_idx = np.where(in_bucket)[0]
                        indices = bucket_idx
                        az_min_deg = np.rad2deg(az_min) % 360
                        az_max_deg = np.rad2deg(az_max) % 360
                        lower_deg = np.rad2deg(lower) % 360
                        upper_deg = np.rad2deg(upper) % 360
                        print(
                            f"Azimuth range [{az_min_deg:.2f}°, {az_max_deg:.2f}°] "
                            f"bucket {bucket}/{bucket_total-1} "
                            f"[{lower_deg:.2f}°, {upper_deg:.2f}°] "
                            f"traces={len(bucket_idx)}, kept={len(indices)}"
                        )
                    elif self.filter_by_azimuth:
                        unique_azimuths, azimuth_counts = np.unique(azimuth_old, return_counts=True)
                        most_common_azimuth = unique_azimuths[np.argmax(azimuth_counts)]
                        indices = np.where(azimuth_old == most_common_azimuth)[0]
                        print(
                            f"Filtering by azimuth: selected azimuth={most_common_azimuth}, "
                            f"{len(indices)} traces"
                        )
                    if self.offset_choice_ratio is not None and len(indices) > 0:
                        offset_sub = offset_old[indices]
                        offset_min, offset_max = offset_sub.min(), offset_sub.max()
                        offset_target = offset_min + self.offset_choice_ratio * (offset_max - offset_min)
                        nearest_offset = offset_sub[np.argmin(np.abs(offset_sub - offset_target))]
                        indices = indices[np.where(np.isclose(offset_sub, nearest_offset))[0]]
                        print(
                            f"Offset range [{offset_min:.3f},{offset_max:.3f}] target={offset_target:.3f} "
                            f"picked={nearest_offset:.3f}, traces={len(indices)}"
                        )
                    elif self.filter_by_offset and len(indices) > 0:
                        offset_sub = offset_old[indices]
                        unique_offsets, offset_counts = np.unique(offset_sub, return_counts=True)
                        most_common_offset = unique_offsets[np.argmax(offset_counts)]
                        indices = indices[np.where(offset_sub == most_common_offset)[0]]
                        print(
                            f"Filtering by offset: selected offset={most_common_offset}, "
                            f"{len(indices)} traces"
                        )
                    if self.filter_by_cmp_no and len(indices) > 0:
                        cmp_sub = cmp_no[indices]
                        unique_cmp_nos, cmp_no_counts = np.unique(cmp_sub, return_counts=True)
                        most_common_cmp_no = unique_cmp_nos[np.argmax(cmp_no_counts)]
                        indices = indices[np.where(cmp_sub == most_common_cmp_no)[0]]
                        print(
                            f"Filtering by cmp_no: selected cmp_no={most_common_cmp_no}, "
                            f"{len(indices)} traces"
                        )

                    if len(indices) == 0:
                        continue

                    cmp_x_selected = cmp_x[indices]
                    cmp_y_selected = cmp_y[indices]
                    offset_selected = offset[indices]
                    azimuth_selected = azimuth[indices]
                    data_selected = label[indices]
                    mask_selected = mask[indices]
                    offset_old_selected = offset_old[indices]
                    azimuth_old_selected = azimuth_old[indices]
                    order_offset = np.lexsort((azimuth_old_selected, offset_old_selected))

                    mask_01 = (np.any(mask_selected > 0, axis=1)).astype(np.float32)
                    mask_selected = data_selected * mask_01[:, None]
                    assert np.allclose(mask_selected[mask_01 == 0], 0.0)

                    cmpL.append(cmp_line)
                    dataL.append(data_selected.astype(np.float32))
                    cmp_xL.append(cmp_x_selected.astype(np.float32))
                    cmp_yL.append(cmp_y_selected.astype(np.float32))
                    offsetL.append(offset_selected.astype(np.float32))
                    azimuthL.append(azimuth_selected.astype(np.float32))
                    maskL.append(mask_selected.astype(np.float32))
                    mask_01L.append(mask_01.astype(np.float32))
                    orderL.append(order_offset)

            self.cmpL = cmpL
            self.dataL = dataL
            self.cmp_xL = cmp_xL
            self.cmp_yL = cmp_yL
            self.offsetL = offsetL
            self.azimuthL = azimuthL
            self.maskL = maskL
            self.mask_01L = mask_01L
            self.modecountL = mode_countL
            self.orderL= orderL
            self.bucket_idxL = bucket_idxL

            print(f"Total test gathers: {len(self.dataL)}")
            print(f'Begin Testing at {datetime.datetime.now()}')

    def __len__(self):
        if self.train:
            return len(self.patch_index)
        else:
            return len(self.dataL)

    def compute_coord_stats(self, file_list, attrs_dir, use_raw=True):
        cmp_x_all, cmp_y_all, offset_all, azimuth_all = [], [], [], []
        for cmp_line in file_list:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_cmpline_{cmp_line}.npy"
            else:
                attrs_fp = attrs_dir / f"aligned_raw_attributes_cmpline_{cmp_line}.npy"
            if not attrs_fp.exists():
                print(f"[WARN] Attributes file not found: {attrs_fp}")
                continue
            attrs = np.load(attrs_fp)
            sx = attrs["shot_x"]
            sy = attrs["shot_y"]
            rx = attrs["rec_x"]
            ry = attrs["rec_y"]
            offset = attrs["offset"]
            azimuth = attrs["azimuth"]
            
            # 计算 CMP 坐标
            cmp_x = (sx + rx) / 2.0
            cmp_y = (sy + ry) / 2.0
            
            cmp_x_all.append(cmp_x)
            cmp_y_all.append(cmp_y)
            offset_all.append(offset)
            azimuth_all.append(azimuth)

        cmp_x_all = np.concatenate(cmp_x_all)
        cmp_y_all = np.concatenate(cmp_y_all)
        offset_all = np.concatenate(offset_all)
        azimuth_all = np.concatenate(azimuth_all)

        stats = {
            "cmp_x_min": cmp_x_all.min(), "cmp_x_max": cmp_x_all.max(),
            "cmp_y_min": cmp_y_all.min(), "cmp_y_max": cmp_y_all.max(),
            "offset_min": offset_all.min(), "offset_max": offset_all.max(),
            "azimuth_min": azimuth_all.min(), "azimuth_max": azimuth_all.max(),
        }
        stats["Lx"] = 0.5 * (stats["cmp_x_max"] - stats["cmp_x_min"])
        stats["Ly"] = 0.5 * (stats["cmp_y_max"] - stats["cmp_y_min"])
        return stats

    def __getitem__(self, idx):
        if self.train:
            g_id, i_start, j_start = self.patch_index[idx]

            data_full = self.gathers_data[g_id]
            cmp_x_full = self.gathers_cmp_x[g_id]
            cmp_y_full = self.gathers_cmp_y[g_id]
            offset_full = self.gathers_offset[g_id]
            azimuth_full = self.gathers_azimuth[g_id]

            n_traces, n_samples = data_full.shape
            assert 0 <= i_start < n_traces
            assert 0 <= j_start < n_samples

            missing_ratio = sample_missing_ratio()
            masked_full, mask_full = apply_mixed_mask(
                data_full,
                missing_ratio,
                block_prob=0.0
            )

            ts = i_start
            tt = j_start
            te = ts + self.trace_ps
            te_t = tt + self.time_ps

            data_patch = data_full[ts:te, tt:te_t]
            masked_patch = masked_full[ts:te, tt:te_t]
            # mask_patch = mask_full[ts:te, tt:te_t]  # 如有需要可以一并返回

            cmp_x_patch = cmp_x_full[ts:te]
            cmp_y_patch = cmp_y_full[ts:te]
            offset_patch = offset_full[ts:te]
            azimuth_patch = azimuth_full[ts:te]

            # 时间轴
            time_idx_1d = np.arange(tt, tt + self.time_ps, dtype=np.int32)
            time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
            time_axis_2d = np.tile(time_axis_1d[None, :], (self.trace_ps, 1))

            return (
                data_patch.astype(np.float32),
                masked_patch.astype(np.float32),
                cmp_x_patch.astype(np.float32),
                cmp_y_patch.astype(np.float32),
                offset_patch.astype(np.float32),
                azimuth_patch.astype(np.float32),
                time_axis_2d.astype(np.float32),
                cmp_x_patch.astype(np.float32),
            )
        else:
            data = self.dataL[idx]
            cmp_x = self.cmp_xL[idx]
            cmp_y = self.cmp_yL[idx]
            offset = self.offsetL[idx]
            azimuth = self.azimuthL[idx]
            data_masked = self.maskL[idx]
            mask_01 = self.mask_01L[idx]
            cmp_line = self.cmpL[idx]
            if len(self.modecountL) != 0:
                mode_count = self.modecountL[idx]
                print(f"Mode count: {mode_count}")
            if self.test_all_azimuth_buckets and len(self.bucket_idxL) > 0:
                bucket_idx = self.bucket_idxL[idx]
                return (
                    data.astype(np.float32),
                    data_masked.astype(np.float32),
                    cmp_x.astype(np.float32),
                    cmp_y.astype(np.float32),
                    offset.astype(np.float32),
                    azimuth.astype(np.float32),
                    cmp_line,
                    self.orderL[idx],
                    bucket_idx,
                )
            else:
                return (
                    data.astype(np.float32),
                    data_masked.astype(np.float32),
                    cmp_x.astype(np.float32),
                    cmp_y.astype(np.float32),
                    offset.astype(np.float32),
                    azimuth.astype(np.float32),
                    cmp_line,
                    self.orderL[idx],
                )

## 使用标签数据作为监督信号的有监督训练数据集
class RAW_dongfang_1031V2_supervised(Dataset):
    """
    使用标签数据作为监督信号的有监督训练数据集
    - 训练时：使用 label_cleaned_data 作为完整标签，mask 作为输入
    - 测试时：与训练时相同，使用固定的 mask
    - 按照检波点（recv_line, recv_no）划分训练集和测试集
    """
    def __init__(self, time_ps, trace_ps, trace_sd, time_sd, train=True, use_all_shots=True, train_ratio=0.8, random_seed=42):
        super().__init__()
        print('Loading Dongfang supervised dataset...')
        label_cleaned_dir = Path(
            "./dongfang/label/cleaned_data"
        )
        mask_dir = Path("./dongfang/aligned_raw_data")
        pattern = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        label_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            match = pattern.match(file.name)
            if match:
                recv_line = int(match.group(1))
                recv_no = int(match.group(2))
                label_files.append((recv_line, recv_no, file))
        print(f"Total receiver points: {len(label_files)}")
        
        # 按照检波点划分训练集和测试集
        # 获取所有唯一的检波点（recv_line, recv_no）
        receiver_points = list(set((recv_line, recv_no) for recv_line, recv_no, _ in label_files))
        receiver_points.sort()  # 排序以保证可重复性
        
        rng_split = np.random.default_rng(random_seed)
        n_total = len(receiver_points)
        n_train = int(n_total * train_ratio)
        train_indices = rng_split.choice(n_total, size=n_train, replace=False)
        train_receivers = {receiver_points[i] for i in train_indices}
        test_receivers = set(receiver_points) - train_receivers
        
        # 根据 train 参数筛选对应的检波点
        if train:
            target_receivers = train_receivers
            print(f"Train mode: Using {len(target_receivers)} receiver points for training ({100*train_ratio:.1f}%)")
        else:
            target_receivers = test_receivers
            print(f"Test mode: Using {len(target_receivers)} receiver points for testing ({100*(1-train_ratio):.1f}%)")
        
        # 筛选出对应检波点的文件
        label_files = [
            (recv_line, recv_no, file) 
            for recv_line, recv_no, file in label_files 
            if (recv_line, recv_no) in target_receivers
        ]
        print(f"Filtered files for current split: {len(label_files)}")
        self._rng = np.random.default_rng(123)
        self.train = train
        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.time_sd = time_sd
        self.trace_sd = trace_sd
        self.dt_ms = 4
        self.t0_ms = 0
        self.use_all_shots = use_all_shots
        
        # 使用 label_cleaned_dir 计算统计信息
        stats = self.compute_coord_stats(label_files, label_cleaned_dir, use_raw=False)
        
        if train:
            # 训练模式：存储整 gather + 预先计算 patch 索引
            self.gathers_data = []
            self.gathers_mask = []
            self.gathers_rx = []
            self.gathers_ry = []
            self.gathers_sx = []
            self.gathers_sy = []
            self.patch_index = []
        else:
            # 测试模式：存储整 gather
            dataL, rxL, ryL, sxL, syL = [], [], [], [], []
            maskL = []
            mask_01L = []
        
        print('use normalize_clip')
        
        for recv_line, recv_no, file in tqdm(label_files, desc="Loading supervised data"):
            label_file = (
                label_cleaned_dir
                / f"label_cleaned_data_recl_{recv_line}_recn_{recv_no}.npy"
            )
            label_attrs_file = (
                label_cleaned_dir
                / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            )
            mask_file = (
                mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
            )
            if not mask_file.exists() or not label_file.exists() or not label_attrs_file.exists():
                continue
                
            label = np.load(label_file)
            label = normalize_clip(label)
            label_attrs = np.load(label_attrs_file)
            mask = np.load(mask_file)
            
            sx_old = label_attrs["shot_x"]
            sy_old = label_attrs["shot_y"]
            rx_old = label_attrs["rec_x"]
            ry_old = label_attrs["rec_y"]
            shot_line = label_attrs["shot_line"]
            
            # 过滤有效数据（保持与原始测试逻辑一致）
            unique_lines, counts = np.unique(shot_line, return_counts=True)
            mode_count = mode(counts, keepdims=True).mode[0]
            valid_lines = unique_lines[counts == mode_count]
            valid_indices_mask = np.isin(shot_line, valid_lines)
            label = label[valid_indices_mask]
            mask = mask[valid_indices_mask]
            sx_old = sx_old[valid_indices_mask]
            sy_old = sy_old[valid_indices_mask]
            rx_old = rx_old[valid_indices_mask]
            ry_old = ry_old[valid_indices_mask]
            shot_line = shot_line[valid_indices_mask]
            
            # 排序：使用 rx, ry, sx, sy 进行排序
            order = np.lexsort((ry_old, rx_old, sy_old, sx_old))
            
            sx_old = sx_old[order]
            sy_old = sy_old[order]
            rx_old = rx_old[order]
            ry_old = ry_old[order]
            label = label[order] 
            mask = mask[order]
            shot_line = shot_line[order]  
            
            # 归一化坐标
            sx = 2*(sx_old-stats["sx_min"])/(stats['sx_max']-stats['sx_min'])-1
            sy = 2*(sy_old-stats["sy_min"])/(stats['sy_max']-stats['sy_min'])-1
            rx = 2*(rx_old-stats["rx_min"])/(stats['rx_max']-stats['rx_min'])-1
            ry = 2*(ry_old-stats["ry_min"])/(stats['ry_max']-stats['ry_min'])-1
            
            # 按 shot_line 归一化数据
            shot_line_unique = np.unique(shot_line)
            for line_id in shot_line_unique:
                indices = np.where(shot_line == line_id)[0]
                label[indices] = normalize_clip(label[indices])
            
            # 选择 shot_line：训练时可以选择使用所有或部分
            if use_all_shots:
                selected_indices = np.arange(len(label))
            else:
                # 默认使用前4个 shot_line（与测试逻辑一致）
                idx = shot_line_unique[:4]
                idx = np.sort(idx)
                selected_indices = np.where(np.isin(shot_line, idx))[0]
            
            sx_selected = sx[selected_indices]
            sy_selected = sy[selected_indices]
            rx_arr = rx[selected_indices]
            ry_arr = ry[selected_indices]
            data_selected = label[selected_indices]
            mask_full = mask[selected_indices]  # 原始 mask 数据
            
            # 生成 mask_01（标记哪些trace有数据）
            mask_01 = (np.any(mask_full > 0, axis=1)).astype(np.float32)
            
            if train:
                # 训练模式：存储整 gather，预先计算 patch 索引
                self.gathers_data.append(data_selected.astype(np.float32))
                self.gathers_mask.append(mask_full.astype(np.float32))
                self.gathers_rx.append(rx_arr.astype(np.float32))
                self.gathers_ry.append(ry_arr.astype(np.float32))
                self.gathers_sx.append(sx_selected.astype(np.float32))
                self.gathers_sy.append(sy_selected.astype(np.float32))
                
                # 当前 gather 在列表中的索引
                g_id = len(self.gathers_data) - 1
                
                patches, rx_p, ry_p, sx_p, sy_p, t_idx = gen_patches(
                        data_selected,
                        [rx_arr, ry_arr, sx_selected, sy_selected],
                        patch_size=(self.trace_ps, self.time_ps),
                        stride=(self.trace_sd, self.time_sd),
                    )
                    
                patches_f, coords_f, kept_idx = keep_noblank_patches(
                    patches, [rx_p, ry_p, sx_p, sy_p]
                )
                if patches_f is None:
                    continue
                n_j = (1251 - self.time_ps) // self.time_sd + 1
                for k in kept_idx:
                    trace_step = k // n_j
                    time_step = k % n_j
                    i_start = trace_step * self.trace_sd
                    j_start = time_step * self.time_sd
                    self.patch_index.append((g_id, i_start, j_start))
            else:
                # 测试模式：存储整 gather
                mask_selected = data_selected * mask_01[:, None]
                assert np.allclose(mask_selected[mask_01 == 0], 0.0)
                dataL.append(data_selected.astype(np.float32))
                sxL.append(sx_selected.astype(np.float32))
                syL.append(sy_selected.astype(np.float32))
                rxL.append(rx_arr.astype(np.float32))
                ryL.append(ry_arr.astype(np.float32))
                maskL.append(mask_selected.astype(np.float32))
                mask_01L.append(mask_01.astype(np.float32))

        if train:
            print(f"Total training patches (indexed): {len(self.patch_index)}")
            print(f'Begin Training at {datetime.datetime.now()}')
        else:
            self.dataL = dataL
            self.rxL = rxL
            self.ryL = ryL
            self.sxL = sxL
            self.syL = syL
            self.maskL = maskL
            self.mask_01L = mask_01L
            print(f"Total test gathers: {len(self.dataL)}")
            print(f'Begin Testing at {datetime.datetime.now()}')

    def __len__(self):
        if self.train:
            return len(self.patch_index)
        else:
            return len(self.dataL)
    
    def compute_coord_stats(self, label_files, attrs_dir, use_raw=True):
        sx_all, sy_all, rx_all, ry_all = [], [], [], []
        for recv_line, recv_no, _ in label_files:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            else:
                attrs_fp = attrs_dir / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            if not attrs_fp.exists():
                continue
            attrs = np.load(attrs_fp)
            sx_all.append(attrs["shot_x"])
            sy_all.append(attrs["shot_y"])
            rx_all.append(attrs["rec_x"])
            ry_all.append(attrs["rec_y"])

        sx_all = np.concatenate(sx_all); sy_all = np.concatenate(sy_all)
        rx_all = np.concatenate(rx_all); ry_all = np.concatenate(ry_all)

        stats = {
            "sx_min": sx_all.min(), "sx_max": sx_all.max(),
            "sy_min": sy_all.min(), "sy_max": sy_all.max(),
            "rx_min": rx_all.min(), "rx_max": rx_all.max(),
            "ry_min": ry_all.min(), "ry_max": ry_all.max(),
        }
        stats["Lx"] = 0.5 * max(stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"])
        stats["Ly"] = 0.5 * max(stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"])
        return stats

    def __getitem__(self, idx):
        if self.train:
            # 训练模式：从整 gather 中切 patch
            g_id, i_start, j_start = self.patch_index[idx]
            
            data_full = self.gathers_data[g_id]  # (N_traces, N_samples)
            mask_full = self.gathers_mask[g_id]   # (N_traces, N_samples)
            rx_full = self.gathers_rx[g_id]
            ry_full = self.gathers_ry[g_id]
            sx_full = self.gathers_sx[g_id]
            sy_full = self.gathers_sy[g_id]
            
            # 在整 gather 上应用 mask：mask_full 标记哪些位置有数据
            mask_01_full = (np.any(mask_full > 0, axis=1)).astype(np.float32)  # (N_traces,)
            masked_full = data_full * mask_01_full[:, None]  # 应用 mask
            
            # 按 (i_start, j_start) 切 patch
            ts = i_start
            tt = j_start
            te = ts + self.trace_ps
            te_t = tt + self.time_ps
            
            data_patch = data_full[ts:te, tt:te_t]
            masked_patch = masked_full[ts:te, tt:te_t]
            
            rx_patch = rx_full[ts:te]
            ry_patch = ry_full[ts:te]
            sx_patch = sx_full[ts:te]
            sy_patch = sy_full[ts:te]
            
            # 构造时间轴：起点是 tt（sample index）
            time_idx_1d = np.arange(tt, tt + self.time_ps, dtype=np.int32)
            time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
            time_axis_2d = np.tile(time_axis_1d[None, :], (self.trace_ps, 1))
            
            return (
                data_patch.astype(np.float32),       # 完整 patch
                masked_patch.astype(np.float32),     # 掩码后 patch
                rx_patch.astype(np.float32),
                ry_patch.astype(np.float32),
                sx_patch.astype(np.float32),
                sy_patch.astype(np.float32),
                time_axis_2d.astype(np.float32),
                rx_patch.astype(np.float32),
            )
        else:
            # 测试模式：返回整 gather
            data = self.dataL[idx]
            rx = self.rxL[idx]
            ry = self.ryL[idx]
            sx = self.sxL[idx]
            sy = self.syL[idx]
            data_masked = self.maskL[idx]
            mask = self.mask_01L[idx]
            return (
                data.astype(np.float32),
                data_masked.astype(np.float32),
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                mask.astype(np.float32),
                rx.astype(np.float32),
            )   

##tool class for get .mat format    
from scipy import io as sio
class mat_cover(Dataset):
    def __init__(self,save_path:str=None):
        super().__init__()
        print('Loading Dongfang dataset...')
        #raw_data_dir = Path("./dongfang/raw")
        label_cleaned_dir = Path(
            "./dongfang/label/cleaned_data"
        )
        mask_dir = Path("./dongfang/aligned_raw_data")
        pattern = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        label_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            match = pattern.match(file.name)
            if match:
                recv_line = int(match.group(1))
                recv_no = int(match.group(2))
                label_files.append((recv_line, recv_no, file))
        print(f"Total files: {len(label_files)}")
        dataL, rxL, ryL, sxL, syL, maskL = [], [], [], [], [], []
        timeL = [] 
        orderL=[]
        self._rng = np.random.default_rng(123)
        os.makedirs(save_path,exist_ok=True)
        self.save_path=save_path
        #self.train = train
        mask_01L=[]
        print('use normalize_clip')
        for recv_line, recv_no, file in tqdm(label_files, desc="Loading test data"):
            label_file = (
                label_cleaned_dir
                / f"label_cleaned_data_recl_{recv_line}_recn_{recv_no}.npy"
            )
            label_attrs_file = (
                label_cleaned_dir
                / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            )
            mask_file = (
                mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
            )
            if not mask_file.exists():
                continue
            label = np.load(label_file)
            label /= label.std()
            label = normalize_clip(label)
            #label /= label.std()
            label_attrs = np.load(label_attrs_file)
            mask = np.load(mask_file)
            sx_old = label_attrs["shot_x"]
            sy_old = label_attrs["shot_y"]
            rx_old = label_attrs["rec_x"]
            ry_old = label_attrs["rec_y"]
            shot_line = label_attrs["shot_line"]
            #print(len(np.unique(shot_line)))
            unique_lines, counts = np.unique(shot_line, return_counts=True)
            mode_count = mode(counts, keepdims=True).mode[0]
            valid_lines = unique_lines[counts == mode_count]
            valid_indices_mask = np.isin(shot_line, valid_lines)
            label = label[valid_indices_mask]
            mask = mask[valid_indices_mask]
            sx_old = sx_old[valid_indices_mask]
            sy_old = sy_old[valid_indices_mask]
            rx_old = rx_old[valid_indices_mask]
            ry_old = ry_old[valid_indices_mask]
            offset  = np.hypot(rx_old - sx_old, ry_old - sy_old)           
            azimuth = np.arctan2(sy_old - ry_old, sx_old - rx_old)/np.pi
            order = np.lexsort((azimuth,offset))
            mask = mask[order]
            mask_01 = (np.any(mask > 0, axis=1)).astype(np.float32)
            dataL.append(label[order])
            orderL.append(order)
            maskL.append(label[order] * mask_01[:,None])
            mask_01L.append(mask_01)
    
        self.dataL = dataL
        self.rxL = rxL
        self.ryL = ryL
        self.sxL = sxL
        self.syL = syL
        self.maskL = maskL
        self.mask_01L = mask_01L
        self.timeL = timeL
        self.label_files=label_files
        print(f"Total patches: {len(self.dataL)}")
    
    def get_matfile(self, idx):
        data = self.dataL[idx] #label
        #print(data.shape)
        mask = self.maskL[idx]
        mask_01 = self.mask_01L[idx]
        mask_01=mask_01.reshape(-1,31)
        (rec_line, rec_no, _) = self.label_files[idx]
        file_name=os.path.join(self.save_path,f'dongfang_recl_{rec_line}_recn_{rec_no}.mat')
        mat_data = {
        'data': data.T,
        'mask': mask.T,
        'mask_01': mask_01
        }
        print(data.shape)
        print(mask.shape)
        print(mask_01.shape)
        sio.savemat(file_name, mat_data)
        print(f'Saved {file_name} at {datetime.datetime.now()}')        

class RAW_dongfang_1031V1(Dataset):
    def __init__(self, label_file_path, time_ps, trace_ps, trace_sd, time_sd, train=True):
        super().__init__()
        print('Loading Dongfang dataset...')
        raw_data_dir = Path("./dongfang/raw")
        label_cleaned_dir = Path(
            "/home/czt/seismic_ddpm/Seis_DiT/dongfang/label/cleaned_data"
        )
        mask_dir = Path("./dongfang/aligned_raw_data")
        pattern = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        label_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            match = pattern.match(file.name)
            if match:
                recv_line = int(match.group(1))
                recv_no = int(match.group(2))
                label_files.append((recv_line, recv_no, file))

        dataL, rxL, ryL, sxL, syL, maskL = [], [], [], [], [], []
        self._rng = np.random.default_rng(123)
        self.train = train

        # ------------------------
        # 训练阶段：仅保存原始完整patch，不提前mask
        # ------------------------
        if train:
            for recv_line, recv_no, file in tqdm(label_files, desc="Loading training data"):
                raw_data_file = raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                raw_attrs_file = raw_data_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                if not raw_data_file.exists() or not raw_attrs_file.exists():
                    continue
                raw_data = np.load(raw_data_file)
                raw_attrs = np.load(raw_attrs_file)
                sx = raw_attrs["shot_x"]
                sy = raw_attrs["shot_y"]
                rx = raw_attrs["rec_x"]
                ry = raw_attrs["rec_y"]

                rx, ry, sx, sy = augment_coordinates(rx, ry, sx, sy)
                traces = normalize_clip(raw_data)
                patches, rx_p, ry_p, sx_p, sy_p, _ = gen_patches(
                    traces, [rx, ry, sx, sy],
                    (trace_ps, time_ps),
                    (trace_sd, time_sd)
                )

                dataL.extend(patches)
                sxL.extend(sx_p)
                syL.extend(sy_p)
                rxL.extend(rx_p)
                ryL.extend(ry_p)

        # ------------------------
        # 测试阶段：固定mask和数据
        # ------------------------
        else:
            for recv_line, recv_no, file in tqdm(label_files, desc="Loading test data"):
                label_file = (
                    label_cleaned_dir
                    / f"label_cleaned_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                label_attrs_file = (
                    label_cleaned_dir
                    / f"label_cleaned_attributes_recl_{recv_line}_recn_{recv_no}.npy"
                )
                mask_file = (
                    mask_dir / f"aligned_raw_data_recl_{recv_line}_recn_{recv_no}.npy"
                )
                if not mask_file.exists():
                    continue
                label = np.load(label_file)
                label_attrs = np.load(label_attrs_file)
                mask = np.load(mask_file)
                sx = label_attrs["shot_x"]
                sy = label_attrs["shot_y"]
                rx = label_attrs["rec_x"]
                ry = label_attrs["rec_y"]
                shot_line = label_attrs["shot_line"]

                shot_line_unique = np.unique(shot_line)
                idx = self._rng.choice(shot_line_unique, size=4, replace=False)
                idx = np.sort(idx)
                indices = np.where(np.isin(shot_line, idx))[0]

                sx_selected = sx[indices]
                sy_selected = sy[indices]
                rx_arr = rx[indices]
                ry_arr = ry[indices]
                data_selected = normalize_clip(label[indices])
                mask_selected = normalize_clip(mask[indices])

                dataL.append(data_selected)
                sxL.append(sx_selected)
                syL.append(sy_selected)
                rxL.append(rx_arr)
                ryL.append(ry_arr)
                maskL.append(mask_selected)

        self.dataL = dataL
        self.rxL = rxL
        self.ryL = ryL
        self.sxL = sxL
        self.syL = syL
        self.maskL = maskL
        print(f"Total patches: {len(self.dataL)}")
        print(f'Begin Training at {datetime.datetime.now()}')

    def __len__(self):
        return len(self.dataL)

    def __getitem__(self, idx):
        data = self.dataL[idx]
        rx = self.rxL[idx]
        ry = self.ryL[idx]
        sx = self.sxL[idx]
        sy = self.syL[idx]

        if self.train:
            missing_ratio = sample_missing_ratio()
            masked, mask = apply_mixed_mask(data, missing_ratio, block_prob=0.3)

            return (
                data.astype(np.float32),          # 完整数据
                masked.astype(np.float32),        # 掩码后输入        
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                np.mean(rx).astype(np.float32),
                rx.astype(np.float32),
            )
        else:
            # --- 测试阶段：固定mask ---
            data_masked = self.maskL[idx]
            return (
                data.astype(np.float32),
                data_masked.astype(np.float32),
                rx.astype(np.float32),
                ry.astype(np.float32),
                sx.astype(np.float32),
                sy.astype(np.float32),
                np.mean(rx).astype(np.float32),
                rx.astype(np.float32),
            ) 

class RAW_dongfang_neighbors(Dataset):
    def __init__(
        self,
        trace_num,
        time_num,
        neighbors_path,
        raw_data_dir="./dongfang/raw",
        attrs_dir="./dongfang/raw",
        label_file_dir="./dongfang/label/cleaned_data",
        attrs_align_dir="/home/chengzhitong/Seis_DiT/dongfang/aligned_raw_data_50m",
        dt_ms=4,
        t0_ms=0,
    ):
        super().__init__()
        print("Loading Dongfang neighbors-train dataset...")

        self.time_num = time_num
        self.trace_num = trace_num
        self.train = True  
        self.expand =0.25
        raw_data_dir = Path(raw_data_dir)
        attrs_dir = Path(attrs_dir)
        attrs_align_dir = Path(attrs_align_dir)
        label_cleaned_dir = Path(label_file_dir)
        pattern = re.compile(r"label_cleaned_data_recl_(\d+)_recn_(\d+).npy")
        raw_pattern = re.compile(r"raw_data_recl_(\d+)_recn_(\d+).npy")
        label_files = []
        for file in label_cleaned_dir.glob("label_cleaned_data_recl_*_recn_*.npy"):
            m = pattern.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                label_files.append((recv_line, recv_no, file))
        raw_files = []
        for file in raw_data_dir.glob("raw_data_recl_*_recn_*.npy"):
            m = raw_pattern.match(file.name)
            if m:
                recv_line = int(m.group(1))
                recv_no = int(m.group(2))
                raw_files.append((recv_line, recv_no, file))
        # 排序确保可复现：按 (recv_line, recv_no) 排序
        raw_files.sort(key=lambda x: (x[0], x[1]))
        #print(f"Total files (label_files): {len(label_files)}")
        print(f"Total files (raw_files): {len(raw_files)}")
        stats = self.compute_coord_stats(raw_files, attrs_dir,attrs_align_dir, use_raw=False)
        all_traces = []      # [N_traces, n_samples]
        all_sx = []          # shot x
        all_sy = []          # shot y
        all_rx = []          # rec x
        all_ry = []          # rec y
        all_offset = []      # offset（仅用于排序）
        all_azimuth = []     # azimuth（未归一化，仅用于排序）
        self._rng = np.random.default_rng(123)

        for recv_line, recv_no, _ in tqdm(raw_files, desc="Loading & normalizing traces"):
            raw_data_file = raw_data_dir / f"raw_data_recl_{recv_line}_recn_{recv_no}.npy"
            raw_attrs_file = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            raw_data = np.load(raw_data_file)  # [n_traces, n_samples]
            raw_attrs = np.load(raw_attrs_file)

            sx_old = raw_attrs["shot_x"]
            sy_old = raw_attrs["shot_y"]
            rx_old = raw_attrs["rec_x"]
            ry_old = raw_attrs["rec_y"]
            shot_line = raw_attrs["shot_line"]

            raw_data = raw_data.astype(np.float32)
            #raw_data  = normalize_clip(raw_data)
            shot_line_unique = np.unique(shot_line)
            for sl in shot_line_unique:
                idx = np.where(shot_line == sl)[0]
                raw_data[idx] = normalize_clip(raw_data[idx])
                
            offset = np.hypot(rx_old - sx_old, ry_old - sy_old)
            azimuth = np.arctan2(sy_old - ry_old, sx_old - rx_old)  
            all_traces.append(raw_data)   
            all_sx.append(sx_old)
            all_sy.append(sy_old)
            all_rx.append(rx_old)
            all_ry.append(ry_old)
            all_offset.append(offset)
            all_azimuth.append(azimuth)
        self.traces = np.concatenate(all_traces, axis=0)      # [N, n_samples]
        self.sx_all = np.concatenate(all_sx, axis=0)
        self.sy_all = np.concatenate(all_sy, axis=0)
        self.rx_all = np.concatenate(all_rx, axis=0)
        self.ry_all = np.concatenate(all_ry, axis=0)
        #offset_all = np.concatenate(all_offset, axis=0)
        #azimuth_all = np.concatenate(all_azimuth, axis=0)
        N_traces, n_samples = self.traces.shape
        print(f"Global traces: {N_traces}, samples per trace: {n_samples}")
        # 归一化到 [-1, 1]，使用全局统计量
        self.sx_all = 2.0 * (self.sx_all - stats["sx_min"]) / (stats['sx_max'] - stats["sx_min"]) - 1.0
        self.sy_all = 2.0 * (self.sy_all - stats["sy_min"]) / (stats['sy_max'] - stats["sy_min"]) - 1.0
        self.rx_all = 2.0 * (self.rx_all - stats["rx_min"]) / (stats['rx_max'] - stats["rx_min"]) - 1.0
        self.ry_all = 2.0 * (self.ry_all - stats["ry_min"]) / (stats['ry_max'] - stats["ry_min"]) - 1.0
        #sy_n = 2.0 * (sy_all - sy_min) / (sy_max - sy_min) - 1.0
        #rx_n = 2.0 * (rx_all - rx_min) / (rx_max - rx_min) - 1.0
        #ry_n = 2.0 * (ry_all - ry_min) / (ry_max - ry_min) - 1.0
        neighbors = np.load(neighbors_path)  
        neighbors = neighbors.astype(np.int64)
        print(f"Loaded neighbors: shape = {neighbors.shape}")
        self.neighbors = neighbors

    def compute_coord_stats(self, label_files, attrs_dir,attrs_align_dir ,use_raw=True):
        """
        计算全局的 rx, ry, sx, sy 统计量，用于归一化到 [-1, 1]。
        """
        sx_all, sy_all, rx_all, ry_all = [], [], [], []
        for recv_line, recv_no, _ in label_files:
            if use_raw:
                attrs_fp = attrs_dir / f"raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            else:
                attrs_fp = attrs_align_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
            if not attrs_fp.exists():
                continue
            attrs = np.load(attrs_fp)
            sx_all.append(attrs["shot_x"])
            sy_all.append(attrs["shot_y"])
            rx_all.append(attrs["rec_x"])
            ry_all.append(attrs["rec_y"])

        sx_all = np.concatenate(sx_all)
        sy_all = np.concatenate(sy_all)
        rx_all = np.concatenate(rx_all)
        ry_all = np.concatenate(ry_all)

        stats = {
            "sx_min": sx_all.min(),
            "sx_max": sx_all.max(),
            "sy_min": sy_all.min(),
            "sy_max": sy_all.max(),
            "rx_min": rx_all.min(),
            "rx_max": rx_all.max(),
            "ry_min": ry_all.min(),
            "ry_max": ry_all.max(),
        }
        stats["Lx"] = 0.5 * max(
            stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"]
        )
        stats["Ly"] = 0.5 * max(
            stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"]
        )
        return stats

    def __len__(self):
        return len(self.neighbors)*int(270/self.trace_num)
    
    def __getitem__(self, idx):
        #ng_idx = idx
        ng_idx = idx % len(self.neighbors)
        trace_ids = self.neighbors[ng_idx]  # [k]
        trace_ids = trace_ids[(trace_ids >= 0) & (trace_ids < self.traces.shape[0])]
        traces_block = self.traces[trace_ids]  
        if self.time_num is not None and traces_block.shape[-1]>self.time_num:
            diff = traces_block.shape[-1] - self.time_num
            if diff > 0:
                traces_block = traces_block[:,diff:]# [k, n_samples]
        rx_block = self.rx_all[trace_ids]
        ry_block = self.ry_all[trace_ids]
        sx_block = self.sx_all[trace_ids]
        sy_block = self.sy_all[trace_ids]
        trace_num_all,_ = traces_block.shape
        assert trace_num_all>=self.trace_num,f"trace_num_all:{trace_num_all},self.trace_num:{self.trace_num}"
        trace_num = int(min(trace_num_all,self.trace_num*(1+self.expand)))
        trace_id_0 = np.random.randint(0,trace_num_all-trace_num+1)
        traces = np.random.choice(np.arange(trace_num), self.trace_num, replace=False)
        traces = np.sort(traces)
        traces = trace_id_0 + traces
        traces_block_patch = traces_block[traces]
        rx_block_patch = rx_block[traces]
        ry_block_patch = ry_block[traces]
        sx_block_patch = sx_block[traces]
        sy_block_patch = sy_block[traces]
        trace_num_patch,time_num_patch = traces_block_patch.shape
        missing_ratio = sample_missing_ratio()
        masked, _ = apply_mixed_mask(traces_block_patch, missing_ratio, block_prob=0.0)
        return (
            traces_block_patch.astype(np.float32),
            masked.astype(np.float32),
            rx_block_patch.astype(np.float32),
            ry_block_patch.astype(np.float32),
            sx_block_patch.astype(np.float32),
            sy_block_patch.astype(np.float32),
            np.mean(rx_block_patch).astype(np.float32),
            rx_block_patch.astype(np.float32),
        )
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
    }
    )
    dataset = RAW_dongfang_1031V2(
                    time_ps=1248,
                    trace_ps=128,
                    train=True
                    )
    data_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=4)
    for data,data_masked, rx, ry, sx, sy,time,_ in data_loader:
        print(rx.min(),rx.max())
        print(ry.min(),ry.max())
        print(sx.min(),sx.max())
        print(sy.min(),sy.max())
        break
    data,data_masked, rx, ry, sx, sy,time,_= dataset[0]
    print('space scale:',dataset.space_scale)
    print(data.shape)
    print(data_masked.shape)
    
    plt.figure(figsize=(8,12))
    plt.subplot(3,1,1)
    plt.imshow(data.T,vmin=-data.std(),vmax=data.std(),cmap='seismic',aspect='auto')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(data_masked.T,vmin=-data.std(),vmax=data.std(),cmap='seismic',aspect='auto')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(data_masked.T-data.T,vmin=-data.std(),vmax=data.std(),cmap='seismic',aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./dongfang_test.png')
    plt.close()
    def denorm(u, vmin=-1, vmax=1):
        # u in [-1,1] -> physical
        return vmin + (u + 1.0) * 0.5 * (vmax - vmin)
    def plot_std_distribution(dataset, max_patches=None, bins=80):
        std_list, tau_list,mask_list,offset_list = [],[],[],[]
        n_total = len(dataset)
        for i in trange(n_total):
            out = dataset[i]
            data = np.asarray(out[0])
            mask =np.asarray(out[1])
            log_tau =np.asarray(out[-2])
            #print(out[2].shape)
            rx=denorm(np.asarray(out[2]),vmin =np.asarray(out[2]).min(),vmax=np.asarray(out[2]).max())
            ry=denorm(np.asarray(out[3]),vmin =np.asarray(out[3]).min(),vmax=np.asarray(out[3]).max())
            sx=denorm(np.asarray(out[4]),vmin =np.asarray(out[4]).min(),vmax=np.asarray(out[4]).max())
            sy=denorm(np.asarray(out[5]),vmin =np.asarray(out[5]).min(),vmax=np.asarray(out[5]).max())   
            offset = np.hypot(sx - rx, sy - ry)
            std_list.append(float(data.std()))
            tau_list.append(float(np.exp(data.std())))
            mask_list.append(mask.mean())
            offset_list.append(offset.mean())

            if (max_patches is not None) and (len(std_list) >= max_patches):
                break

        stds = np.array(std_list, dtype=np.float32)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(stds, bins=bins)
        ax.set_title("Per-patch std after robust scaling")
        ax.set_xlabel("std")
        ax.set_ylabel("count")
        fig.tight_layout()
        plt.savefig('./dongfang_std_distribution.png')
        print(f"N={len(stds)}")
        print(f"mean={stds.mean():.4f}, median={np.median(stds):.4f}")
        print(f"p05={np.percentile(stds,5):.4f}, p95={np.percentile(stds,95):.4f}")
        return stds,tau_list,mask_list,offset_list
    stds, tau_list,mask_list,offset_list = plot_std_distribution(dataset, max_patches=1000000,bins=80)
    def corr(x, y, name):
        x = np.asarray(x); y = np.asarray(y)
        r = np.corrcoef(x, y)[0,1]
        print(f"corr(std, {name}) = {r:.3f}")
    corr(stds, tau_list, "tau")
    corr(stds, mask_list, "mask")
    corr(stds, offset_list, "offset")
    plt.close()
    '''mat_get = mat_cover(save_path='./dongfang/mat')
    for i in range(len(mat_get.dataL)):
        mat_get.get_matfile(i)
        break'''
    '''dir_path = './dongfang/mat'
    label_path='/home/chengzhitong/Seis_DiT/20251127_interp_dpmsolver/data_0.npy'
    mask_path='/home/chengzhitong/Seis_DiT/20251127_interp_dpmsolver/mask_0.npy'
    dit_path ='/home/chengzhitong/Seis_DiT/20251127_interp_dpmsolver/sampled_imgs_0.npy'
    label =np.load(label_path)
    mask =np.load(mask_path)
    dit_sample =np.load(dit_path)
    label =label[0,0,0:115*4]
    mask =mask[0,0,0:115*4]
    dit_sample =dit_sample[0,0]
    mask_01 = (np.any(mask > 0, axis=1)).astype(np.float32)
    print(label.shape)
    print(mask.shape)
    print(mask_01.shape)
    print(dit_sample.shape)
    data_dict={
        'data':label.T,
        'mask':mask.T,
        'mask_01':mask_01.reshape(115,4,order='F'),
        'DATA_dit':dit_sample.T,
        }
    sio.savemat(os.path.join(dir_path, 'test_dongfang.mat'), data_dict)'''
    