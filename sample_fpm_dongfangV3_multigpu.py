from torch.autograd.forward_ad import exit_dual_level
import seisDiT
import seisdit_trace_axis
import torch
import torch.nn as nn
import FPM  # Flow Matching Model
import pathlib
from typing import BinaryIO, List, Union
from matplotlib import pyplot as plt
import os
import tqdm
import datetime
import sys
import random
import numpy as np
import patched_dataset5d
from train_fpmV3_ddp import time_PS
import time
from matplotlib.colors import LinearSegmentedColormap
import missing_line_inpainting
#python sample_fpm_dongfangV3_multigpu.py interp parallel 1
# ================= 配置区域 =================
PS = 16*3
SD = 16*3
time_PS = 1248
time_SD = 1248
BIN = 50

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

colors = [
    (0.0, "black"),
    (0.5, "white"),
    (1.0, "red"),
]
red_black_cmap = LinearSegmentedColormap.from_list("red_black", colors)

# Flow Matching 参数配置
fpm_kwargs = dict(
    time_num=time_PS,
    trace_num=PS,
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    sample_num=int(sys.argv[3]) if len(sys.argv) > 3 else 1,
    device=None,
    sup_mode="all",
    use_coherence=False,
    sigma_obs=0.001,
    use_bayesian=False,
    sampling_method="ode",
    ode_sampling_method="euler",
    ode_num_steps=50,
    ode_atol=1e-6,
    ode_rtol=1e-3,
    sde_sampling_method="Euler",
    sde_num_steps=250,
    sde_diffusion_form="sigma",
    sde_diffusion_norm=1.0,
    sde_last_step="Mean",
    sde_last_step_size=0.04,
)

pe_type = "transformer"
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "grid.alpha": 0.3,
})

dir_path = f'./{datetime.datetime.now().strftime("%Y%m%d")}_{sys.argv[1]}_{sys.argv[2]}_fpm_{BIN}m'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

# ================= GPU 配置 =================
num_gpus = torch.cuda.device_count()
print(f"检测到 {num_gpus} 张 GPU")
device_ids = list(range(num_gpus))
primary_device = torch.device("cuda:0") # DataParallel 主设备

image_channels = 2 if sys.argv[1] in ["interp"] else 1
model_path = "/home/chengzhitong/Seis_DiT/resultsFPM/dit_datatype_dongfang_V15rope_wo_adaln_Linear_velocity/checkpoints/model-15.pth"
model_path = '/home/chengzhitong/Seis_DiT/resultsFPM/dit_datatype_dongfang_V15rope_wo_adalnV2_Linear_velocity_p_scaleTrue/checkpoints/model-15.pth'
# ================= 模型加载 =================
# 1. 初始化基础模型 (不要在这里用 DataParallel)
model = seisdit_trace_axis.SeisDiTRopeV2(
    image_channels=image_channels,
    d_model=512,
    num_layers=12,
    pe_type=pe_type,
    n_channels=64 // 2,
    rope_p_scale=patched_dataset5d.RAW_dongfang_1031V2(time_ps=time_PS,trace_ps=PS).space_scale,
)

state_dict = torch.load(model_path, map_location="cpu")["model"]
model.load_state_dict(state_dict)
model.eval()
# 先把基础模型放到主设备（DataParallel 会自动复制到其他卡）
model.to(primary_device) 

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

same_seeds(515)

# ================= 核心修改：并行采样包装器 =================

class ParallelFlowMatchingSampler(nn.Module):
    """
    将 FPM 的 sample 方法包装成 nn.Module 的 forward 方法。
    这样 nn.DataParallel 就可以自动切分数据，
    并在每个 GPU 上独立运行完整的 ODE/SDE 循环，最大化并行效率。
    """
    def __init__(self, fpm_instance):
        super().__init__()
        # fpm_instance 中包含了 model (它是 nn.Module)
        # DataParallel 会自动处理 self.fpm.model 的复制和设备移动
        self.fpm = fpm_instance

    def forward(self, x_cond, rx, ry, sx, sy, time_axis=None):
        # x_cond 是已经被 DataParallel 切分后的部分数据
        batch_size = x_cond.shape[0]
        device = x_cond.device
        
        # 更新 FPM 实例的内部状态以匹配当前 GPU 分片
        self.fpm.device = device
        self.fpm.sample_num = batch_size
        
        # 组装条件
        condL = (rx, ry, sx, sy)
        
        # 调用原始的采样逻辑 (这将在当前 GPU 上运行完整的循环)
        return self.fpm.sample(condL=condL, x_cond=x_cond, time_axis=time_axis)

# 创建 Flow Matching 基础实例
fpm = FPM.FlowMatchingModel(
    model=model,  # 这里传入的是原始 model，不是 DataParallel
    trace_num=fpm_kwargs["trace_num"],
    time_steps=fpm_kwargs["time_num"],
    path_type=fpm_kwargs["path_type"],
    prediction=fpm_kwargs["prediction"],
    loss_weight=fpm_kwargs["loss_weight"],
    train_eps=fpm_kwargs["train_eps"],
    sample_eps=fpm_kwargs["sample_eps"],
    sample_num=fpm_kwargs["sample_num"],
    device=primary_device,
    sup_mode=fpm_kwargs["sup_mode"],
    use_coherence=fpm_kwargs["use_coherence"],
    sigma_obs=fpm_kwargs["sigma_obs"],
    use_bayesian=fpm_kwargs["use_bayesian"],
    sampling_method=fpm_kwargs["sampling_method"],
    ode_sampling_method=fpm_kwargs["ode_sampling_method"],
    ode_num_steps=fpm_kwargs["ode_num_steps"],
    ode_atol=fpm_kwargs["ode_atol"],
    ode_rtol=fpm_kwargs["ode_rtol"],
    sde_sampling_method=fpm_kwargs["sde_sampling_method"],
    sde_num_steps=fpm_kwargs["sde_num_steps"],
    sde_diffusion_form=fpm_kwargs["sde_diffusion_form"],
    sde_diffusion_norm=fpm_kwargs["sde_diffusion_norm"],
    sde_last_step=fpm_kwargs["sde_last_step"],
    sde_last_step_size=fpm_kwargs["sde_last_step_size"],
)

# 使用 DataParallel 包装采样器
# 这会自动将输入 batch 切分并分发到各个 GPU
parallel_sampler = ParallelFlowMatchingSampler(fpm)
if num_gpus > 1:
    parallel_sampler = nn.DataParallel(parallel_sampler, device_ids=device_ids)
    print(f"并行采样模式已启用: {device_ids}")

parallel_sampler.to(primary_device)

# ================= 辅助函数 =================

def save_image(tensor, fp, norm_=True, diff=False, order=None):
    # (保持原样...)
    assert len(tensor.shape) == 3
    tensor = tensor[0, :, :].detach().cpu()
    tensor = tensor.numpy()
    plt.figure(figsize=(6, 6))
    plt.pcolor(tensor.T, cmap=red_black_cmap, vmin=-tensor.std(), vmax=tensor.std())
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel("Trace Number")
    plt.ylabel("Time(ms)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fp, dpi=600)
    plt.close()

# (plot_shot_line_comparison 等其他辅助函数保持原样...)
def get_shot_line_info(recv_line, recv_no):
    # (保持原样)
    from pathlib import Path
    mask_dir = Path('./dongfang/aligned_raw_data_25m')
    label_attrs_file = mask_dir / f"aligned_raw_attributes_recl_{recv_line}_recn_{recv_no}.npy"
    if not label_attrs_file.exists(): return None
    label_attrs = np.load(label_attrs_file)
    return label_attrs["shot_line"]

def select_random_adjacent_shot_lines(shot_line_arr, num_lines=3):
    # (保持原样)
    unique_shot_lines = np.sort(np.unique(shot_line_arr))
    if len(unique_shot_lines) < num_lines: return None, None
    start_idx = random.randint(0, len(unique_shot_lines) - num_lines)
    selected = unique_shot_lines[start_idx : start_idx + num_lines]
    indices = {sl: np.where(shot_line_arr == sl)[0] for sl in selected}
    return list(selected), indices

def plot_shot_line_comparison(ori_data, masked_data, sampled_data, shot_line_arr, selected_shot_lines, indices_by_line, save_path, recv_line, recv_no):
    """
    Plot comparison figure for 3 adjacent shot lines within a single receiver point
    
    Args:
        ori_data: Original data (n_traces, n_samples)
        masked_data: Masked data (n_traces, n_samples)
        sampled_data: Sampling results (n_traces, n_samples)
        shot_line_arr: Shot line array
        selected_shot_lines: Selected shot line numbers
        indices_by_line: Trace indices for each shot line
        save_path: Save path
        recv_line: Receiver line number
        recv_no: Receiver point number
    """
    n_lines = len(selected_shot_lines)
   
    # Extract data for each shot line
    ori_list, masked_list, sampled_list, info_list = [], [], [], []
    for sl in selected_shot_lines:
        indices = indices_by_line[sl]
        ori_list.append(ori_data[indices, :])
        masked_list.append(masked_data[indices, :])
        sampled_list.append(sampled_data[indices, :])
        info_list.append(f'Shot Line {sl}')
    
    # Horizontally concatenate data from 3 adjacent shot lines
    # 更简洁的写法
    ori_combined = np.vstack(ori_list)
    masked_combined = np.vstack(masked_list)
    sampled_combined = np.vstack(sampled_list)
    diff_combined = np.abs(ori_combined - sampled_combined)
    np.save(f"{dir_path}/ori_combined_{recv_line}_{recv_no}.npy", ori_combined)
    np.save(f"{dir_path}/masked_combined_{recv_line}_{recv_no}.npy", masked_combined)
    np.save(f"{dir_path}/sampled_combined_{recv_line}_{recv_no}.npy", sampled_combined)
    np.save(f"{dir_path}/info_list_{recv_line}_{recv_no}.npy", info_list)
    

    # Calculate unified color ranges
    vmax_data = max(
        ori_combined.std(), masked_combined.std(), sampled_combined.std()
    ) 
    vmax_diff = diff_combined.std() * 3 if diff_combined.std() > 0 else 0.1
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()
    
    # Separation line positions
    W_cumsum = [0]
    for ol in ori_list:
        W_cumsum.append(W_cumsum[-1] + ol.shape[0])
    
    title_prefix = f'Receiver Point ({recv_line}, {recv_no})'
    
    # Original data
    im0 = axes[0].pcolormesh(ori_combined.T, cmap=red_black_cmap, 
                             vmin=-vmax_data, vmax=vmax_data)
    axes[0].set_title(f'{title_prefix}\nOriginal Data ', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Time (ms)')
    axes[0].invert_yaxis()
    for w in W_cumsum[1:-1]:
        axes[0].axvline(x=w, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Masked data
    im1 = axes[1].pcolormesh(masked_combined.T, cmap=red_black_cmap, 
                             vmin=-vmax_data, vmax=vmax_data)
    axes[1].set_title(f'{title_prefix}\nMasked Data ', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Trace')
    axes[1].set_ylabel('Time (ms)')
    axes[1].invert_yaxis()
    for w in W_cumsum[1:-1]:
        axes[1].axvline(x=w, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Sampling results
    im2 = axes[2].pcolormesh(sampled_combined.T, cmap=red_black_cmap, 
                             vmin=-vmax_data, vmax=vmax_data)
    axes[2].set_title(f'{title_prefix}\nSampling Results', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Trace')
    axes[2].set_ylabel('Time (ms)')
    axes[2].invert_yaxis()
    for w in W_cumsum[1:-1]:
        axes[2].axvline(x=w, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved to: {save_path}")


class TimeStats:
    # (保持原样)
    def __init__(self): self.reset()
    def reset(self):
        self.sample_times = []
        self.total_patches = 0
    def add_sample_time(self, t, p, s=1):
        self.sample_times.append(t)
        self.total_patches += p
    def print_summary(self):
        if not self.sample_times: return
        total = sum(self.sample_times)
        print(f"\nTotal Time: {total:.2f}s, Throughput: {self.total_patches/total:.2f} patches/s")

time_stats = TimeStats()

# ================= 主循环 =================

if sys.argv[2] == "parallel":
    # 多卡时 Batch Size 可以设大一点，DataParallel 会除以显卡数量
    MAX_BATCH_SIZE =15 * num_gpus 
    
    dataset_test = patched_dataset5d.RAW_dongfang_1031V2(
        time_ps=time_PS,
        trace_ps=PS,
        train=False,
        bin_size=BIN,
    )
    dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4
    )
    
    for iter, (data, data_mask, rx, ry, sx, sy, recv_line, recv_no) in tqdm.tqdm(enumerate(dl)):
        print(f"Processing sample {iter+1}/{len(dl)} | Recv: ({recv_line.item()}, {recv_no.item()})")
        
        # 数据预处理 (转tensor, unsqueeze等)
        if isinstance(data, np.ndarray): data_tensor = torch.tensor(data)
        else: data_tensor = data
        if data_tensor.ndim == 3: data_tensor = data_tensor.unsqueeze(1)
        
        if isinstance(data_mask, np.ndarray): data_mask_t = torch.tensor(data_mask)
        else: data_mask_t = data_mask
        if data_mask_t.ndim == 3: data_mask_t = data_mask_t.unsqueeze(1)

        # 移动到 GPU
        data_tensor = data_tensor.to(primary_device)
        data_mask_t = data_mask_t.to(primary_device)
        rx = rx.to(primary_device)
        ry = ry.to(primary_device)
        sx = sx.to(primary_device)
        sy = sy.to(primary_device)
        condL = (rx, ry, sx, sy)
        
        B, C, H, W = data_tensor.shape
        ps = min(PS, H)
        sd = min(SD, H)
        
        # 生成 Patch
        patches, rx_s, ry_s, sx_s, sy_s, t_idx_s = patched_dataset5d.gen_patches_torch(
            data_mask_t, condL, (ps, time_PS), (sd, time_SD), return_t=True
        )
        
        # 归一化 Patches
        K = 1
        P = len(patches)
        thres_list = []
        normalized_patches = []
        
        for p in patches:
            thres = np.percentile(np.abs(p.cpu().numpy()), 99.5)
            if thres == 0: thres = 1e-6
            thres_list.append(thres)
            p_clamped = torch.clamp(p, -thres, thres)
            normalized_patches.append((p_clamped / thres).to(primary_device))
            
        x_batch = torch.cat(normalized_patches, dim=0).unsqueeze(1) # (P, 1, H, W)
        
        # 拼接条件
        rx_b = torch.cat([r.to(primary_device) for r in rx_s], dim=0)
        ry_b = torch.cat([r.to(primary_device) for r in ry_s], dim=0)
        sx_b = torch.cat([s.to(primary_device) for s in sx_s], dim=0)
        sy_b = torch.cat([y.to(primary_device) for y in sy_s], dim=0)
        t_idx_b = t_idx_s.to(primary_device) if t_idx_s is not None else None
        
        if K > 1:
            # 这里的 repeat 逻辑... (省略，保持你原逻辑)
            pass

        # === 批次处理逻辑优化 ===
        sampled_chunks = []
        num_chunks = (P + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        print(f"分成 {num_chunks} 个块处理...")
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.inference_mode():
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_BATCH_SIZE
                end = min(start + MAX_BATCH_SIZE, P)
                
                # 准备当前 Chunk 数据
                x_chunk = x_batch[start:end]
                rx_chunk = rx_b[start:end]
                ry_chunk = ry_b[start:end]
                sx_chunk = sx_b[start:end]
                sy_chunk = sy_b[start:end]
                t_chunk = t_idx_b[start:end] if t_idx_b is not None else None
                
                # === 关键点：调用并行采样器 ===
                # 这里直接传入 tensor，DataParallel 会自动按 dim=0 切分给各 GPU
                # parallel_sampler.forward() 会在各 GPU 上并发执行
                y_chunk = parallel_sampler(
                    x_chunk, 
                    rx_chunk, ry_chunk, sx_chunk, sy_chunk, 
                    t_chunk
                )
                
                # 反归一化
                chunk_len = y_chunk.shape[0]
                y_chunk_list = []
                for i in range(chunk_len):
                    y_patch = y_chunk[i]
                    thres = thres_list[start + i]
                    y_denorm = (y_patch - y_patch.mean()) * thres
                    y_chunk_list.append(y_denorm)
                
                sampled_chunks.append(torch.stack(y_chunk_list, dim=0))

        y = torch.cat(sampled_chunks, dim=0)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        time_stats.add_sample_time(elapsed, P)
        print(f"  Sampled {P} patches in {elapsed:.2f}s")
        
        # 重构图像
        sampleL = [y[i].unsqueeze(0) for i in range(P)]
        sampled_imgs = patched_dataset5d.reconstruct_from_patches_torch(
            sampleL, 1, (H, W), (ps, time_PS), (sd, time_SD)
        )
        
        # 保存结果
        np.save(f"{dir_path}/sampled_imgs_{recv_line.item()}_{recv_no.item()}.npy", sampled_imgs.cpu().numpy())
        
        # 绘图逻辑 (Shot Line Comparison)
        rl, rn = recv_line.item(), recv_no.item()
        shot_line_arr = get_shot_line_info(rl, rn)
        if shot_line_arr is not None:
            sel_lines, idx_lines = select_random_adjacent_shot_lines(shot_line_arr)
            if sel_lines:
                plot_shot_line_comparison(
                    data_tensor[0,0].cpu().numpy(),
                    data_mask_t[0,0].cpu().numpy(),
                    sampled_imgs[0,0].cpu().numpy(),
                    shot_line_arr, sel_lines, idx_lines,
                    f"{dir_path}/comp_{rl}_{rn}.png", rl, rn
                )
        
        if iter >= 100000: break

    time_stats.print_summary()
elif sys.argv[2] == "missing_line":
    csv_path = "./mask_reports/per_receiver_20260121_231141.csv"
    ps = 16
    assert ps*3==PS,"PS must be 3 times of ps"
    inpainter = missing_line_inpainting.MissingLineRegular(time_PS,ps,bin_size=BIN,K=2*ps,sort_by='shot')
    receivers = missing_line_inpainting.load_receivers_with_d_missing(csv_path)
    
    # Load label data directory
    label_cleaned_dir = pathlib.Path("./dongfang/label/cleaned_data")
    
    for idx, (recl, recn) in enumerate(receivers):
        print(f"\n{'='*60}")
        print(f"Processing receiver {idx+1}/{len(receivers)}: recl={recl}, recn={recn}")
        print(f"{'='*60}")
        results = inpainter.get_patches(recl, recn)
        if len(results) == 0:
            print(f"No patches found for recl={recl}, recn={recn}")
            continue
        
        # Group patches by line_id
        patches_by_line = {}
        for result in results:
            line_id = result['line_id']
            if line_id not in patches_by_line:
                patches_by_line[line_id] = []
            patches_by_line[line_id].append(result)
        
        # Process each missing line
        for line_id, line_patches in patches_by_line.items():
            print(f"\nProcessing line {line_id} with {len(line_patches)} patches")
            
            # Collect all patches for this line
            all_concatenated_patches = []
            all_original_indices = []
            all_thres_list = []
            all_missing_info = []  # Store info for restoring each patch
            
            for patch_idx, result in enumerate(line_patches):
                concatenated_patch = result.get('concatenated_patch')
                original_indices = result.get('original_indices')
                
                if concatenated_patch is None or len(concatenated_patch) == 0:
                    print(f"  Skipping patch {patch_idx} (empty)")
                    continue
                
                # Normalize patch
                thres = np.percentile(np.abs(concatenated_patch), 99.5)
                if thres == 0:
                    thres = 1e-6
                all_thres_list.append(thres)
                
                # Store info for restoration
                n_knn = len(concatenated_patch) - ps
                missing_start_idx_original = n_knn
                missing_end_idx_original = n_knn + ps
                all_missing_info.append({
                    'missing_start_idx_original': missing_start_idx_original,
                    'missing_end_idx_original': missing_end_idx_original,
                    'original_indices': original_indices,
                    'time_start': result.get('time_start', 0),
                    'trace_indices': result.get('trace_indices', [])
                })
                
                all_concatenated_patches.append(concatenated_patch)
                all_original_indices.append(original_indices)
            
            if len(all_concatenated_patches) == 0:
                print(f"  No valid patches for line {line_id}")
                continue
            
            # Concatenate all patches along trace dimension
            # Each patch is (total_traces, time_PS), we stack them
            # But we need to handle coordinates separately for each patch
            print(f"  Concatenating {len(all_concatenated_patches)} patches for batch processing...")
            
            # Prepare batch data: collect all traces from all patches
            all_x_batch = []
            all_sx_batch = []
            all_sy_batch = []
            all_rx_batch = []
            all_ry_batch = []
            patch_boundaries = []  # Track which traces belong to which patch
            
            for patch_idx, (concatenated_patch, original_indices) in enumerate(zip(all_concatenated_patches, all_original_indices)):
                thres = all_thres_list[patch_idx]
                concatenated_patch_tensor = torch.tensor(concatenated_patch, dtype=torch.float32)
                concatenated_patch_clamped = torch.clamp(concatenated_patch_tensor, -thres, thres)
                normalized_patch = (concatenated_patch_clamped / thres).to(primary_device)
                
                # Get coordinates for this patch
                patch_result = line_patches[patch_idx]
                concatenated_sx = patch_result.get('concatenated_sx')
                concatenated_sy = patch_result.get('concatenated_sy')
                concatenated_rx = patch_result.get('concatenated_rx')
                concatenated_ry = patch_result.get('concatenated_ry')
                
                if concatenated_sx is None:
                    raise ValueError(f"Concatenated coordinates not found for patch {patch_idx}")
                
                # Store normalized patch and coordinates
                start_idx = len(all_x_batch)
                all_x_batch.append(normalized_patch)
                all_sx_batch.append(torch.tensor(concatenated_sx, dtype=torch.float32).to(primary_device))
                all_sy_batch.append(torch.tensor(concatenated_sy, dtype=torch.float32).to(primary_device))
                all_rx_batch.append(torch.tensor(concatenated_rx, dtype=torch.float32).to(primary_device))
                all_ry_batch.append(torch.tensor(concatenated_ry, dtype=torch.float32).to(primary_device))
                patch_boundaries.append((start_idx, start_idx + len(normalized_patch)))
            
            # Concatenate all patches
            x_batch = torch.stack(all_x_batch, dim=0).unsqueeze(1)  # (total_traces_all_patches, 1, time_PS)
            sx_b = torch.stack(all_sx_batch, dim=0)
            sy_b = torch.stack(all_sy_batch, dim=0)
            rx_b = torch.stack(all_rx_batch, dim=0)
            ry_b = torch.stack(all_ry_batch, dim=0)
            
            # Batch processing
            P = x_batch.shape[0]
            MAX_BATCH_SIZE = 15 * num_gpus
            sampled_chunks = []
            num_chunks = (P + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            print(f"  Processing {P} traces in {num_chunks} chunks...")
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.inference_mode():
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * MAX_BATCH_SIZE
                    end = min(start + MAX_BATCH_SIZE, P)
                    
                    x_chunk = x_batch[start:end]
                    rx_chunk = rx_b[start:end]
                    ry_chunk = ry_b[start:end]
                    sx_chunk = sx_b[start:end]
                    sy_chunk = sy_b[start:end]
                    
                    y_chunk = parallel_sampler(
                        x_chunk,
                        rx_chunk, ry_chunk, sx_chunk, sy_chunk,
                        None
                    )
                    
                    # Denormalize (need to find which patch each trace belongs to)
                    y_chunk_list = []
                    for i in range(y_chunk.shape[0]):
                        global_idx = start + i
                        # Find which patch this trace belongs to
                        patch_idx = None
                        for p_idx, (p_start, p_end) in enumerate(patch_boundaries):
                            if p_start <= global_idx < p_end:
                                patch_idx = p_idx
                                local_idx = global_idx - p_start
                                break
                        
                        if patch_idx is None:
                            raise ValueError(f"Could not find patch for trace {global_idx}")
                        
                        y_patch = y_chunk[i]  # Shape: (1, time_PS) or (time_PS,)
                        # Remove channel dimension if present
                        if y_patch.ndim > 1:
                            y_patch = y_patch.squeeze(0)  # Remove first dimension if it's 1
                        thres = all_thres_list[patch_idx]
                        y_denorm = (y_patch - y_patch.mean()) * thres
                        y_chunk_list.append(y_denorm)
                    
                    sampled_chunks.append(torch.stack(y_chunk_list, dim=0))
            
            y = torch.cat(sampled_chunks, dim=0)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_stats.add_sample_time(elapsed, P)
            print(f"  Sampled {P} traces in {elapsed:.2f}s")
            
            # Restore each patch and concatenate along time dimension
            y_np = y.cpu().numpy()  # Shape: (total_traces_all_patches, time_PS) or (total_traces_all_patches, 1, time_PS)
            
            # Remove channel dimension if present
            if y_np.ndim == 3:
                # Shape is (total_traces, 1, time_PS), squeeze to (total_traces, time_PS)
                y_np = y_np.reshape(-1,time_PS)
            elif y_np.ndim != 2:
                raise ValueError(f"Unexpected y_np shape: {y_np.shape}, expected 2D or 3D")
            
            # Get all coordinates for sorting (convert from tensor to numpy)
            all_sx_np = sx_b.cpu().numpy()  # (total_traces_all_patches,)
            all_sy_np = sy_b.cpu().numpy()
            all_rx_np = rx_b.cpu().numpy()
            all_ry_np = ry_b.cpu().numpy()
            
            # Sort by shot coordinates first (sx, sy), then by receiver coordinates (rx, ry)
            # Use lexsort: sorts by last key first, then by previous keys
            sort_indices = np.lexsort((all_ry_np, all_rx_np, all_sy_np, all_sx_np))
            
            # Apply sorting to sampled data and coordinates
            y_np_sorted = y_np[sort_indices].reshape(-1,time_PS)  # (total_traces_all_patches, time_PS)
            # Update patch boundaries after sorting
            # We need to remap the boundaries since indices have changed
            # For now, we'll process patches in the sorted order
            # But we need to track which sorted traces belong to which original patch
            print(f"  Sorted {len(sort_indices)} traces by coordinates (sx, sy, rx, ry)")
            
            restored_patches = []
            
            for patch_idx, (patch_start, patch_end) in enumerate(patch_boundaries):
                # Find the sorted indices that correspond to this original patch
                # Original patch indices are [patch_start, patch_end)
                original_patch_indices = np.arange(patch_start, patch_end)
                
                # Find where these original indices ended up after sorting
                # sort_indices maps: sorted_position -> original_position
                # i.e., sort_indices[i] = j means sorted position i corresponds to original position j
                # We need inverse: original_position -> sorted_position
                # Create inverse mapping: for each original position, find its sorted position
                inverse_sort = np.argsort(sort_indices.reshape(-1))  # This gives: original_position -> sorted_position
                
                # Get sorted positions for this patch's original indices
                sorted_patch_indices = inverse_sort[original_patch_indices]
                
                # Extract sorted traces for this patch
                # y_np_sorted shape: (total_traces_all_patches, time_PS)
                # sorted_patch_indices: indices in sorted array that belong to this patch
                patch_y_sorted = y_np_sorted[sorted_patch_indices]  # Shape: (n_traces_in_patch, time_PS)
                missing_info = all_missing_info[patch_idx]
                
                # Ensure patch_y_sorted is 2D: (total_traces_in_patch, time_PS)
                if patch_y_sorted.ndim != 2:
                    raise ValueError(f"patch_y_sorted shape should be 2D, got {patch_y_sorted.shape}")
                
                # Restore to original order within the patch
                # Note: original_indices are relative to the concatenated patch (KNN + missing)
                # After sorting, we need to restore the order
                restored_missing_patch = inpainter.restore_patch_order(
                    patch_y_sorted,
                    missing_info['original_indices'],
                    missing_info['missing_start_idx_original'],
                    missing_info['missing_end_idx_original']
                )
                restored_patches.append({
                    'data': restored_missing_patch,
                    'time_start': missing_info['time_start'],
                    'trace_indices': missing_info['trace_indices']
                })
            
            # Sort patches by time_start and concatenate
            #restored_patches.sort(key=lambda x: x['trace_indices'])
            #print(len(restored_patches),len(restored_patches[0]['data']))
            # Reconstruct full line by concatenating patches along time dimension
            # Each patch is (PS, time_PS), we need to concatenate them along time
            full_line_data = None
            full_trace_indices = None
            
            for patch_info in restored_patches:
                patch_data = patch_info['data']  # (PS, time_PS)
                trace_indices = patch_info['trace_indices']
                
                if full_line_data is None:
                    full_line_data = patch_data
                    full_trace_indices = trace_indices
                else:
                    # Concatenate along time dimension
                    full_line_data = np.concatenate([full_line_data, patch_data], axis=0)
                    full_trace_indices = np.concatenate([full_trace_indices, trace_indices], axis=0)
                    # Trace indices should be the same for all patches of the same line
                    #assert np.array_equal(full_trace_indices, trace_indices), f"Trace indices should be consistent, but got {full_trace_indices} and {trace_indices}"
            print('full_line_data shape:',full_line_data.shape)
            print('full_trace_indices shape:',full_trace_indices.shape)
            # Load label data for comparison
            label_file = label_cleaned_dir / f"label_cleaned_data_recl_{recl}_recn_{recn}.npy"
            if label_file.exists():
                label_data = np.load(label_file)  # (n_traces, n_samples)
                
                # Extract label data for this missing line
                if full_trace_indices is not None and len(full_trace_indices) > 0:
                    label_line_data = label_data[full_trace_indices, :]
                    # Extract same time range
                    if full_line_data is not None:
                        time_end = full_line_data.shape[1]
                        label_line_data = label_line_data[:, :time_end]
                else:
                    label_line_data = None
            else:
                print(f"  [WARN] Label file not found: {label_file}")
                label_line_data = None
            
            # Save results
            np.save(f"{dir_path}/missing_line_sampled_recl_{recl}_recn_{recn}_line_{line_id}.npy",
                   full_line_data)
            if label_line_data is not None:
                np.save(f"{dir_path}/missing_line_label_recl_{recl}_recn_{recn}_line_{line_id}.npy",
                       label_line_data)
            
            # Visualize comparison
            if full_line_data is not None and label_line_data is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Calculate unified color range
                vmax = max(full_line_data.std(), label_line_data.std())
                
                # Plot 1: Sampled result
                im0 = axes[0].pcolormesh(full_line_data.T, cmap=red_black_cmap,
                                        vmin=-vmax, vmax=vmax)
                axes[0].set_title(f'Receiver ({recl}, {recn}) Line {line_id}\nSampled Result', 
                                 fontsize=12, fontweight='bold')
                axes[0].set_xlabel('Trace Index')
                axes[0].set_ylabel('Time Sample')
                axes[0].invert_yaxis()
                plt.colorbar(im0, ax=axes[0], shrink=0.8)
     
                # Plot 2: Label (ground truth)
                im1 = axes[1].pcolormesh(label_line_data.T, cmap=red_black_cmap,
                                        vmin=-vmax, vmax=vmax)
                axes[1].set_title(f'Receiver ({recl}, {recn}) Line {line_id}\nGround Truth', 
                                 fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Trace Index')
                axes[1].set_ylabel('Time Sample')
                axes[1].invert_yaxis()
                plt.colorbar(im1, ax=axes[1], shrink=0.8)
                
                
                plt.tight_layout()
                save_path = f"{dir_path}/missing_line_comparison_recl_{recl}_recn_{recn}_line_{line_id}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved comparison figure to: {save_path}")
            
            print(f"  Completed line {line_id}")
           
            
    time_stats.print_summary()
        
else:
    raise ValueError("Only 'parallel' mode supported in this snippet.")