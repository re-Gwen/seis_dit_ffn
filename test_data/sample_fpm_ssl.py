"""
Flow Matching Model (FPM) 采样脚本 - SSL 数据集版本
基于 sample_ddpm_ssl.py 对齐，支持 C3NA 和 xbfy 数据集

用法:
    python sample_fpm_ssl.py <sample_mode> <data_type> <sample_num> [其他参数...]

参数说明:
    sample_mode: interp, sample, block 等
    data_type: parallel, parallel_xbfy, C3NA 等
    sample_num: 每个patch的采样数量
    PS: Patch Size (空间维度)
    SD: Stride (空间维度)
    time_PS: Time Patch Size (时间维度)
    time_SD: Time Stride (时间维度)
    missing_ratio: 缺失比例 (0.0-1.0)
    missing_mode: 缺失模式 (random, block 等)
    steps: ODE/SDE 采样步数
    cuda_devices: CUDA可见设备 (如: 0,1,2,3)
    output_prefix: 输出目录前缀
    device_id: 使用的GPU设备ID
    model_path: 模型检查点路径
    model_type: 模型类型 (vit_bottleneck, trace_axis)

示例:
    python sample_fpm_ssl.py interp parallel 1 64 64 624 624 0.5 random 50 0,1,2,3 ./FPM_sample 1 /path/to/model.pth vit_bottleneck
"""

import seisdit_trace_axis
import seisdit_vit_bottleneck
import torch
import FPM  # Flow Matching Model
import pathlib
from typing import BinaryIO, List, Union
from matplotlib import pyplot as plt
import os
import tqdm
import datetime
import json
import sys
import random
import numpy as np
import patched_dataset5d
import time
from self_datasets.segy_ssl_dataset import SegySSLConfig

# 默认参数，可通过命令行参数覆盖
PS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
SD = int(sys.argv[5]) if len(sys.argv) > 5 else 64
time_PS = int(sys.argv[6]) if len(sys.argv) > 6 else 624
time_SD = int(sys.argv[7]) if len(sys.argv) > 7 else 624

MISSING_RATIO = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
MISSING_MODE = sys.argv[9] if len(sys.argv) > 9 else 'random'

STEP = int(sys.argv[10]) if len(sys.argv) > 10 else 50
CUDA_DEVICES = sys.argv[11] if len(sys.argv) > 11 else '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES


def snr(signal, signal_predict):
    """计算信噪比 (SNR)"""
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum((signal - signal_predict) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


from matplotlib.colors import LinearSegmentedColormap

colors = [
    (0.0, "black"),  # 最小值（负值）
    (0.5, "white"),  # 中间值（0）
    (1.0, "red"),    # 最大值（正值）
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
    device=None,  # 后面设置
    sup_mode="all",
    use_coherence=False,
    sigma_obs=0.001,
    use_bayesian=False,
    sampling_method="ode",
    ode_sampling_method="euler",
    ode_num_steps=STEP,
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
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
})

# 根据数据类型选择保存路径
if "C3NA" in sys.argv[2] or "c3na" in sys.argv[2]:
    base_dir = './C3NA_sample_fpm'
elif "xbfy" in sys.argv[2] or "XBFY" in sys.argv[2]:
    base_dir = './xbfy_sample_fpm'
else:
    base_dir = './sample_fpm'

# 支持自定义输出目录前缀
OUTPUT_PREFIX = sys.argv[12] if len(sys.argv) > 12 else base_dir

plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

DEVICE_ID = int(sys.argv[13]) if len(sys.argv) > 13 else 0
device = f"cuda:{DEVICE_ID}"
image_channels = 2 if sys.argv[1] in ["interp", "raw", "C3NA", "patched"] else 1

# 模型路径和类型可通过命令行参数指定
MODEL_PATH = sys.argv[14] if len(sys.argv) > 14 else "/NAS/czt/mount/Seis_DiT/resultsFPM/dit_datatype_c3NA_ssl_vit_bottleneck_Linear_velocity/checkpoints/model-20.pth"
MODEL_TYPE = sys.argv[15] if len(sys.argv) > 15 else "vit_bottleneck"  # vit_bottleneck, trace_axis

# 根据模型类型创建模型
if MODEL_TYPE == "vit_bottleneck":
    model = seisdit_vit_bottleneck.SeisDiTRope(
        image_channels=image_channels,
        d_model=384,
        num_layers=8,
        pe_type=pe_type,
        n_channels=32,
    )
elif MODEL_TYPE == "trace_axis":
    model = seisdit_trace_axis.SeisDiTRopeV2(
        image_channels=image_channels,
        d_model=384,
        num_layers=8,
        pe_type=pe_type,
        n_channels=32,
        rope_p_scale= {'sx': 6.5, 'sy': 95.5, 'rx': 29.5, 'ry': 102.0}
    )
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model_path = MODEL_PATH

state_dict = torch.load(
    model_path,
    map_location="cpu",
)["model"]

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 创建输出目录
dir_path = f'{OUTPUT_PREFIX}/{datetime.datetime.now().strftime("%Y%m%d")}_{sys.argv[1]}_{sys.argv[2]}_{PS}_{MISSING_MODE}_{MISSING_RATIO}_{MODEL_TYPE}_fpm'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


same_seeds(515)

# 创建 Flow Matching Model
fpm_kwargs['device'] = device
fpm = FPM.FlowMatchingModel(
    model=model,
    trace_num=fpm_kwargs["trace_num"],
    time_steps=fpm_kwargs["time_num"],
    path_type=fpm_kwargs["path_type"],
    prediction=fpm_kwargs["prediction"],
    loss_weight=fpm_kwargs["loss_weight"],
    train_eps=fpm_kwargs["train_eps"],
    sample_eps=fpm_kwargs["sample_eps"],
    sample_num=fpm_kwargs["sample_num"],
    device=fpm_kwargs["device"],
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


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    norm_=True,
    diff: bool = False,
) -> None:
    print(tensor.shape)
    assert len(tensor.shape) == 3
    tensor = tensor[0, :, :].detach().cpu()
    tensor = tensor.numpy()
    plt.figure(figsize=(6, 6))
    if diff:
        # 差异图：使用对称的颜色范围，基于数据的std
        vmax = max(np.abs(tensor).max(), 1e-6)
        vmin = -vmax
        plt.pcolor(tensor.T, cmap=red_black_cmap, vmin=-1, vmax=1)
        plt.title('Difference (GT - Predicted)')
    else:
        # 普通图像：归一化后的数据范围在[-1, 1]
        plt.pcolor(tensor.T, cmap=red_black_cmap, vmin=-1, vmax=1)
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel("Trace Number")
    plt.ylabel("Time(ms)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fp, dpi=600)
    plt.close()


# ==================== C3NA 数据集 ====================
if "C3NA" in sys.argv[2] or "c3na" in sys.argv[2]:
    contin_missing = True if 'block' in sys.argv[1] else False
    print(f"contin_missing: {contin_missing}")
    cfg_c3NA_val = {
        'max_traces': 1000000,
        'root_List': ['/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_3601-4781.sgy'],
        'time_step': 512,
        'missing_rate_list': [0.7],
        'contin_missing': contin_missing,
        'ps': PS,
    }
    dataset_test = patched_dataset5d.SEG_C3NA_patched(**cfg_c3NA_val)
    dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4
    )
    results = {}
    ps = PS
    sd = SD
    for iter, (data, data_mask, rx, ry, sx, sy, _, _) in tqdm.tqdm(enumerate(dl)):
        print(f"Processing sample {iter+1}/{len(dl)}")
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data)
        else:
            data_tensor = data
        if data_tensor.ndim == 3:
            data_tensor = data_tensor.unsqueeze(1)
        if isinstance(data_mask, np.ndarray):
            data_mask_t = torch.tensor(data_mask)
        else:
            data_mask_t = data_mask
        if data_mask_t.ndim == 3:
            data_mask_t = data_mask_t.unsqueeze(1)

        rx_tensor = rx if isinstance(rx, torch.Tensor) else torch.tensor(rx)
        ry_tensor = ry if isinstance(ry, torch.Tensor) else torch.tensor(ry)
        sx_tensor = sx if isinstance(sx, torch.Tensor) else torch.tensor(sx)
        sy_tensor = sy if isinstance(sy, torch.Tensor) else torch.tensor(sy)

        data_tensor = data_tensor.to(device)
        data_tensor -= data_tensor.mean()
        B, C, H, W = data_tensor.shape
        data_mask_t = data_mask_t.to(device)
        rx_tensor = rx_tensor.to(device)
        ry_tensor = ry_tensor.to(device)
        sx_tensor = sx_tensor.to(device)
        sy_tensor = sy_tensor.to(device)
        condL = (rx_tensor, ry_tensor, sx_tensor, sy_tensor)

        # 保存原始与掩码数据
        save_image(
            data_tensor[0],
            fp=str(f"{dir_path}/ori_{iter}.png"),
            norm_=False,
        )
        save_image(
            data_mask_t[0],
            fp=str(f"{dir_path}/masked_{iter}.png"),
            norm_=False,
        )
        np.save(f"{dir_path}/data_{iter}.npy", data_tensor.detach().cpu().numpy())
        np.save(f"{dir_path}/mask_{iter}.npy", data_mask_t.detach().cpu().numpy())
        patches, rx_s, ry_s, sx_s, sy_s, _ = patched_dataset5d.gen_patches_torch(
            data_mask_t, condL, (ps, time_PS), (sd, time_SD)
        )
        with torch.inference_mode():
            sampleL = []
            for patch, rx_patch, ry_patch, sx_patch, sy_patch in zip(
                patches, rx_s, ry_s, sx_s, sy_s
            ):
                patch = patch.repeat(int(sys.argv[3]), 1, 1, 1).to(device)
                condL_sample_patches = (rx_patch, ry_patch, sx_patch, sy_patch)
                fpm.sample_num = patch.shape[0]
                sampled_img = fpm.sample(
                    condL=condL_sample_patches,
                    x_cond=patch,
                )
                sampleL.append(sampled_img - sampled_img.mean())
            sampled_imgs = patched_dataset5d.reconstruct_from_patches_torch(
                sampleL, 1, (H, W), (ps, time_PS), (sd, time_SD)
            )
        np.save(
            f"{dir_path}/sampled_imgs_{iter}.npy", sampled_imgs.detach().cpu().numpy()
        )
        for i in range(len(sampled_imgs)):
            sample = sampled_imgs[i]
            save_image(
                sample,
                fp=str(f"{dir_path}/sample_{iter}_{i}.png"),
            )
            if i >= 10:
                break
        if iter >= 10:
            break

# ==================== parallel 模式 - C3NA SSL 数据集 ====================
elif sys.argv[2] == "parallel":
    MAX_BATCH_SIZE = 10
    # 使用与训练时相同的数据集配置
    _, val_dataset = SegySSLConfig.create_C3NA_datasets(
        domain='receiver',
        spatial_window=(PS,),
        train_ranges=[(2401, 3600)],
        val_ranges=[(2401, 3600)],
        val_split='test',
        missing_mode=MISSING_MODE,
        missing_ratio=MISSING_RATIO,
    )
    dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    results = {}
    ps = PS
    sd = SD
    
    for iter, batch in tqdm.tqdm(enumerate(dl)):
        print(f"Processing sample {iter+1}/{len(dl)}")
        # 从字典中提取数据（与训练时对齐）
        data = batch['x_gt']
        data_mask = batch['x_obs']
        rx = batch['gx']
        ry = batch['gy']
        sx = batch['sx']
        sy = batch['sy']
        time_axis = batch.get('time_axis', None)

        # 转换为tensor并处理维度
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data)
        else:
            data_tensor = data
        if data_tensor.ndim == 2:
            data_tensor = data_tensor.unsqueeze(0)
        if data_tensor.ndim == 3:
            data_tensor = data_tensor.unsqueeze(1)

        if isinstance(data_mask, np.ndarray):
            data_mask_t = torch.tensor(data_mask)
        else:
            data_mask_t = data_mask
        if data_mask_t.ndim == 2:
            data_mask_t = data_mask_t.unsqueeze(0)
        if data_mask_t.ndim == 3:
            data_mask_t = data_mask_t.unsqueeze(1)

        rx_tensor = rx if isinstance(rx, torch.Tensor) else torch.tensor(rx)
        ry_tensor = ry if isinstance(ry, torch.Tensor) else torch.tensor(ry)
        sx_tensor = sx if isinstance(sx, torch.Tensor) else torch.tensor(sx)
        sy_tensor = sy if isinstance(sy, torch.Tensor) else torch.tensor(sy)

        data_tensor = data_tensor.to(device)
        B, C, H, W = data_tensor.shape
        data_mask_t = data_mask_t.to(device)
        rx_tensor = rx_tensor.to(device)
        ry_tensor = ry_tensor.to(device)
        sx_tensor = sx_tensor.to(device)
        sy_tensor = sy_tensor.to(device)
        condL = (rx_tensor, ry_tensor, sx_tensor, sy_tensor)

        save_image(
            data_tensor[0],
            fp=str(f"{dir_path}/ori_{iter}.png"),
            norm_=False,
        )
        save_image(
            data_mask_t[0],
            fp=str(f"{dir_path}/masked_{iter}.png"),
            norm_=False,
        )
        np.save(f"{dir_path}/data_{iter}.npy", data_tensor.detach().cpu().numpy())
        np.save(f"{dir_path}/mask_{iter}.npy", data_mask_t.detach().cpu().numpy())
        patches, rx_s, ry_s, sx_s, sy_s, t_idx_s = patched_dataset5d.gen_patches_torch(
            data_mask_t, condL, (ps, time_PS), (sd, time_SD), return_t=True
        )
        
        with torch.inference_mode():
            K = 1
            P = len(patches)
            fpm.sample_num = P * K
            thres_list = []
            normalized_patches = []
            for p in patches:
                thres = np.percentile(np.abs(p.cpu().numpy()), 99.5)
                if thres == 0:
                    thres = 1e-6
                thres_list.append(thres)
                p_clamped = torch.clamp(p, -thres, thres)
                p_normalized = (p_clamped / thres).to(device)
                normalized_patches.append(p_normalized)
            x_batch = torch.cat([p.to(device) for p in normalized_patches], dim=0)
            rx_b = torch.cat([r.to(device) for r in rx_s], dim=0)
            ry_b = torch.cat([r.to(device) for r in ry_s], dim=0)
            sx_b = torch.cat([s.to(device) for s in sx_s], dim=0)
            sy_b = torch.cat([y.to(device) for y in sy_s], dim=0)
            if t_idx_s is not None:
                t_idx_b = t_idx_s.to(device)
            else:
                t_idx_b = None
            x_batch = x_batch.unsqueeze(1)
            print("x_batch:", x_batch.shape)
            print("rx_b:", rx_b.shape)
            print(f"Total patches: {P}, Max batch size: {MAX_BATCH_SIZE}")

            if K > 1:
                x_batch = x_batch.repeat_interleave(K, dim=0)
                rx_b = rx_b.repeat_interleave(K, dim=0)
                ry_b = ry_b.repeat_interleave(K, dim=0)
                sx_b = sx_b.repeat_interleave(K, dim=0)
                sy_b = sy_b.repeat_interleave(K, dim=0)
                if t_idx_b is not None:
                    t_idx_b = t_idx_b.repeat_interleave(K, dim=0)

            # 分块处理
            if P <= MAX_BATCH_SIZE:
                fpm.sample_num = P * K
                y = fpm.sample(
                    condL=(rx_b, ry_b, sx_b, sy_b), x_cond=x_batch, time_axis=t_idx_b
                )
                if K > 1:
                    y = y.view(P, K, *y.shape[1:]).mean(dim=1)
                y_list = []
                for i in range(P):
                    y_patch = y[i]
                    thres = thres_list[i]
                    y_denorm = (y_patch - y_patch.mean()) * thres
                    y_list.append(y_denorm)
                y = torch.stack(y_list, dim=0)
            else:
                sampled_chunks = []
                num_chunks = (P + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
                print(f"Splitting into {num_chunks} chunks...")

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * MAX_BATCH_SIZE
                    end_idx = min(start_idx + MAX_BATCH_SIZE, P)
                    chunk_size = end_idx - start_idx

                    print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (patches {start_idx}-{end_idx-1})")
                    x_chunk = x_batch[start_idx:end_idx]
                    rx_chunk = rx_b[start_idx:end_idx]
                    ry_chunk = ry_b[start_idx:end_idx]
                    sx_chunk = sx_b[start_idx:end_idx]
                    sy_chunk = sy_b[start_idx:end_idx]

                    t_idx_chunk = None
                    if t_idx_b is not None:
                        t_idx_chunk = t_idx_b[start_idx:end_idx]
                    fpm.sample_num = chunk_size * K

                    y_chunk = fpm.sample(
                        condL=(rx_chunk, ry_chunk, sx_chunk, sy_chunk),
                        x_cond=x_chunk,
                        time_axis=t_idx_chunk
                    )
                    y_chunk_list = []
                    if K > 1:
                        y_chunk = y_chunk.view(chunk_size, K, *y_chunk.shape[1:]).mean(dim=1)
                    for i in range(chunk_size):
                        y_patch = y_chunk[i]
                        thres = thres_list[start_idx + i]
                        y_denorm = (y_patch - y_patch.mean()) * thres
                        y_chunk_list.append(y_denorm)
                    y_chunk = torch.stack(y_chunk_list, dim=0)
                    sampled_chunks.append(y_chunk)

                y = torch.cat(sampled_chunks, dim=0)

            sampleL = [y[i].unsqueeze(0) for i in range(P)]
            sampled_imgs = patched_dataset5d.reconstruct_from_patches_torch(
                sampleL, 1, (H, W), (ps, time_PS), (sd, time_SD)
            )

        # 转换为numpy进行后处理
        sampled_imgs_np = sampled_imgs.detach().cpu().numpy()
        data_gt_np = data_tensor.detach().cpu().numpy()
        data_mask_np = data_mask_t.detach().cpu().numpy()

        unified_thres = np.percentile(np.abs(data_mask_np[0, 0]), 99.5)
        if unified_thres == 0:
            unified_thres = 1e-6
        data_gt_normalized = np.clip(data_gt_np[0, 0], -unified_thres, unified_thres) / unified_thres
        data_mask_normalized = np.clip(data_mask_np[0, 0], -unified_thres, unified_thres) / unified_thres

        np.save(f"{dir_path}/data_{iter}.npy", data_gt_np)
        np.save(f"{dir_path}/mask_{iter}.npy", data_mask_np)
        np.save(f"{dir_path}/sampled_imgs_{iter}.npy", sampled_imgs_np)
        np.save(f"{dir_path}/unified_thres_{iter}.npy", unified_thres)

        snr_results = []
        snr_mask_results = []
        for i in range(len(sampled_imgs)):
            sample_np = sampled_imgs_np[i, 0]

            sample_clamped = np.clip(sample_np, -unified_thres, unified_thres)
            sample_normalized = (sample_clamped / unified_thres)
            snr_value = snr(data_gt_normalized, sample_normalized)
            snr_mask_value = snr(data_gt_normalized, data_mask_normalized)
            snr_results.append(snr_value)
            snr_mask_results.append(snr_mask_value)
            print(f"Sample {iter}_{i} SNR: {snr_value:.2f} dB")
            print(f"Sample {iter}_{i} SNR_mask: {snr_mask_value:.2f} dB")

            diff = data_gt_normalized - sample_normalized

            sample_tensor = torch.from_numpy(sample_normalized).unsqueeze(0).unsqueeze(0)
            save_image(
                sample_tensor[0],
                fp=str(f"{dir_path}/sample_{iter}_{i}.png"),
            )

            diff_tensor = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0)
            save_image(
                diff_tensor[0],
                fp=str(f"{dir_path}/diff_{iter}_{i}.png"),
                diff=True,
            )

            if i == 0:
                gt_tensor = torch.from_numpy(data_gt_normalized).unsqueeze(0).unsqueeze(0)
                save_image(
                    gt_tensor[0],
                    fp=str(f"{dir_path}/gt_normalized_{iter}.png"),
                )
                mask_tensor = torch.from_numpy(data_mask_normalized).unsqueeze(0).unsqueeze(0)
                save_image(
                    mask_tensor[0],
                    fp=str(f"{dir_path}/mask_normalized_{iter}.png"),
                )

        snr_values_python = [float(x) for x in snr_results] if snr_results else []
        snr_mask_values_python = [float(x) for x in snr_mask_results] if snr_mask_results else []
        snr_dict = {
            'sample_indices': list(range(len(sampled_imgs))),
            'snr_values': snr_values_python,
            'mean_snr': float(np.mean(snr_results)) if snr_results else 0.0,
            'std_snr': float(np.std(snr_results)) if snr_results else 0.0,
            'mean_snr_mask': float(np.mean(snr_mask_results)) if snr_mask_results else 0.0,
            'std_snr_mask': float(np.std(snr_mask_results)) if snr_mask_results else 0.0,
        }
        with open(f"{dir_path}/snr_{iter}.json", 'w') as f:
            json.dump(snr_dict, f, indent=4)
        print(f"Sample {iter} - Mean SNR: {snr_dict['mean_snr']:.2f} dB, Std SNR: {snr_dict['std_snr']:.2f} dB")

        if iter >= 100000:
            break

# ==================== parallel_xbfy 模式 - xbfy 数据集 ====================
elif sys.argv[2] == "parallel_xbfy":
    MAX_BATCH_SIZE = 10
    # 使用与训练时相同的xbfy数据集配置
    val_dataset = SegySSLConfig.create_xbfy_datasets(
        split='test',
        domain='receiver',
        spatial_window=(PS,),
        missing_mode=MISSING_MODE,
        missing_ratio=MISSING_RATIO,
        time_skip=14,
        time_bins=2,
    )
    dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    results = {}
    ps = PS
    sd = SD
    
    for iter, batch in tqdm.tqdm(enumerate(dl)):
        print(f"Processing sample {iter+1}/{len(dl)}")
        data = batch['x_gt']
        data_mask = batch['x_obs']
        rx = batch['gx']
        ry = batch['gy']
        sx = batch['sx']
        sy = batch['sy']
        time_axis = batch.get('time_axis', None)

        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data)
        else:
            data_tensor = data
        if data_tensor.ndim == 2:
            data_tensor = data_tensor.unsqueeze(0)
        if data_tensor.ndim == 3:
            data_tensor = data_tensor.unsqueeze(1)

        if isinstance(data_mask, np.ndarray):
            data_mask_t = torch.tensor(data_mask)
        else:
            data_mask_t = data_mask
        if data_mask_t.ndim == 2:
            data_mask_t = data_mask_t.unsqueeze(0)
        if data_mask_t.ndim == 3:
            data_mask_t = data_mask_t.unsqueeze(1)

        rx_tensor = rx if isinstance(rx, torch.Tensor) else torch.tensor(rx)
        ry_tensor = ry if isinstance(ry, torch.Tensor) else torch.tensor(ry)
        sx_tensor = sx if isinstance(sx, torch.Tensor) else torch.tensor(sx)
        sy_tensor = sy if isinstance(sy, torch.Tensor) else torch.tensor(sy)

        data_tensor = data_tensor.to(device)
        B, C, H, W = data_tensor.shape
        data_mask_t = data_mask_t.to(device)
        rx_tensor = rx_tensor.to(device)
        ry_tensor = ry_tensor.to(device)
        sx_tensor = sx_tensor.to(device)
        sy_tensor = sy_tensor.to(device)
        condL = (rx_tensor, ry_tensor, sx_tensor, sy_tensor)

        save_image(
            data_tensor[0],
            fp=str(f"{dir_path}/ori_{iter}.png"),
            norm_=False,
        )
        save_image(
            data_mask_t[0],
            fp=str(f"{dir_path}/masked_{iter}.png"),
            norm_=False,
        )
        np.save(f"{dir_path}/data_{iter}.npy", data_tensor.detach().cpu().numpy())
        np.save(f"{dir_path}/mask_{iter}.npy", data_mask_t.detach().cpu().numpy())
        patches, rx_s, ry_s, sx_s, sy_s, t_idx_s = patched_dataset5d.gen_patches_torch(
            data_mask_t, condL, (ps, time_PS), (sd, time_SD), return_t=True
        )
        
        with torch.inference_mode():
            K = 1
            P = len(patches)
            fpm.sample_num = P * K
            thres_list = []
            normalized_patches = []
            for p in patches:
                thres = np.percentile(np.abs(p.cpu().numpy()), 99.5)
                if thres == 0:
                    thres = 1e-6
                thres_list.append(thres)
                p_clamped = torch.clamp(p, -thres, thres)
                p_normalized = (p_clamped / thres).to(device)
                normalized_patches.append(p_normalized)
            x_batch = torch.cat([p.to(device) for p in normalized_patches], dim=0)
            rx_b = torch.cat([r.to(device) for r in rx_s], dim=0)
            ry_b = torch.cat([r.to(device) for r in ry_s], dim=0)
            sx_b = torch.cat([s.to(device) for s in sx_s], dim=0)
            sy_b = torch.cat([y.to(device) for y in sy_s], dim=0)
            if t_idx_s is not None:
                t_idx_b = t_idx_s.to(device)
            else:
                t_idx_b = None
            x_batch = x_batch.unsqueeze(1)
            print("x_batch:", x_batch.shape)
            print("rx_b:", rx_b.shape)
            print(f"Total patches: {P}, Max batch size: {MAX_BATCH_SIZE}")

            if K > 1:
                x_batch = x_batch.repeat_interleave(K, dim=0)
                rx_b = rx_b.repeat_interleave(K, dim=0)
                ry_b = ry_b.repeat_interleave(K, dim=0)
                sx_b = sx_b.repeat_interleave(K, dim=0)
                sy_b = sy_b.repeat_interleave(K, dim=0)
                if t_idx_b is not None:
                    t_idx_b = t_idx_b.repeat_interleave(K, dim=0)

            if P <= MAX_BATCH_SIZE:
                fpm.sample_num = P * K
                y = fpm.sample(
                    condL=(rx_b, ry_b, sx_b, sy_b), x_cond=x_batch, time_axis=t_idx_b
                )
                if K > 1:
                    y = y.view(P, K, *y.shape[1:]).mean(dim=1)
                y_list = []
                for i in range(P):
                    y_patch = y[i]
                    thres = thres_list[i]
                    y_denorm = (y_patch - y_patch.mean()) * thres
                    y_list.append(y_denorm)
                y = torch.stack(y_list, dim=0)
            else:
                sampled_chunks = []
                num_chunks = (P + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
                print(f"Splitting into {num_chunks} chunks...")

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * MAX_BATCH_SIZE
                    end_idx = min(start_idx + MAX_BATCH_SIZE, P)
                    chunk_size = end_idx - start_idx

                    print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (patches {start_idx}-{end_idx-1})")
                    x_chunk = x_batch[start_idx:end_idx]
                    rx_chunk = rx_b[start_idx:end_idx]
                    ry_chunk = ry_b[start_idx:end_idx]
                    sx_chunk = sx_b[start_idx:end_idx]
                    sy_chunk = sy_b[start_idx:end_idx]

                    t_idx_chunk = None
                    if t_idx_b is not None:
                        t_idx_chunk = t_idx_b[start_idx:end_idx]
                    fpm.sample_num = chunk_size * K

                    y_chunk = fpm.sample(
                        condL=(rx_chunk, ry_chunk, sx_chunk, sy_chunk),
                        x_cond=x_chunk,
                        time_axis=t_idx_chunk
                    )
                    y_chunk_list = []
                    if K > 1:
                        y_chunk = y_chunk.view(chunk_size, K, *y_chunk.shape[1:]).mean(dim=1)
                    for i in range(chunk_size):
                        y_patch = y_chunk[i]
                        thres = thres_list[start_idx + i]
                        y_denorm = (y_patch - y_patch.mean()) * thres
                        y_chunk_list.append(y_denorm)
                    y_chunk = torch.stack(y_chunk_list, dim=0)
                    sampled_chunks.append(y_chunk)

                y = torch.cat(sampled_chunks, dim=0)

            sampleL = [y[i].unsqueeze(0) for i in range(P)]
            sampled_imgs = patched_dataset5d.reconstruct_from_patches_torch(
                sampleL, 1, (H, W), (ps, time_PS), (sd, time_SD)
            )

        sampled_imgs_np = sampled_imgs.detach().cpu().numpy()
        data_gt_np = data_tensor.detach().cpu().numpy()
        data_mask_np = data_mask_t.detach().cpu().numpy()

        unified_thres = np.percentile(np.abs(data_mask_np[0, 0]), 99.5)
        if unified_thres == 0:
            unified_thres = 1e-6
        data_gt_normalized = np.clip(data_gt_np[0, 0], -unified_thres, unified_thres) / unified_thres
        data_mask_normalized = np.clip(data_mask_np[0, 0], -unified_thres, unified_thres) / unified_thres

        np.save(f"{dir_path}/data_{iter}.npy", data_gt_np)
        np.save(f"{dir_path}/mask_{iter}.npy", data_mask_np)
        np.save(f"{dir_path}/sampled_imgs_{iter}.npy", sampled_imgs_np)
        np.save(f"{dir_path}/unified_thres_{iter}.npy", unified_thres)

        snr_results = []
        snr_mask_results = []
        for i in range(len(sampled_imgs)):
            sample_np = sampled_imgs_np[i, 0]

            sample_clamped = np.clip(sample_np, -unified_thres, unified_thres)
            sample_normalized = (sample_clamped / unified_thres)
            snr_value = snr(data_gt_normalized, sample_normalized)
            snr_mask_value = snr(data_gt_normalized, data_mask_normalized)
            snr_results.append(snr_value)
            snr_mask_results.append(snr_mask_value)
            print(f"Sample {iter}_{i} SNR: {snr_value:.2f} dB")
            print(f"Sample {iter}_{i} SNR_mask: {snr_mask_value:.2f} dB")

            diff = data_gt_normalized - sample_normalized

            sample_tensor = torch.from_numpy(sample_normalized).unsqueeze(0).unsqueeze(0)
            save_image(
                sample_tensor[0],
                fp=str(f"{dir_path}/sample_{iter}_{i}.png"),
            )

            diff_tensor = torch.from_numpy(diff).unsqueeze(0).unsqueeze(0)
            save_image(
                diff_tensor[0],
                fp=str(f"{dir_path}/diff_{iter}_{i}.png"),
                diff=True,
            )

            if i == 0:
                gt_tensor = torch.from_numpy(data_gt_normalized).unsqueeze(0).unsqueeze(0)
                save_image(
                    gt_tensor[0],
                    fp=str(f"{dir_path}/gt_normalized_{iter}.png"),
                )
                mask_tensor = torch.from_numpy(data_mask_normalized).unsqueeze(0).unsqueeze(0)
                save_image(
                    mask_tensor[0],
                    fp=str(f"{dir_path}/mask_normalized_{iter}.png"),
                )

        snr_values_python = [float(x) for x in snr_results] if snr_results else []
        snr_mask_values_python = [float(x) for x in snr_mask_results] if snr_mask_results else []
        snr_dict = {
            'sample_indices': list(range(len(sampled_imgs))),
            'snr_values': snr_values_python,
            'mean_snr': float(np.mean(snr_results)) if snr_results else 0.0,
            'std_snr': float(np.std(snr_results)) if snr_results else 0.0,
            'mean_snr_mask': float(np.mean(snr_mask_results)) if snr_mask_results else 0.0,
            'std_snr_mask': float(np.std(snr_mask_results)) if snr_mask_results else 0.0,
        }
        with open(f"{dir_path}/snr_{iter}.json", 'w') as f:
            json.dump(snr_dict, f, indent=4)
        print(f"Sample {iter} - Mean SNR: {snr_dict['mean_snr']:.2f} dB, Std SNR: {snr_dict['std_snr']:.2f} dB")

        if iter >= 100000:
            break

else:
    raise ValueError(f"Invalid sample type: {sys.argv[2]}. Supported: C3NA, c3na, parallel, parallel_xbfy")
