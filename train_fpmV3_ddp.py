from datetime import datetime, timedelta
import math
import os
import pathlib
import random
from typing import BinaryIO, List, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
# Flow Matching Model
from FPM import FlowMatchingModel
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import dataset_5d
import dataset
import patched_dataset5d
import dataset_dongfang_neighbors_block
import seisDiT
import seisdit_trace_axis
import seisdit_vit_bottleneck
import json
import matplotlib.pyplot as plt
import dataset_5d
from accelerate import Accelerator
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       message="The dataloader does not have many workers.")
PS = 64+24
SD= 64+24           
time_PS=1248
time_SD=1248
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,'
# Accelerate 会自动处理 NCCL 配置
def normalize_clip(data):
    threshold = np.percentile(np.abs(data), 99.5)
    data =np.clip(data, -threshold, threshold)
    data = data / threshold
    return data

RAW_DATA_PATH = "/home/chengzhitong/seismic_ddpm/data/006_3a3_nucns_3a2_data_DX004_p2.sgy"
SIM_DATA_PATH = "/home/czt/seismic_ddpm/data/data_25000_111_noS_cs_smooth5_5d_half_halfPT.h5"
SIM_DATA_PATH = '/NAS/data/data/Syn_seisData/Data/data5d_19_smooth5_Half_Half.h5'
SIM_DATA_PATH = '/home/czt/seismic_ddpm/data/data_25000_111_noS_cs_smooth5_5d_half_halfPT.h5'
SIM_DATA_PATH = '/NAS/data/data/SeismicData/Marmousi/data5d_6_smooth5_Half_Half.h5'
# SIM_DATA_PATH='/NAS/data/data/SeismicData/Overthrust/data5d_6_smooth5_Half_Half.h5'
DATA_NUM = 1200
MODEL_CFG = "sam2_hiera_l.yaml"
SAM2_CKP = '/data/sam2_hiera_large.pt'


def cycle(dl: DataLoader):  # 返回一个迭代器
    while True:
        for data in dl:
            yield data


def save_hyperparameters(res_dir, kwargs, accelerator=None):
    """
    将超参数保存到JSON文件中
    
    Args:
        res_dir: 结果目录路径
        kwargs: 包含超参数的字典
        accelerator: Accelerator 实例
    """
    # 确保只在主进程（rank 0）执行保存
    if accelerator is not None and not accelerator.is_main_process:
        return
    os.makedirs(res_dir, exist_ok=True)  # 创建目录（如果不存在）
    hyperparams = kwargs.copy()
    with open(os.path.join(res_dir, "interp_settings.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    norm: bool = False,
    accelerator=None
) -> None:
    # 确保只在主进程（rank 0）执行保存
    if accelerator is not None and not accelerator.is_main_process:
        return

    print(tensor.shape)
    assert len(tensor.shape) == 3
    tensor = tensor[0, :, :].detach().cpu()
    tensor =tensor-tensor.mean()
    # ori_tensor = ori_tensor[0, 0, :, :].detach().cpu()
    plt.figure(figsize=(6, 6))
    if norm:
       tensor /= torch.abs(torch.max(tensor, dim=0, keepdim=True)[0])
    else:
      tensor = tensor
    plt.pcolor(tensor.T, cmap="seismic", vmin=-tensor.std(), vmax=tensor.std())
    plt.ylim(plt.ylim()[::-1])
    plt.title('generate data')
    plt.xticks([])
    plt.xlabel('Trace index')
    plt.ylabel('Time(s)')
    plt.colorbar()
    # plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(fp, dpi=600)
    plt.close()


def config():
    parser = argparse.ArgumentParser()
    # train config
    parser.add_argument("--model_name", type=str,
                        default="Unet", help="model name")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--seed", type=int, default=515, help="random seed")
    parser.add_argument('--Type', type=str, default='nomal', help='No help')
    # model config
    parser.add_argument("--in_channels", type=int,
                        default=1, help="input channels")
    parser.add_argument("--time_steps", type=int,
                        default=1000, help="number of steps")
    parser.add_argument('--data_type',type = str ,default='sim',help='no help')
    # Flow matching specific arguments
    parser.add_argument("--path_type", type=str, default="Linear", choices=["Linear", "GVP", "VP"],
                        help="Flow matching path type")
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"],
                        help="Model prediction type")
    parser.add_argument("--loss_weight", type=str, default=None, choices=[None, "velocity", "likelihood"],
                        help="Loss weighting type")
    parser.add_argument("--sampling_method", type=str, default="ode", choices=["ode", "sde"],
                        help="Sampling method: ode or sde")
    parser.add_argument("--ode_num_steps", type=int, default=50,
                        help="Number of ODE steps for sampling")
    parser.add_argument("--sde_num_steps", type=int, default=250,
                        help="Number of SDE steps for sampling")
    # others
    # --- 添加分布式训练相关参数 ---
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Required for multi-GPU training.")
    # ---
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        embedding_model: torch.nn.Module,
        flow_matching_model: FlowMatchingModel,
        results_folder: str,
        dl: DataLoader,
        tgt_dl: DataLoader,
        val_dl: DataLoader,
        args: argparse.Namespace,
        accelerator: Accelerator,
        train_batch_size: int = 8,
        train_lr: float = 1e-4,
        epochs: int = 1000,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        save_and_sample_every: int = 100,
        num_samples: int = 2,
        finetune: bool = False,
    ):
        """
        初始化训练器
        
        Args:
            embedding_model: 嵌入模型
            flow_matching_model: Flow Matching 模型
            results_folder: 结果保存文件夹路径
            dl: 训练数据加载器
            tgt_dl: 目标数据加载器
            val_dl: 验证数据加载器
            args: 命令行参数
            accelerator: Accelerator 实例
            train_batch_size: 训练批次大小
            train_lr: 训练学习率
            epochs: 训练总轮数
            adam_betas: Adam优化器的beta参数
            save_and_sample_every: 保存和采样的间隔轮数
            num_samples: 采样数量
            finetune: 是否进行微调
        """
        # self.model = model
        self.embedding_model = embedding_model
        self.flow_matching_model = flow_matching_model
        self.channels = 1
        self.step = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        self.results_folder = Path(results_folder)
        # 只在主进程创建目录
        if accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True)
            self.ckp_folder = self.results_folder / "checkpoints"
            self.ckp_folder.mkdir(exist_ok=True)
            self.img_folder = self.results_folder / "images"
            self.img_folder.mkdir(exist_ok=True)
            self.log_folder = self.results_folder / "logs"
            self.log_folder.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_folder))
        else:
            self.writer = None 

        self.dl = dl
        self.val_dl = val_dl
        self.tgt_dl = tgt_dl

        self.train_epochs = epochs
        # Accelerate 会自动处理 DataLoader 的长度
        self.num_steps = len(self.dl)
        self.train_num_steps = self.train_epochs * self.num_steps

        self.batch_size = train_batch_size
        self.train_lr = train_lr

        # frozen_layers_=[self.flow_matching_model.model.tokenier,self.flow_matching_model.model.down]
        frozen_layers_ = []

        if finetune:
          if len(frozen_layers_) != 0:
            for m in self.flow_matching_model.model.modules():
              if m in frozen_layers_:
                 for params in m.parameters():
                   params.requires_grad = False
          else:
              pass

        # Accelerate 会自动处理模型包装
        model_to_optimize = self.flow_matching_model.model
        if hasattr(model_to_optimize, 'module'):
            model_to_optimize = model_to_optimize.module

        self.opt = AdamW(
            [
                {
                    "params": model_to_optimize.parameters(),
                    "lr": train_lr,
                },
            ],
            lr=train_lr,
            betas=adam_betas,
            weight_decay=5e-4,
        )

        self.opt_fint = AdamW(
            [
                {
                    "params": filter(lambda p: p.requires_grad, model_to_optimize.parameters()),
                    "lr": train_lr,
                }
            ],
            lr=train_lr,
            betas=adam_betas,
            weight_decay=5e-4,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=20, T_mult=2, last_epoch=-1, eta_min=1e-5,
        )
        
        # 使用 Accelerate 准备模型、优化器和数据加载器
        self.flow_matching_model.model, self.opt, self.dl, self.val_dl = accelerator.prepare(
            self.flow_matching_model.model, self.opt, self.dl, self.val_dl
        )

    def save(self, milestone: int) -> None:
        # 只在主进程保存模型
        if not self.accelerator.is_main_process:
            return

        # Accelerate 提供了简化的保存方法
        unwrapped_model = self.accelerator.unwrap_model(self.flow_matching_model.model)
        data = {
            "model": unwrapped_model.state_dict(),
            #"opt": self.opt.state_dict(),
        }

        # torch.save(data, str(self.ckp_folder / f"model-{milestone}_RAW.pth"))
        torch.save(data, str(self.ckp_folder / f"model-{milestone}.pth"))
        # torch.save(data, str(self.ckp_folder / f"model-{milestone}_RAW-liked.pth"))

    def gen(self):
        """
        Generate samples from the model.
        
        This method is intended for generating samples from the trained model.
        Currently it's a placeholder and the main implementation is in train_interpolate method.
        """
        # gen 方法也需要类似修改，但 train_interpolate 是主要训练方法，这里省略 gen 的修改
        pass

    def train_interpolate_improved(self):
        accumulation_steps = 4
        with tqdm(range(self.train_epochs), total=self.train_epochs, desc=f"Training",
                disable=not self.accelerator.is_main_process) as pbar:
            data_sample, data_mask_sample, rx, ry, sx, sy, _,_ = next(iter(self.val_dl))
            _, H, W = data_sample.shape
            data_mask_sample = data_mask_sample.unsqueeze(1)
            condL_sample = (rx, ry, sx, sy)
            if self.accelerator.is_main_process:
                #save_image(data_sample, fp=str(self.img_folder / "data_ori.png"),
                #        accelerator=self.accelerator)
                pass
            for epoch in pbar:
                loss_list = []
                if hasattr(self.dl.sampler, 'set_epoch'):
                    self.dl.sampler.set_epoch(epoch)
                self.flow_matching_model.train()
                self.opt.zero_grad(set_to_none=True) 
                for idx, (data, data_mask, rx, ry, sx, sy, time, coord) in enumerate(self.dl):
                    data = data.unsqueeze(1).to(self.device)
                    data_mask = data_mask.unsqueeze(1).to(self.device)
                    rx, ry, sx, sy = rx.to(self.device), ry.to(self.device), sx.to(self.device), sy.to(self.device)
                    condL = (rx, ry, sx, sy)
                    time = time.to(self.device)
                    with self.accelerator.autocast():
                        # Flow Matching 的 forward 方法返回 loss
                        loss = self.flow_matching_model(data, condL=condL, x_cond=data_mask, time=time)
                    loss = loss / accumulation_steps
                    self.accelerator.backward(loss)
                    do_step = ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(self.dl))
                    if do_step:
                        self.accelerator.clip_grad_norm_(self.flow_matching_model.parameters(), max_norm=1.0)
                        self.opt.step()
                        # 提前推进学习率调度，降低初期抖动
                        step_progress = epoch + (idx + 1) / len(self.dl)
                        self.opt.zero_grad(set_to_none=True)

                    # 记录原始 loss（放大回来）
                    loss_list.append(loss.item() * accumulation_steps)

                    # 只打印一次首个 loss
                    if epoch == 0 and idx == 0 and self.accelerator.is_main_process:
                        print("the first loss is:", loss.item() * accumulation_steps)
                self.lr_scheduler.step(epoch)
                avg_loss = sum(loss_list) / len(loss_list) if loss_list else float('nan')

                gc.collect(); torch.cuda.empty_cache()
                if self.accelerator.is_main_process:
                    pbar.set_postfix({"current_epoch": epoch + 1, "loss": avg_loss})
                if ((epoch+1) % self.save_and_sample_every == 0 or epoch == 0) and self.accelerator.is_main_process:
                    #print('Sampling...')
                    unwrapped_model = self.accelerator.unwrap_model(self.flow_matching_model.model)
                    self.flow_matching_model.model.eval()
                    old_disable = os.environ.get('TQDM_DISABLE', '0')
                    os.environ['TQDM_DISABLE'] = '1'
                    try:
                        milestone = (epoch + 1) // self.save_and_sample_every
                        # Sampling code can be added here if needed
                        self.save(milestone)
                        torch.cuda.empty_cache()
                    finally:
                        # 恢复 tqdm 环境变量
                        os.environ['TQDM_DISABLE'] = old_disable
                # Accelerate 会自动处理同步
                self.accelerator.wait_for_everyone()


def main():
    """
    Main function to run the training process.
    
    This function orchestrates the complete training pipeline including:
    - Parsing command line arguments
    - Setting up Accelerate framework
    - Loading datasets based on data type
    - Initializing the model and flow matching process
    - Creating the trainer and starting the training loop
    
    The function supports multiple data types and uses Accelerate to simplify
    distributed training setup.
    """
    args = config()
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',  # 可以改为 'fp16' 或 'bf16' 来加速训练
        kwargs_handlers=[ddp_kwargs],
    )
    
    # 获取设备信息
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    print(f"[Rank {rank}/{world_size}] Using device: {device}")
    # set seed
    # 手动设置随机种子以确保可重复性
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    interp_kwargs = dict(
        missing_type='V15rope_wo_adalnV2_vit',
        missing_rate=[0.5,0.6,0.7,],
        interval=4,  # 间隔采样
    )
    
    # Flow Matching 参数配置
    fpm_kwargs = dict(
        time_num=time_PS,
        trace_num=PS,
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=None,  # 使用默认值
        sample_eps=None,  # 使用默认值
        sample_num=1,
        device=device,
        sup_mode='all',  # 保持兼容性
        use_coherence=False,
        sigma_obs=0.001,
        use_bayesian=False,
        sampling_method=args.sampling_method,
        ode_num_steps=args.ode_num_steps,
        sde_num_steps=args.sde_num_steps,
    )
    
    # model init
    pe_type = 'transformer'
    p_scale =patched_dataset5d.RAW_dongfang_1031V2(time_ps=time_PS,trace_ps=PS).space_scale
    print('p_scale:',p_scale)
    if p_scale is not None:
        use_p_scale = True
    else:
        use_p_scale = False
    #model_unet = seisdit_trace_axis.SeisDiTRopeV2(image_channels=2, n_channels=32,f_dict=None,num_layers=12,d_model=512,pe_type=pe_type,rope_p_scale=p_scale)
    model_unet = seisdit_vit_bottleneck.SeisDiTRope(image_channels=2, n_channels=32,f_dict=None,num_layers=12,d_model=512,pe_type=pe_type,p_scale=p_scale)
    print("time_steps:", args.time_steps)
    res_dir = f"./resultsFPM/{args.model_name}_datatype_{args.data_type}_{interp_kwargs['missing_type']}_{args.path_type}_{args.prediction}_p_scale{use_p_scale}"

    ds2d_kwargs = dict(
        h5File_ori="/home/czt/seismic_ddpm/Seis_DiT/forward/model_show_marmousi/synthetic_data_1024.h5",
        time_num=512*2,
        shot_id=list(range(115)),
        trainning=False,
    )
    if args.data_type == 'marmousi':
        ds_kwargs = dict(
            file_path=SIM_DATA_PATH,
            data_num=DATA_NUM,
            batch_size=args.batch_size,
            timesize=1024,
            datasize=(16, 16, 39),
            trace_num=64,
            random_trace=False,
            idx_list='/home/czt/seismic_ddpm/Seis_DiT/index_info_shotnum256_random.json',
            sub=False,
        )
        ds_kwargs_mask = dict(
            file_path=SIM_DATA_PATH,
            data_num=DATA_NUM,
            batch_size=args.batch_size,
            timesize=1024//2,
            datasize=(16, 16, 39),
            trace_num=64,
            random_trace=False,
            idx_list='/home/chengzhitong/Seis_DiT/index_info_shotnum256.json',
            sub=False,
            missing_mode=interp_kwargs['missing_type'],
            missing_ratio=interp_kwargs['missing_rate'],
            interval=4,
        )
        dl_SIM, dl_SIM_VAL = dataset.load_sim_data_mask(
            **ds_kwargs_mask,
        )
    # field data setting
    if args.data_type == 'raw':
        with open('/home/chengzhitong/Seis_DiT/data/segy2H5/006_3a3_nucns_3a2_data_DX004_p2_fintune_list.json') as f:
            idx_log = json.load(f)
        rawdata_kwargs_fint = dict(
            file_path='/home/chengzhitong/Seis_DiT/data/segy2H5/006_3a3_nucns_3a2_data_DX004_p2.h5',
            data_num=None,
            data_name='006_3a3_nucns_3a2_data_DX004_p2.sgy',
            trace_mode='order',
            slice_id=idx_log['fintune']+idx_log['test'],
            traces_num=270,
            missing_mode=interp_kwargs['missing_type'],
            missing_ratio=interp_kwargs['missing_rate'],
        )
        rawdata_kwargs_test = dict(
            file_path='/home/chengzhitong/Seis_DiT/data/segy2H5/006_3a3_nucns_3a2_data_DX004_p2.h5',
            data_num=None,
            data_name='006_3a3_nucns_3a2_data_DX004_p2.sgy',
            trace_mode='order',
            slice_id=idx_log['test'],
            traces_num=270,
            missing_mode=interp_kwargs['missing_type'],
            missing_ratio=interp_kwargs['missing_rate'],
        )
        raw_dataset_fint = dataset.raw_dataset(
            **rawdata_kwargs_fint,
        )
        raw_dataset_test = dataset.raw_dataset(
            **rawdata_kwargs_test,
        )

        train_sampler = DistributedSampler(raw_dataset_fint) if world_size > 1 else None
        val_sampler = DistributedSampler(raw_dataset_test, shuffle=False) if world_size > 1 else None

        dl_SIM = DataLoader(raw_dataset_fint,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=4,
                            sampler=train_sampler)
        dl_SIM_VAL = DataLoader(raw_dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                sampler=val_sampler)
    if args.data_type == '2d':
        dataset2d = dataset.dataset_2d(**ds2d_kwargs)
        train_sampler = DistributedSampler(dataset2d) if world_size > 1 else None
        val_sampler = DistributedSampler(dataset2d, shuffle=False) if world_size > 1 else None

        dl_SIM = DataLoader(
           dataset2d,
           batch_size=args.batch_size,
           shuffle=(train_sampler is None),
           num_workers=4,
           sampler=train_sampler)
        dl_SIM_VAL = DataLoader(
           dataset2d,
           batch_size=1,
           shuffle=False,
           num_workers=4,
           sampler=val_sampler)
    if args.data_type =='marmousi_5d':
        cfg_5d={
            'data_path': SIM_DATA_PATH,
            'batch_size': args.batch_size,
            'time_size':fpm_kwargs['time_num'],
            "shot_win": (2,2),
            "rec_win": (1,1),
            "mask_value": 0.0,
            "missing_ratio": 0.5,
            "missing_type": "both",
            "missing_mode": "random",
            "missing_seed": 1221,
            "user_shot_mask": None,
            "user_rec_mask": None
        }
        if accelerator.is_main_process:
            hyperparams = cfg_5d.copy()
            os.makedirs(res_dir, exist_ok=True)
            with open(os.path.join(res_dir, "config.json"), "w") as f:
                json.dump(hyperparams, f, indent=4)
        dl_SIM,dl_SIM_VAL = dataset_5d.load_5d_dataset(**cfg_5d)
        try:
            train_dataset = dl_SIM.dataset
            val_dataset = dl_SIM_VAL.dataset
        except AttributeError:
            print("Warning: Could not access dataset from DataLoader. Distributed training may not work correctly for marmousi_5d.")
            train_dataset = dl_SIM.dataset if hasattr(dl_SIM, 'dataset') else None
            val_dataset = dl_SIM_VAL.dataset if hasattr(dl_SIM_VAL, 'dataset') else None

        if train_dataset is not None:
            train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
            dl_SIM = DataLoader(train_dataset, batch_size=cfg_5d['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler, num_workers=dl_SIM.num_workers if hasattr(dl_SIM, 'num_workers') else 0)

        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
            dl_SIM_VAL = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=dl_SIM_VAL.num_workers if hasattr(dl_SIM_VAL, 'num_workers') else 0)

    if args.data_type == 'c3NA':
        from torch.utils.data import Subset
        cfg_c3NA_train={'max_traces':350000,
                        'root_List':['/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_1201-2400.sgy',
                                     '/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_2401-3600.sgy',
                                     ],
                        'time_step':512,
                        'missing_rate_list':[0.2,0.3,0.4,0.5,0.6,0.7,],
                        'contin_missing':True,
                        }
        dataset_c3NA_train = patched_dataset5d.SEG_C3NA_patched(**cfg_c3NA_train,ps=PS,)
        cfg_c3NA_val={'max_traces':3000000,
                      'root_List':['/home/chengzhitong/Seis_DiT/data/C3/SEG_C3NA_ffid_3601-4781.sgy'],
                      'time_step':512,
                      'missing_rate_list':[0.3,0.5,0.7],
                      'contin_missing':True,
                    }
        dataset_c3NA_val = dataset_5d.SEG_C3NA(**cfg_c3NA_val)

        train_sampler = DistributedSampler(dataset_c3NA_train) if world_size > 1 else None
        val_sampler = DistributedSampler(dataset_c3NA_val, shuffle=False) if world_size > 1 else None

        dl_SIM = DataLoader(dataset_c3NA_train,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=4,
                            sampler=train_sampler)
        dl_SIM_VAL = DataLoader(dataset_c3NA_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                sampler=val_sampler)
        print(f'Rank {rank}: C3NA dataset loaded')
        print(f"Rank {rank}: train dataset size: {len(dataset_c3NA_train)}")
        print(f"Rank {rank}: val dataset size: {len(dataset_c3NA_val)}")

    if args.data_type == 'dongfang':
        dataset_train = patched_dataset5d.RAW_dongfang_1031V2(
                            time_ps=time_PS,
                            trace_ps=PS,
                            )
        dataset_test = patched_dataset5d.RAW_dongfang_1031V2(
                            time_ps=time_PS,
                            trace_ps=PS,
                            train=False)
        train_sampler = DistributedSampler(dataset_train) if world_size > 1 else None
        val_sampler = DistributedSampler(dataset_test, shuffle=False) if world_size > 1 else None

        dl_SIM = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=4,
                            sampler=train_sampler,
                            drop_last=True,
                            )
        dl_SIM_VAL = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                sampler=val_sampler)
        print(len(dataset_train),len(dataset_test))
    if args.data_type == 'dongfang_neighbors_block':
        dataset_train = dataset_dongfang_neighbors_block.RAW_dongfang_neighbors_block(
                            neighbors_path="./dongfang/train_neighbors_greedy.npy",
                            time_ps=time_PS,
                            trace_ps=PS,
                            )
        dataset_test = patched_dataset5d.RAW_dongfang_1031V2(
                            time_ps=time_PS,
                            trace_ps=PS,
                            train=False)
        train_sampler = DistributedSampler(dataset_train) if world_size > 1 else None
        val_sampler = DistributedSampler(dataset_test, shuffle=False) if world_size > 1 else None
        dl_SIM = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=4,
                            sampler=train_sampler,
                            drop_last=True,
                            )
        dl_SIM_VAL = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                sampler=val_sampler,
                                )
        print(len(dataset_train),len(dataset_test))

    # 将模型移动到正确的设备
    model_unet = model_unet.to(device)

    # 创建 Flow Matching Model
    fpm = FlowMatchingModel(
        model=model_unet,
        trace_num=fpm_kwargs['trace_num'],
        time_steps=fpm_kwargs['time_num'],
        path_type=fpm_kwargs['path_type'],
        prediction=fpm_kwargs['prediction'],
        loss_weight=fpm_kwargs['loss_weight'],
        train_eps=fpm_kwargs['train_eps'],
        sample_eps=fpm_kwargs['sample_eps'],
        sample_num=fpm_kwargs['sample_num'],
        device=fpm_kwargs['device'],
        sup_mode=fpm_kwargs['sup_mode'],
        use_coherence=fpm_kwargs['use_coherence'],
        sigma_obs=fpm_kwargs['sigma_obs'],
        use_bayesian=fpm_kwargs['use_bayesian'],
        sampling_method=fpm_kwargs['sampling_method'],
        ode_num_steps=fpm_kwargs['ode_num_steps'],
        sde_num_steps=fpm_kwargs['sde_num_steps'],
    )

    # 只在主进程创建结果目录
    if accelerator.is_main_process:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    # init trainer
    trainer_fpm = trainer(
        embedding_model=None,
        flow_matching_model=fpm,
        results_folder=res_dir,
        dl=dl_SIM,
        tgt_dl=None,
        val_dl=dl_SIM_VAL,
        args=args,
        accelerator=accelerator,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        epochs=args.epochs,
        save_and_sample_every=10,
        num_samples=args.batch_size,
        finetune=False
    )
    # train
    trainer_fpm.train_interpolate_improved()
if __name__ == "__main__":
    main()
