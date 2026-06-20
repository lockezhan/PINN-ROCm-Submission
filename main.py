import os
# 设置为 PyTorch 后端
os.environ["DDE_BACKEND"] = "pytorch"

import argparse
import numpy as np
import torch
import torch.distributed as dist
from core.pde_def import get_ns_equation_data
from core.network import build_network
from core.trainer import train_model
from core.visualizer import plot_ns_results, plot_loss_curve

def parse_args():
    parser = argparse.ArgumentParser(description="PINN Framework with 8-GPU DDP Support")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument("--scale", type=str, choices=["small", "large", "extreme"], default="small", 
                        help="Data scale: small (local test), large (server smooth), extreme (OOM boundary/8-GPU max)")
    parser.add_argument("--precision", type=str, choices=["float32", "float16", "bfloat16"], default="float32",
                        help="Precision format. Note: float16 underflows Hessian. bfloat16 recommended for ROCm.")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler for performance tracing")
    parser.add_argument("--batch_size", type=int, default=0, help="Mini-batch size. 0 means auto-scale to saturate GPU based on scale.")
    parser.add_argument("--tol", type=float, default=-1.0, help="Convergence tolerance for early stopping. -1.0 means disabled.")
    parser.add_argument("--out_dir", type=str, default="", help="Custom output directory. If empty, auto-generates based on scale & precision.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint model file (.pt) to resume training from.")
    parser.add_argument("--time_limit", type=float, default=-1.0, help="Maximum training time limit in seconds. Default is -1.0 (no limit).")
    return parser.parse_args()

def init_distributed():
    """初始化 DDP 分布式进程组。兼容单卡与多卡。"""
    is_ddp = False
    local_rank = 0
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl") # NCCL on ROCm defaults to RCCL
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_ddp = True
    return is_ddp, local_rank

def print_hardware_info(is_ddp):
    print("=" * 60)
    print("Hardware & Environment Info:")
    print(f"DeepXDE Backend: {os.environ['DDE_BACKEND']}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"DDP Multi-GPU Mode: {'ENABLED' if is_ddp else 'DISABLED'}")
    if is_ddp:
        print(f"World Size: {dist.get_world_size()} GPUs")
    if torch.cuda.is_available():
        print("Device Available: True (GPU Detected)")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("Device Available: False (Running on CPU)")
    print("=" * 60)

def main():
    # 1. 尝试初始化 DDP
    is_ddp, local_rank = init_distributed()
    args = parse_args()
    
    # 2. 动态自适应 Batch Size 分配 (榨干硬件算力)
    if args.batch_size == 0:
        if args.scale == "small":
            args.batch_size = 2000     # 全量运行，不切分
        elif args.scale == "large":
            args.batch_size = 40000    # 大约占用 6-7GB VRAM
        elif args.scale == "extreme":
            args.batch_size = 150000   # 极限压榨：大约占用 24GB VRAM (刚好塞满 4090，在 W7900 上也能跑出极高并发)

    # 自动生成隔离输出目录，解决并发时文件写入竞争
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"outputs_{args.scale}_{args.precision}"

    if local_rank == 0:
        print_hardware_info(is_ddp)
        print(f"[Main] Scale: {args.scale.upper()} | Precision: {args.precision.upper()} | Batch: {args.batch_size} | Out Dir: {out_dir}")

    # 2. 获取几何结构与函数，不再生成深耦合 of dde.data.PDE
    geom, pde, funcs, num_domain, num_boundary, num_test = get_ns_equation_data(scale_factor=args.scale)

    # 3. 构建神经网络
    net = build_network(scale_factor=args.scale, precision=args.precision)

    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            # map_location="cpu" to safely load onto CPU before placing on DDP GPU
            checkpoint_state = torch.load(args.resume, map_location="cpu")
            # Clean "module." prefix if the checkpoint was saved from a DDP model
            cleaned_state = {k.replace("module.", ""): v for k, v in checkpoint_state.items()}
            net.load_state_dict(cleaned_state)
            if local_rank == 0:
                print(f"[Main] Successfully loaded checkpoint weights from {args.resume}")
            
            # 尝试从文件名解析起始 epoch，例如 "model_ep5000.pt"
            import re
            match = re.search(r"model_ep(\d+)\.pt", os.path.basename(args.resume))
            if match:
                start_epoch = int(match.group(1))
                if local_rank == 0:
                    print(f"[Main] Detected start epoch from checkpoint filename: {start_epoch}")
        else:
            if local_rank == 0:
                print(f"[Main] Error: Checkpoint file {args.resume} not found!")

    # 4. 执行 DDP/单卡自适应的自定义 PyTorch 训练循环
    trained_net, loss_history = train_model(
        geom=geom, pde_fn=pde, funcs=funcs,
        num_domain=num_domain, num_boundary=num_boundary,
        net=net, epochs=args.epochs, batch_size=args.batch_size, 
        precision=args.precision, tol=args.tol, out_dir=out_dir, profile=args.profile,
        start_epoch=start_epoch, time_limit=args.time_limit
    )

    # 5. 仅在主进程进行流场生成，防止 8 个进程同时读写 IO 冲突
    if local_rank == 0:
        print(f"\n[Main] Training finished. Generating 2D Navier-Stokes visualizations in {out_dir}...")
        
        # 因为在 custom trainer 中生成了 loss_history 列表，保存它
        os.makedirs(f"{out_dir}/figures", exist_ok=True)
        
        # 从实时不断追加生成的 loss.dat 中直接读取完整记录用于绘制最终收敛图
        loss_file_path = f"{out_dir}/figures/loss.dat"
        if os.path.exists(loss_file_path):
            try:
                full_loss = np.loadtxt(loss_file_path, skiprows=1)
                if len(full_loss.shape) == 1:
                    full_loss = full_loss.reshape(1, -1)
                loss_history = [tuple(row) for row in full_loss]
            except Exception as e:
                print(f"[Main] Warning: Could not read full loss history for plotting: {e}")
        
        # 为了给可视化函数喂测试数据，我们在主进程单独采样
        X_test = geom.random_points(num_test)
        # 解包 DDP net 提取底层模块用于推理
        base_net = trained_net.module if hasattr(trained_net, 'module') else trained_net
        base_net.eval()
        
        # 使用 visualizer 中改造过的绘制接口
        plot_ns_results(base_net, funcs, X_test, out_dir=out_dir)
        plot_loss_curve(loss_history, out_dir=out_dir)
        print(f"\n✅ All artifacts saved in {out_dir}/ directory.")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
