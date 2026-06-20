import torch
import torch.nn as nn
import torch.distributed as dist
import os
from .profiler import ProfilerContext, HardwareMonitor

class GPUDataLoader:
    def __init__(self, tensors, batch_size, shuffle=True, is_ddp=False, local_rank=0, world_size=1):
        self.tensors = tensors
        self.dataset_size = tensors[0].size(0)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_ddp = is_ddp
        self.local_rank = local_rank
        self.world_size = world_size
        
        if self.is_ddp:
            self.rank_dataset_size = (self.dataset_size - self.local_rank + self.world_size - 1) // self.world_size
        else:
            self.rank_dataset_size = self.dataset_size

    def __len__(self):
        return (self.rank_dataset_size + self.batch_size - 1) // self.batch_size

    def get_epoch_iterator(self, epoch=0):
        if self.batch_size >= self.rank_dataset_size:
            # 全量训练优化：直接 yield 显存原张量，避免任何索引切片和 GPU 拷贝
            if self.is_ddp:
                yield tuple(t[self.local_rank::self.world_size] for t in self.tensors)
            else:
                yield self.tensors
            return

        # 微批次采样 (Mini-batch)
        if self.shuffle:
            if self.is_ddp:
                g = torch.Generator(device=self.tensors[0].device)
                g.manual_seed(epoch)
                perm = torch.randperm(self.dataset_size, generator=g, device=self.tensors[0].device)
                rank_perm = perm[self.local_rank::self.world_size]
            else:
                rank_perm = torch.randperm(self.dataset_size, device=self.tensors[0].device)
        else:
            if self.is_ddp:
                rank_perm = torch.arange(self.local_rank, self.dataset_size, self.world_size, device=self.tensors[0].device)
            else:
                rank_perm = torch.arange(self.dataset_size, device=self.tensors[0].device)

        num_batches = len(self)
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.rank_dataset_size)
            batch_perm_indices = rank_perm[start_idx:end_idx]
            yield tuple(t[batch_perm_indices] for t in self.tensors)

def train_model(geom, pde_fn, funcs, num_domain, num_boundary, net, epochs=15000, batch_size=8192, precision="float32", tol=-1.0, out_dir="outputs", profile=False, start_epoch=0, time_limit=-1.0):
    import time
    start_time = time.time()
    # DeepXDE 默认将全局设备设置为 cuda，这会导致 DataLoader 内部基于 CPU 的随机生成器 (Generator) 崩溃。
    # 因为我们已经在代码里手动使用了 .to(device) 转移张量，所以这里安全地将全局默认恢复为 cpu
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
        
    # 根据用户指定的精度确定是否启用自动混合精度 (AMP)
    # 如果使用 float32 模式，我们将关闭 autocast，确保高阶微分和梯度计算精度不损失
    use_amp = (precision in ["bfloat16", "float16"])
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        
    is_ddp = dist.is_initialized()
    local_rank = dist.get_rank() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    net = net.to(device)
    if is_ddp:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    if local_rank == 0:
        print("[Trainer] Note: torch.compile is disabled as it does not support double backward for PINNs.")
        
    # 【优化项 3：开启 Fused Adam】
    # 将 Adam 内部的数十次分散的显存读写操作融合成单个 CUDA Kernel，显著降低 CPU 开销
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, fused=(torch.cuda.is_available()))
    
    # 手动为新创建的 optimizer 注入 initial_lr，以防止 PyTorch scheduler 在断点续训 (last_epoch > -1) 时报 KeyError
    if start_epoch > 0:
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', 1e-3)
            
    # 引入余弦退火学习率调度器，从 1e-3 降至 1e-5，精细微调后期收敛精度，支持断点续训
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(start_epoch + epochs), eta_min=1e-5, last_epoch=start_epoch - 1
    )
    
    # 【性能核弹级优化1：提前计算目标值，彻底剔除内部循环的 CPU 拷贝与 Numpy 转换】
    if local_rank == 0:
        print("[Trainer] Generating collocation points and moving to GPU...")
    X_domain = geom.random_points(num_domain)
    X_bc = geom.random_boundary_points(num_boundary)
    
    u_func, v_func, p_func = funcs
    # 提前在外部计算好所有边界点对应的 Ground Truth，避免在每个 Batch 中重复计算
    U_true_np = u_func(X_bc)
    V_true_np = v_func(X_bc)
    P_true_np = p_func(X_bc)
    
    # 【性能核弹级优化2：将所有数据提前常驻显存 (GPU)，利用 DataLoader 的 GPU 内部切片消除 PCIe 带宽瓶颈】
    tensor_domain = torch.tensor(X_domain, dtype=torch.float32, device=device)
    tensor_bc_x = torch.tensor(X_bc, dtype=torch.float32, device=device)
    tensor_bc_u = torch.tensor(U_true_np, dtype=torch.float32, device=device)
    tensor_bc_v = torch.tensor(V_true_np, dtype=torch.float32, device=device)
    tensor_bc_p = torch.tensor(P_true_np, dtype=torch.float32, device=device)

    # 实例化 GPU 自定义 DataLoader，消除 PyTorch 原生 DataLoader 重复创建/销毁迭代器导致的 GPU 饥饿
    world_size = dist.get_world_size() if is_ddp else 1
    loader_domain = GPUDataLoader(
        [tensor_domain], 
        batch_size=batch_size, 
        shuffle=True, 
        is_ddp=is_ddp, 
        local_rank=local_rank, 
        world_size=world_size
    )
    bc_batch_size = max(1, int(batch_size * (len(tensor_bc_x) / len(tensor_domain))))
    loader_bc = GPUDataLoader(
        [tensor_bc_x, tensor_bc_u, tensor_bc_v, tensor_bc_p], 
        batch_size=bc_batch_size, 
        shuffle=True, 
        is_ddp=is_ddp, 
        local_rank=local_rank, 
        world_size=world_size
    )
    
    if local_rank == 0:
        os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
        num_gpus = dist.get_world_size() if is_ddp else 1
        monitor = HardwareMonitor(log_dir=f"{out_dir}/profiling", interval=2.0, num_gpus=num_gpus, is_ddp=is_ddp)
        monitor.start()

    loss_history = []
    
    # 【极致性能优化：CUDA Graph 静态图加速】
    # 针对单卡全量训练 (Full-batch) 触发 CUDA Graph。通过将复杂的 Autograd 双重反向求导流录制为静态 GPU 指令流，
    # 消除所有的 CPU 派发和 kernel 启动延迟，真正吃满显卡算力！
    use_cuda_graph = torch.cuda.is_available() and not is_ddp and (len(loader_domain) == 1) and (len(loader_bc) == 1)
    
    if use_cuda_graph:
        if local_rank == 0:
            print("[Trainer] CUDA Graph is ENABLED for full-batch training acceleration!")
            
        # 1. 提取静态输入数据
        domain_batch = next(iter(loader_domain.get_epoch_iterator(0)))
        bc_batch = next(iter(loader_bc.get_epoch_iterator(0)))
        
        static_batch_domain = domain_batch[0].clone().detach().requires_grad_(True)
        static_batch_bc_x = bc_batch[0].clone().detach()
        static_batch_bc_u = bc_batch[1].clone().detach()
        static_batch_bc_v = bc_batch[2].clone().detach()
        static_batch_bc_p = bc_batch[3].clone().detach()
        
        # 2. 建立静态 Loss 缓存用于将数值传出图外
        static_loss_pde = torch.zeros(1, device=device)
        static_loss_bc = torch.zeros(1, device=device)
        
        # 3. 定义可被录制的单步计算
        def graph_step():
            optimizer.zero_grad(set_to_none=False)  # 静态图下必须使用 set_to_none=False 保持显存地址不变
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # --- PDE Loss ---
                static_batch_domain.requires_grad_(True)
                y_pred_domain = net(static_batch_domain)
                residuals = pde_fn(static_batch_domain, y_pred_domain)
                loss_pde = sum(torch.mean(r**2) for r in residuals)
                
                # --- BC Loss ---
                y_pred_bc = net(static_batch_bc_x)
                u_pred, v_pred, p_pred = y_pred_bc[:, 0:1], y_pred_bc[:, 1:2], y_pred_bc[:, 2:3]
                
                loss_bc = torch.mean((u_pred - static_batch_bc_u)**2) + \
                          torch.mean((v_pred - static_batch_bc_v)**2) + \
                          torch.mean((p_pred - static_batch_bc_p)**2)
                          
                loss = loss_pde + loss_bc * 10.0
                
            loss.backward()
            static_loss_pde.copy_(loss_pde)
            static_loss_bc.copy_(loss_bc)

        # 4. 执行 Warmup (预热以稳定显存池和 CUDA 流)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(11):
                graph_step()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(s)
        
        # 5. 正式录制 CUDA Graph (仅包含 Forward & Loss & Backward，Optimizer 在外侧以 eager 模式最大兼容执行)
        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(g):
            graph_step()

    if local_rank == 0:
        print("[Trainer] Starting custom PyTorch DDP Mini-Batch training loop...")

    with ProfilerContext(use_profiler=(profile and local_rank == 0), log_dir=f"{out_dir}/profiling/tensorboard_traces"):
        for epoch in range(start_epoch, epochs):
            # 检查时间限制，防范多卡 DDP 异步锁死，所有 rank 协同退出
            if time_limit > 0 and (time.time() - start_time) > time_limit:
                if local_rank == 0:
                    print(f"\n[Trainer] Time limit of {time_limit:.1f}s reached at epoch {epoch}. Saving checkpoint and exiting gracefully...")
                    torch.save(net.state_dict(), f"{out_dir}/checkpoints/model_ep{epoch}.pt")
                break
            net.train()
            epoch_loss_pde = 0.0
            epoch_loss_bc = 0.0
            batches = 0
            
            if use_cuda_graph:
                # 运行录制好的 CUDA Graph
                g.replay()
                # 依然在 eager 模式下安全运行优化器步骤，最大化兼容性
                optimizer.step()
                
                epoch_loss_pde += static_loss_pde.item()
                epoch_loss_bc += static_loss_bc.item()
                batches += 1
            else:
                # Eager 模式运行 (支持多 batch 与 DDP)
                for (batch_domain,), (batch_bc_x, batch_u, batch_v, batch_p) in zip(loader_domain.get_epoch_iterator(epoch), loader_bc.get_epoch_iterator(epoch)):
                    optimizer.zero_grad(set_to_none=True)
                    
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        # --- PDE Loss ---
                        batch_domain.requires_grad_(True)
                        y_pred_domain = net(batch_domain)
                        residuals = pde_fn(batch_domain, y_pred_domain)
                        loss_pde = sum(torch.mean(r**2) for r in residuals)
                        
                        # --- BC Loss ---
                        y_pred_bc = net(batch_bc_x)
                        u_pred, v_pred, p_pred = y_pred_bc[:, 0:1], y_pred_bc[:, 1:2], y_pred_bc[:, 2:3]
                        
                        loss_bc = torch.mean((u_pred - batch_u)**2) + \
                                  torch.mean((v_pred - batch_v)**2) + \
                                  torch.mean((p_pred - batch_p)**2)
                                  
                        loss = loss_pde + loss_bc * 10.0
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss_pde += loss_pde.item()
                    epoch_loss_bc += loss_bc.item()
                    batches += 1
            
            epoch_loss_pde /= max(1, batches)
            epoch_loss_bc /= max(1, batches)
            
            # 同步各卡之间的 Loss 用于日志打印
            if is_ddp:
                pde_t = torch.tensor(epoch_loss_pde, device=device)
                bc_t = torch.tensor(epoch_loss_bc, device=device)
                dist.all_reduce(pde_t, op=dist.ReduceOp.AVG)
                dist.all_reduce(bc_t, op=dist.ReduceOp.AVG)
                epoch_loss_pde = pde_t.item()
                epoch_loss_bc = bc_t.item()
            
            # 步进学习率调度器，随 Epoch 衰减以微调收敛精度
            scheduler.step()
            
            # 早停机制 (Early Stopping)：基于收敛容差 tol 自动判断退出
            total_loss = epoch_loss_pde + epoch_loss_bc
            if tol > 0 and total_loss < tol:
                if local_rank == 0:
                    print(f"\n[Trainer] Convergence reached at epoch {epoch}: Total Loss {total_loss:.4e} < tol {tol:.4e}. Early stopping...")
                break
            
            if local_rank == 0 and epoch % 100 == 0:
                print(f"Epoch {epoch:5d} | PDE Loss: {epoch_loss_pde:.4e} | BC Loss: {epoch_loss_bc:.4e}")
                loss_history.append((epoch, epoch_loss_pde, epoch_loss_bc))
                
                # Real-time append to loss.dat to survive unexpected kills/crashes
                loss_file = f"{out_dir}/figures/loss.dat"
                os.makedirs(os.path.dirname(loss_file), exist_ok=True)
                # Overwrite at the very beginning of a fresh start, otherwise append
                mode = "w" if (epoch == 0 and start_epoch == 0) else "a"
                with open(loss_file, mode) as f:
                    if mode == "w":
                        f.write("Epoch, PDE_Loss, BC_Loss\n")
                    f.write(f"{epoch} {epoch_loss_pde} {epoch_loss_bc}\n")
                
                if epoch % 1000 == 0:
                    base_net = net.module if hasattr(net, 'module') else net
                    torch.save(base_net.state_dict(), f"{out_dir}/checkpoints/model_ep{epoch}.pt")

    if local_rank == 0:
        monitor.stop()
        
    return net, loss_history
