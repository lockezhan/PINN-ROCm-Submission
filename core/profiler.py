import os
import time
import threading
import subprocess

def get_active_physical_gpus(is_ddp=False):
    """
    获取当前运行进程实际上被分配的物理 GPU 索引列表。
    通过解析 HIP_VISIBLE_DEVICES 或 CUDA_VISIBLE_DEVICES 环境变量，并结合 PyTorch 的当前设备来判定。
    """
    visible_str = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_str is None:
        visible_str = os.environ.get("CUDA_VISIBLE_DEVICES")
        
    visible_gpus = None
    if visible_str is not None:
        if not visible_str.strip():
            return []
        try:
            visible_gpus = []
            for x in visible_str.split(","):
                x = x.strip()
                if x.isdigit():
                    visible_gpus.append(int(x))
        except:
            visible_gpus = None

    # 如果是 DDP 模式，主进程监控当前可见的所有 GPU
    if is_ddp:
        return visible_gpus

    # 如果是单卡模式，我们只关心 PyTorch 当前正在使用的那个物理 GPU
    try:
        import torch
        if torch.cuda.is_available():
            logical_idx = torch.cuda.current_device()
            if visible_gpus is not None:
                if logical_idx < len(visible_gpus):
                    return [visible_gpus[logical_idx]]
                else:
                    return [visible_gpus[0]]
            else:
                return [logical_idx]
    except:
        pass
        
    return [0] if visible_gpus is None else visible_gpus

class HardwareMonitor:
    def __init__(self, log_dir="outputs/profiling", interval=1.0, num_gpus=1, is_ddp=False):
        self.log_dir = log_dir
        self.interval = interval
        self.num_gpus = num_gpus
        self.is_ddp = is_ddp
        self.running = False
        self.thread = None
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "hardware_metrics.log")
        
        # 动态查询当前 GPU 总显存，用于将 rocm-smi 的百分比转换为 MB
        self.total_mem_mb = 24576.0  # 默认 24GB (RTX 4090 / Radeon VII 等)
        try:
            import torch
            if torch.cuda.is_available():
                self.total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except:
            pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_loop(self):
        active_gpus = get_active_physical_gpus(self.is_ddp)
        
        file_exists = os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0
        with open(self.log_file, "a") as f:
            if not file_exists:
                f.write("Time,Backend,GPU_ID,VRAM_MB,Power_W,GPU_Util\n")
            while self.running:
                # 优先尝试 amdsmi (Python API)
                amdsmi_success = False
                try:
                    import amdsmi
                    amdsmi.amdsmi_init()
                    devices = amdsmi.amdsmi_get_processor_handles()
                    
                    # 遍历探测到的所有 GPU
                    for i, handle in enumerate(devices):
                        if active_gpus is not None and i not in active_gpus:
                            continue
                        try:
                            power = amdsmi.amdsmi_get_power_info(handle).average_socket_power
                            vram = amdsmi.amdsmi_get_vram_usage(handle) / (1024 * 1024)
                            try:
                                activity = amdsmi.amdsmi_get_gpu_activity(handle)
                                if isinstance(activity, dict):
                                    util = activity.get('gfx_activity', 0)
                                else:
                                    util = getattr(activity, 'gfx_activity', 0)
                            except:
                                util = 0
                            f.write(f"{time.time():.2f},AMD_API,GPU_{i},{vram:.1f},{power:.1f},{util:.1f}\n")
                            amdsmi_success = True
                        except:
                            pass
                    f.flush()
                except Exception:
                    pass
                
                # 如果 amdsmi (Python API) 失败/不可用，尝试通过命令行调用 rocm-smi 工具解析文本
                if not amdsmi_success:
                    rocm_smi_success = False
                    try:
                        res = subprocess.check_output(["rocm-smi"], stderr=subprocess.STDOUT).decode()
                        for line in res.strip().split('\n'):
                            line = line.strip()
                            parts = line.split()
                            # 过滤出以数字开头的行（设备数据行，如 "0       2     0x744b, ..."）
                            if len(parts) >= 12 and parts[0].isdigit():
                                idx = parts[0]
                                try:
                                    idx_int = int(idx)
                                except:
                                    idx_int = -1
                                if active_gpus is not None and idx_int not in active_gpus:
                                    continue
                                    
                                gpu_util_str = parts[-1].replace('%', '')
                                vram_pct_str = parts[-2].replace('%', '')
                                power_str = parts[-11].replace('W', '')
                                
                                try:
                                    util = float(gpu_util_str)
                                except:
                                    util = 0.0
                                try:
                                    power = float(power_str)
                                except:
                                    power = 0.0
                                try:
                                    vram_pct = float(vram_pct_str)
                                    vram = (vram_pct / 100.0) * self.total_mem_mb
                                except:
                                    vram = 0.0
                                    
                                f.write(f"{time.time():.2f},ROCM_SMI,GPU_{idx},{vram:.1f},{power:.1f},{util:.1f}\n")
                                rocm_smi_success = True
                        f.flush()
                    except Exception:
                        pass
                        
                    # 如果 rocm-smi 命令行也失败，则最终尝试 nvidia-smi 命令行探测（兼容英伟达环境）
                    if not rocm_smi_success:
                        try:
                            res = subprocess.check_output(
                                ["nvidia-smi", "--query-gpu=index,memory.used,power.draw,utilization.gpu", "--format=csv,noheader,nounits"], 
                                stderr=subprocess.STDOUT
                            ).decode()
                            for line in res.strip().split('\n'):
                                parts = line.split(',')
                                if len(parts) == 4:
                                    idx, vram, power, util = parts
                                    try:
                                        idx_int = int(idx.strip())
                                    except:
                                        idx_int = -1
                                    if active_gpus is not None and idx_int not in active_gpus:
                                        continue
                                    f.write(f"{time.time():.2f},NVIDIA,GPU_{idx.strip()},{vram.strip()},{power.strip()},{util.strip()}\n")
                            f.flush()
                        except Exception:
                            # 所有监控工具均失败 (例如 CPU 环境)
                            pass
                
                time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

class ProfilerContext:
    def __init__(self, use_profiler=False, log_dir="outputs/profiling/tensorboard_traces"):
        self.use_profiler = use_profiler
        self.log_dir = log_dir
        self.prof = None

    def __enter__(self):
        if self.use_profiler:
            import torch
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.prof.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_profiler and self.prof:
            self.prof.stop()
