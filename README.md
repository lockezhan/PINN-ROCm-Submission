# High-Performance DDP & CUDA-Graph PINN Solver for 2D Navier-Stokes

This project is an industry-grade, highly optimized Physics-Informed Neural Network (PINN) solver designed to solve the steady-state 2D Navier-Stokes equations (Kovasznay Flow benchmark) with maximum hardware efficiency. 

The framework is optimized for both **NVIDIA CUDA** (e.g., RTX 4090 D) and **AMD ROCm** (e.g., Radeon PRO W7900) architectures, featuring Distributed Data Parallel (DDP) scaling and CUDA Graphs static acceleration.

---

## 📖 Mathematical Formulation

The steady-state 2D incompressible Navier-Stokes equations are defined as:

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0$$

$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0$$

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

where $\nu = 0.05$ is the kinematic viscosity, and $u, v, p$ are the velocity components and pressure.

### Kovasznay Flow Benchmark
The analytical solution (Kovasznay Flow) on the domain $[-0.5, 1.0] \times [-0.5, 1.5]$ is used for boundary conditions and error validation:

$$u_{true}(x,y) = 1 - e^{\lambda x} \cos(2\pi y)$$

$$v_{true}(x,y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y)$$

$$p_{true}(x,y) = \frac{1}{2} (1 - e^{2\lambda x})$$

where $\lambda = 10 - \sqrt{100 + 4\pi^2}$.

---

## ⚡ High-Performance HPC Engineering Highlights

Standard PINN implementations suffer from severe CPU-bound bottlenecks due to the complex computational graphs required for second-order derivatives. This framework implements five key optimizations to bypass these bottlenecks:

### 1. CUDA Graphs Static Recording
- **Problem**: In PyTorch eager mode, computing Hessians launches hundreds of tiny GPU kernels sequentially. The CPU-GPU kernel launch latency dominates, leaving the GPU underutilized (often <10% core load).
- **Solution**: For single-GPU full-batch training, the entire forward, double-backward (Hessians), and backward pass are recorded as a static GPU execution sequence (**CUDA Graph**). During training, the CPU calls a single execution hook `g.replay()`, allowing the GPU to run back-to-back at hardware speed. This increases SM utilization from <20% to 90%+ and increases training speed by 3x–10x.

### 2. GPUDataLoader (Zero-Copy Data Engine)
- **Problem**: Re-creating PyTorch `DataLoader` iterators at every epoch introduces major CPU overhead.
- **Solution**: We bypass PyTorch's native `DataLoader` with `GPUDataLoader`. All collocation points are stored in GPU memory. When the batch size covers the dataset (full-batch), it triggers a **zero-copy pointer bypass**, yielding the original tensor directly without indexing or memory copying. For mini-batching, index shuffling (`torch.randperm`) is performed entirely on the GPU.

### 3. Native PyTorch Autograd Optimization
- **Problem**: DeepXDE wrappers cache gradients in global python dictionaries, creating significant Python overhead.
- **Solution**: We calculate all derivatives using pure PyTorch `torch.autograd.grad` directly. By using `grad_outputs=torch.ones_like(u)`, we obtain both $\frac{\partial u}{\partial x}$ and $\frac{\partial u}{\partial y}$ in a single backward pass, halving the autograd calls.

### 4. Dynamic Precision Control (FP32 vs. BF16 AMP)
- **BFloat16 AMP**: Speeds up matrix multiplications by utilizing Tensor Cores. It keeps the dynamic range of Float32 (exponent) to prevent the Hessian underflow common in standard FP16.
- **Float32 Eager**: If high physical accuracy is required, passing `--precision float32` disables mixed precision. This prevents quantization noise from propagating through second-order derivatives, lowering the maximum velocity error (e.g. from 0.029 to under 0.005).

### 5. Multi-GPU Distributed Data Parallel (DDP)
- Supports distributed training over multiple GPUs (up to 8 cards) via PyTorch DDP.
- Employs a custom seed-reproducible `DistributedSampler` directly in the GPU loader to partition collocation points across ranks, avoiding duplication.

---

## 🛠️ Installation & Environment Setup

Select the requirements file based on your OS and GPU backend:

### Windows (NVIDIA CUDA)
1. Install CUDA Toolkit 12.1+.
2. Install Python 3.10+.
3. Run:
   ```bash
   pip install -r requirements_win_cuda.txt
   ```

### Linux (AMD ROCm / ROCm 6.0+)
1. Ensure the ROCm kernel module and driver are active.
2. Install PyTorch with ROCm support.
3. Run:
   ```bash
   pip install -r requirements_linux_rocm.txt
   ```

---

## 🏃‍♂️ Running Guide

### 1. Single-GPU Local Run (e.g. RTX 4090 D)
Run with different precision modes:
- **High Accuracy Mode (Float32)**:
  ```bash
  python main.py --scale large --batch_size 200000 --precision float32
  ```
- **High Throughput Mode (BFloat16 - CUDA Graphs enabled)**:
  ```bash
  python main.py --scale large --batch_size 200000 --precision bfloat16
  ```

### 2. Multi-GPU DDP Run (e.g. 8x Radeon PRO W7900)
Use `torchrun` to scale across all available GPUs:
```bash
torchrun --nproc_per_node=8 main.py --scale extreme --precision float32 --batch_size 16384
```

---

## 📊 Academic Profiling & Analysis

The framework logs hardware performance metrics (VRAM, Power, SM Util) to `outputs/profiling/hardware_metrics.log`.

To generate publication-ready academic plots, run:
```bash
python analyze_hardware.py
```
This generates:
* **`outputs/figures/hardware_academic_profile.png`**: A consolidated 3x1 multi-panel plot showing VRAM limits (OOM boundary lines) and training stages (Warmup vs. Hessian computation) in line with NeurIPS/IEEE paper standards.
* **Individual Metric Curves**: `vram_usage.png`, `power_usage.png`, and `gpu_utilization.png`.

---

## 📂 Sub-Project Structure

- `core/`
  - `network.py`: Configures the FNN layers and floating-point parameters.
  - `pde_def.py`: Contains the 2D Navier-Stokes equation using native Autograd.
  - `trainer.py`: Holds the main DDP loop, GPUDataLoader, and CUDA Graphs logic.
  - `profiler.py`: Records real-time system metrics (NVIDIA and AMD).
  - `visualizer.py`: Visualizes prediction flow fields and error distributions.
- `main.py`: Entry point handling parsing and DDP initialization.
- `analyze_hardware.py`: Plots academic hardware logs.
