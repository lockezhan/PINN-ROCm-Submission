# HPC-optimized PINN Solver on AMD ROCm for 2D Navier-Stokes

[简体中文说明入口](./README_CN.md)

This repository contains an industry-grade, highly optimized Physics-Informed Neural Network (PINN) solver designed to solve the steady-state 2D incompressible Navier-Stokes equations (Kovasznay Flow benchmark) with maximum hardware efficiency.

The solver framework is specifically tailored for the **AMD ROCm** ecosystem (e.g., Radeon PRO W7900 or W7000 series GPUs), utilizing Distributed Data Parallel (DDP) scaling and HIP static graph execution (HIP Graph) acceleration.

---

## 📂 Repository Structure

```text
PINN-ROCm-Submission/
├── core/                         # Core implementation of the algorithm
│   ├── network.py                # FNN architecture and float configuration
│   ├── pde_def.py                # Navier-Stokes residual formulation using native Autograd
│   ├── trainer.py                # Main DDP training loops, GPUDataLoader, and HIP Graph execution
│   ├── profiler.py               # Real-time hardware telemetry recorder (VRAM, Power, SM Util)
│   └── visualizer.py             # Vis field and error distribution plotting script
├── outputs_extreme_8_float32/    # Training results (extreme scale, 8 GPUs)
├── outputs_large_4_float32/      # Training results (large scale, 4 GPUs)
├── outputs_large_8_float32/      # Training results (large scale, 8 GPUs)
├── outputs_large_single_float32/ # Training results (large scale, 1 GPU)
├── outputs_small_float32/        # Training results (small scale, 1 GPU)
├── Dockerfile_rocm               # Docker container specification for AMD ROCm env
├── requirements_linux_rocm.txt   # Pip package requirements for AMD ROCm
├── run_and_push.sh               # One-click benchmark and execution script
├── main.py                       # Project CLI and distributed entrypoint
├── analyze_hardware.py           # Evaluates telemetry log and plots academic figures
├── plot_loss.py                  # Utility script to plot Loss history
├── combined_ns_flow.png          # Visualized velocity and pressure field comparison
└── combined_loss_curve.png        # Convergence loss curves comparison
```

---

## 🛠️ Installation & Environment Setup

This solver supports both native installation on a host machine and containerized deployment using Docker.

### 1. Local Pip Installation (Non-Containerized)
Ensure your host machine has ROCm drivers and ROCm-compatible PyTorch installed, then run:
```bash
pip install -r requirements_linux_rocm.txt
```

### 2. Docker Container Setup (Recommended for Evaluation)
To easily reproduce the results in an isolated container environment, we provide an AMD ROCm Dockerfile [Dockerfile_rocm](./Dockerfile_rocm).

*   **Build the Docker Image**:
    ```bash
    docker build -f Dockerfile_rocm -t pinn-solver:rocm .
    ```
*   **Run the Container** (binds the AMD host graphics devices and maps the `/outputs` directory to extract logs):
    ```bash
    # 1. Create a local output directory on your host
    mkdir -p outputs

    # 2. Run the container with GPU devices and directory mapping
    docker run -it --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --ipc=host \
      -v $(pwd)/outputs:/workspace/PINN_Framework/outputs \
      pinn-solver:rocm
    ```
    Once the training completes, you can directly access the visualized figures and profiling metrics under the `./outputs/` directory on your host machine.

---

## 🏃‍♂️ Execution & Benchmark Guide

### 1. One-Click Benchmark Script
We provide a unified, automated benchmarking script [run_and_push.sh](./run_and_push.sh) to update dependencies, coordinate single/multi-card training, clean up redundant checkpoints, and generate profiling plots.

*   **Single-GPU Benchmark**:
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale small --precision float32 --gpus 0
    ```
*   **Multi-GPU Parallel Benchmark** (e.g., 8-GPU DDP training):
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale large --precision float32 --gpus 0,1,2,3,4,5,6,7
    ```

### 2. Fine-grained CLI Command Execution
To run training directly using `main.py`, make sure to pass the GPU bindings via the environment variables `HIP_VISIBLE_DEVICES` (for AMD ROCm) or `CUDA_VISIBLE_DEVICES` (for NVIDIA CUDA):

*   **High-Accuracy Single-GPU Mode (Float32)**:
    ```bash
    HIP_VISIBLE_DEVICES=0 python main.py --scale small --precision float32 --epochs 2000 --out_dir outputs/small
    ```
*   **High-Performance Single-GPU Mode (BFloat16 + HIP Graph Acceleration)**:
    ```bash
    HIP_VISIBLE_DEVICES=0 python main.py --scale large --precision bfloat16 --epochs 2000 --out_dir outputs/large_bf16
    ```
*   **Multi-GPU DDP Distributed Parallel Mode** (e.g., 4 GPUs):
    ```bash
    HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --scale large --precision float32 --batch_size 200000 --epochs 2000 --out_dir outputs/large_4gpus
    ```

---

## 📖 Mathematical Formulation

Steady-state 2D incompressible Navier-Stokes equations:

- x-direction momentum equation:
$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
$$

- y-direction momentum equation:
$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
$$

- Continuity equation:
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

where kinematic viscosity $\nu = 0.05$, $u$ and $v$ represent the velocity fields, and $p$ represents the pressure field.

### Kovasznay Flow Benchmark
Analytical solutions for boundaries and error validation:

$$
u_{true}(x, y) = 1 - e^{\lambda x} \cos(2\pi y)
$$

$$
v_{true}(x, y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y)
$$

$$
p_{true}(x, y) = \frac{1}{2} (1 - e^{2\lambda x})
$$

where parameter $\lambda = 10 - \sqrt{100 + 4\pi^2}$.

---

## 📊 Telemetry Profiling & Visualizations

The framework automatically saves VRAM occupancy, power consumption, and GPU utilization metrics inside `${OUT_DIR}/profiling/hardware_metrics.log`.

To plot NeurIPS/IEEE paper-standard academic diagrams evaluating hardware footprints, run:
```bash
python analyze_hardware.py --dir <OUTPUT_DIRECTORY>
```
This produces the following figures under the `figures/` subfolder:
- **`hardware_academic_profile.png`**: Consolidated 3x1 metrics plot illustrating hardware limits, Hessian computation bottlenecks, and warmup stages.
- **`ns_flow_field.png`**: 2D visualized Kovasznay flow fields comparing PINN predictions with analytical truths.
