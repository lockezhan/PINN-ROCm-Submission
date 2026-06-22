# 基于 AMD ROCm 平台优化的高性能分布式数据并行与 HIP 静态图二维纳维-斯托克斯方程 PINN 求解器

[English Version](./README.md)

本项目是一个工业级、针对 AMD ROCm 架构深度优化的物理信息神经网络 (PINN) 求解器，专门用于在最大化硬件效率下高效求解稳态二维不可压缩 Navier-Stokes 方程 (Kovasznay Flow 基准测试)。

该求解器框架专为 **AMD ROCm** 生态（如 Radeon PRO W7900 或 W7000 系列 GPU）进行底层硬件级优化，提供分布式数据并行 (DDP) 扩展以及 HIP 静态图 (HIP Graph) 加速。

---

## 📂 项目目录结构说明

```text
PINN-ROCm-Submission/
├── core/                         # 核心算法实现文件夹
│   ├── network.py                # 配置网络隐藏层结构及浮点参数
│   ├── pde_def.py                # 基于原生 Autograd 构建的二维 Navier-Stokes 方程残差约束
│   ├── trainer.py                # 主 DDP 训练循环、GPUDataLoader 及 HIP 静态图录制运行逻辑
│   ├── profiler.py               # 实时监控并记录显存、功耗及利用率等物理硬件指标的记录器
│   └── visualizer.py             # 流场预测场、误差场等学术图表的可视化脚本
├── outputs_extreme_8_float32/    # extreme 规模下的 8 卡分布式训练结果
├── outputs_large_4_float32/      # large 规模下的 4 卡分布式训练结果
├── outputs_large_8_float32/      # large 规模下的 8 卡分布式训练结果
├── outputs_large_single_float32/ # large 规模下的单卡训练结果
├── outputs_small_float32/        # small 规模下的单卡训练结果
├── Dockerfile_rocm               # 适用于 AMD ROCm 环境的 Docker 镜像容器配置文件
├── requirements_linux_rocm.txt   # 适用于 AMD ROCm 的 Pip 依赖包声明文件
├── run_and_push.sh               # 一键测试与自动性能分析脚本
├── main.py                       # 命令行参数解析与分布式 DDP 启动主入口
├── analyze_hardware.py           # 提取性能指标日志并一键绘制学术负载图表
├── plot_loss.py                  # 绘制 Loss 收敛历史曲线的辅助脚本
├── combined_ns_flow.png          # 各分支预测速度场与压力场的对比图
└── combined_loss_curve.png        # 各分支 Loss 收敛曲线对比图
```

---

## 🛠️ 安装依赖与 Docker 环境配置

本项目支持本地 Python 依赖安装或基于 Docker 容器的快速部署。

### 1. 本地 Python 环境安装
若直接在物理机环境运行，请确保系统已安装 **ROCm 驱动与 ROCm 版本的 PyTorch**，并在 Python 虚拟环境中执行依赖安装：
```bash
pip install -r requirements_linux_rocm.txt
```

### 2. Docker 容器化环境配置 (推荐比赛评测使用)
项目提供了适用于 AMD ROCm 显卡的容器配置文件 [Dockerfile_rocm](./Dockerfile_rocm)：

*   **构建 Docker 镜像**：
    ```bash
    docker build -f Dockerfile_rocm -t pinn-solver:rocm .
    ```
*   **启动容器运行（推荐评测复现使用，挂载输出目录以提取图表与日志）**：
    为了方便评测委员会直接在宿主机上查看生成的流场图、收敛曲线以及硬件监控日志，建议将宿主机的 `outputs` 目录挂载到容器中：
    ```bash
    # 1. 在宿主机上创建输出目录
    mkdir -p outputs

    # 2. 启动容器运行（自动挂载输出目录，并挂载 AMD 显卡计算与渲染设备）
    docker run -it --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --ipc=host \
      -v $(pwd)/outputs:/workspace/PINN_Framework/outputs \
      pinn-solver:rocm
    ```
    运行结束后，您可以在宿主机的 `./outputs/` 目录下直接查看到所有的流场预测图、Loss 收敛图和硬件指标日志。

---

## 🏃‍♂️ 比赛一键测试与常规运行指南

### 1. 一键测试基准脚本
项目根目录下提供了 [run_and_push.sh](./run_and_push.sh) 自动化一键测试脚本。该脚本会自动更新依赖、调度多卡 DDP 训练、清理旧的历史权重，并自动绘制硬件负载图。

*   **单卡一键测试**：
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale small --precision float32 --gpus 0
    ```
*   **多卡一键并行测试**（以 8 卡并行 DDP 训练为例）：
    ```bash
    chmod +x run_and_push.sh
    ./run_and_push.sh --scale large --precision float32 --gpus 0,1,2,3,4,5,6,7
    ```

### 2. 精细常规测试命令
若需手动微调训练参数，可直接执行 `main.py` 入口。请在运行命令前通过环境变量指定 GPU 绑定：

*   **高精度单卡运行 (Float32 满精度)**：
    ```bash
    HIP_VISIBLE_DEVICES=0 python main.py --scale small --precision float32 --epochs 2000 --out_dir outputs/small
    ```
*   **高性能单卡运行 (BFloat16 AMP 混合精度 + HIP Graph 静态图加速)**：
    ```bash
    HIP_VISIBLE_DEVICES=0 python main.py --scale large --precision bfloat16 --epochs 2000 --out_dir outputs/large_bf16
    ```
*   **多卡 DDP 并行精细运行**（以 4 卡为例）：
    ```bash
    HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --scale large --precision float32 --batch_size 200000 --epochs 2000 --out_dir outputs/large_4gpus
    ```

---

## 📖 数学物理方程

稳态二维不可压缩 Navier-Stokes 方程定义如下：

- x 方向动量方程：
$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
$$

- y 方向动量方程：
$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
$$

- 连续性方程（质量守恒）：
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

其中 运动粘度系数 $\nu = 0.05$，$u$ 和 $v$ 分别为速度分量，$p$ 为压力。

### Kovasznay Flow 基准测试
求解域为 $[-0.5, 1.0] \times [-0.5, 1.5]$。其解析解用于边界条件定义和误差精度验证：

$$
u_{true}(x, y) = 1 - e^{\lambda x} \cos(2\pi y)
$$

$$
v_{true}(x, y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y)
$$

$$
p_{true}(x, y) = \frac{1}{2} (1 - e^{2\lambda x})
$$

其中参数 $\lambda = 10 - \sqrt{100 + 4\pi^2}$。

---

## 📊 硬件性能监控与分析

框架运行期间会自动将显存占用、功耗和利用率等性能指标记录在 `${OUT_DIR}/profiling/hardware_metrics.log` 中。

可通过运行以下分析脚本一键绘制学术论文级别的负载变化图表：
```bash
python analyze_hardware.py --dir <输出目录路径>
```
这将在对应的 `figures/` 目录下生成：
*   **`hardware_academic_profile.png`**：包含显存 OOM 边界线及 Hessian 阶段的 3x1 学术监控大图。
*   **`ns_flow_field.png`**：二维 Kovasznay 流场预测对比图（PINN 预测值 vs 真实解析解）。
