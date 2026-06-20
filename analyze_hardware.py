import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot(out_dir="outputs"):
    log_file = os.path.join(out_dir, "profiling/hardware_metrics.log")
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
        return
        
    print(f"Parsing distributed hardware metrics from {log_file}...")
    
    # 使用 pandas 解析带标题的 CSV
    try:
        df = pd.read_csv(log_file)
        # 清理列名和数据中的空格
        df.columns = [c.strip() for c in df.columns]
        for col in ['Backend', 'GPU_ID']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    except Exception as e:
        print(f"Error parsing log: {e}")
        return
        
    if df.empty or 'GPU_ID' not in df.columns:
        print("No valid distributed hardware data found.")
        return
        
    # 自动探测平台后端，决定首选的 VRAM 上限线
    backend = "ROCM_SMI"
    if not df.empty and 'Backend' in df.columns:
        backend_series = df['Backend'].dropna()
        if not backend_series.empty:
            backend = backend_series.iloc[0]
    is_amd = "AMD" in backend or "ROCM" in backend
        
    # 1. 消除因中断重连导致的瞬时掉电/零利用率毛刺（忽略掉到0的数据）
    # 我们保留最初始始的 1 分钟作为合法预热，之后的 0 数据全部判定为断线重启带来的毛刺并剔除
    t_start = df['Time'].min()
    valid_mask = ((df['Time'] - t_start) <= 60) | (df['GPU_Util'] > 1) | (df['Power_W'] > 50)
    df = df[valid_mask]
    
    # 2. 消除因中断重连导致的时间跳变断层（将断点之间的长间隔剔除，使曲线完美衔接）
    df = df.sort_values(by=['Time', 'GPU_ID']).reset_index(drop=True)
    
    # 获取各个 GPU 的时间线
    unique_times = sorted(df['Time'].unique())
    time_mapping = {}
    cumulative_gap = 0.0
    gap_threshold = 300.0  # 如果采样间隔超过 5 分钟，视为断点续训
    
    if len(unique_times) > 0:
        time_mapping[unique_times[0]] = unique_times[0]
        for i in range(1, len(unique_times)):
            diff = unique_times[i] - unique_times[i-1]
            if diff > gap_threshold:
                cumulative_gap += (diff - 1.0) # 剔除空白，保留 1 秒的视觉连接
            time_mapping[unique_times[i]] = unique_times[i] - cumulative_gap
            
    df['Time'] = df['Time'].map(time_mapping)
    
    # 转换为相对分钟数
    t0 = df['Time'].min()
    df['Time_Min'] = (df['Time'] - t0) / 60.0
    
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 学术配色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 4. 识别阶段性分界点（利用率首次突破 20% 视为进入正式计算阶段）
    active_runs = df[df['GPU_Util'] > 20]
    if not active_runs.empty:
        warmup_end_time = active_runs['Time_Min'].min()
    else:
        warmup_end_time = 0.05  # 默认 3 秒左右
        
    # ======================================================
    # 核心重构：共享 X 轴的紧凑子图（3x1 Shared X-Axis Grid）
    # ======================================================
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 11), dpi=300)
    
    # --- (a) VRAM 子图 ---
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        vram_gb = group['VRAM_MB'] / 1024.0
        c = colors[i % len(colors)]
        axs[0].plot(group['Time_Min'], vram_gb, label=f"GPU {gpu_id.split('_')[-1]}", color=c, linewidth=1.5, alpha=0.85)
        
    # 3. 绘制 OOM 关键物理边界虚线
    max_vram_gb = df['VRAM_MB'].max() / 1024.0
    if is_amd:
        limit_y = 48.0
        label = "AMD W7900 VRAM Limit (48GB)"
        color = "darkblue"
    else:
        limit_y = 24.0
        label = "NVIDIA RTX 4090 VRAM Limit (24GB)"
        color = "r"
        
    axs[0].axhline(y=limit_y, color=color, linestyle='--', linewidth=1.2, alpha=0.85)
    axs[0].text(df['Time_Min'].max() * 0.98, limit_y + 0.5, label, color=color, fontsize=8, ha='right', va='bottom', fontweight='semibold')
    
    # 动态适应 Y 轴高度范围
    y_max = max(limit_y * 1.15, max_vram_gb * 1.15)
    axs[0].set_ylim(0, y_max)
        
    axs[0].set_ylabel("VRAM Usage [GB]", fontsize=11, fontweight='semibold')
    axs[0].set_title("(a) Memory Footprint & Physical Boundaries", fontsize=12, fontweight='bold', pad=8)
    axs[0].grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- (b) GPU 核心利用率子图 ---
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        c = colors[i % len(colors)]
        axs[1].plot(group['Time_Min'], group['GPU_Util'], color=c, linewidth=1.5, alpha=0.85)
    axs[1].set_ylabel("Core Utilization [%]", fontsize=11, fontweight='semibold')
    axs[1].set_title("(b) Computational Load (SM Utilization)", fontsize=12, fontweight='bold', pad=8)
    axs[1].set_ylim(-5, 105)
    axs[1].grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- (c) 功耗子图 ---
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        c = colors[i % len(colors)]
        axs[2].plot(group['Time_Min'], group['Power_W'], color=c, linewidth=1.5, alpha=0.85)
    axs[2].set_xlabel("Elapsed Time [min]", fontsize=11, fontweight='semibold')
    axs[2].set_ylabel("Power Draw [W]", fontsize=11, fontweight='semibold')
    axs[2].set_title("(c) Power Consumption", fontsize=12, fontweight='bold', pad=8)
    axs[2].grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    
    # --- 4. 阶段性阴影高亮（Shaded Regions / axvspan） ---
    for ax in axs:
        # 阶段 A：数据加载与静态图预热 (Warmup & Capture)
        ax.axvspan(0, warmup_end_time, color='grey', alpha=0.12)
        # 阶段 B：PINN 2D N-S Hessian 计算阶段
        ax.axvspan(warmup_end_time, df['Time_Min'].max(), color='green', alpha=0.05)
        
    # 添加阶段性文字标注（标在利用率子图中央，保持排版精美）
    axs[1].text(warmup_end_time / 2.0, 50, "Warmup &\nCapture", color='#555555', fontsize=8, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    axs[1].text((warmup_end_time + df['Time_Min'].max()) / 2.0, 50, "PINN 2D N-S Hessian Computation (CUDA Graph Replay)", color='darkgreen', fontsize=9, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 统一合并 Legend 到整张图表的最下方
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(8, len(labels)), bbox_to_anchor=(0.5, 0.01), fontsize=10)
    
    # 2. 紧凑子图布局调节，极度压缩垂直间距 (hspace)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12, bottom=0.11)
    
    academic_path = os.path.join(figures_dir, "hardware_academic_profile.png")
    plt.savefig(academic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================
    # 辅助单图：依然保留，供多场景剪裁使用
    # ======================================================
    # 1. VRAM
    plt.figure(figsize=(10, 5))
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        vram_gb = group['VRAM_MB'] / 1024.0
        c = colors[i % len(colors)]
        plt.plot(group['Time_Min'], vram_gb, label=f"VRAM ({gpu_id})", color=c, linewidth=2, alpha=0.8)
    plt.xlabel("Elapsed Time (min)", fontsize=11)
    plt.ylabel("VRAM Usage (GB)", fontsize=11)
    plt.title("VRAM Consumption over Time", fontsize=13, pad=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(os.path.join(figures_dir, "vram_usage.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Power
    plt.figure(figsize=(10, 5))
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        c = colors[i % len(colors)]
        plt.plot(group['Time_Min'], group['Power_W'], label=f"Power ({gpu_id})", color=c, linewidth=2, alpha=0.8)
    plt.xlabel("Elapsed Time (min)", fontsize=11)
    plt.ylabel("Power Draw (Watts)", fontsize=11)
    plt.title("Power Draw over Time", fontsize=13, pad=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(os.path.join(figures_dir, "power_usage.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. GPU Util
    plt.figure(figsize=(10, 5))
    for i, (gpu_id, group) in enumerate(df.groupby('GPU_ID')):
        c = colors[i % len(colors)]
        plt.plot(group['Time_Min'], group['GPU_Util'], label=f"Util ({gpu_id})", color=c, linewidth=2, alpha=0.8)
    plt.xlabel("Elapsed Time (min)", fontsize=11)
    plt.ylabel("GPU Core Utilization (%)", fontsize=11)
    plt.title("GPU Core Utilization over Time", fontsize=13, pad=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(os.path.join(figures_dir, "gpu_utilization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[OK] Hardware Profiler Analysis Complete!")
    print(f"  - [ACADEMIC COMPLEMENT] 3x1 Shared Grid: {academic_path}")
    print(f"  - Single VRAM Curve: {os.path.join(figures_dir, 'vram_usage.png')}")
    print(f"  - Single Power Curve: {os.path.join(figures_dir, 'power_usage.png')}")
    print(f"  - Single GPU Util Curve: {os.path.join(figures_dir, 'gpu_utilization.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze hardware metrics for PINN training")
    parser.add_argument("--dir", type=str, default="outputs", help="Directory containing the logs and where plots will be saved")
    args = parser.parse_args()
    analyze_and_plot(args.dir)
