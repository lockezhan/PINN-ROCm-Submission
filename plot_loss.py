import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def ema(data, alpha=0.05):
    """Exponential Moving Average for smoothing."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def setup_nature_style():
    """Apply Nature-style plot aesthetics."""
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['legend.frameon'] = False

def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from existing loss.dat")
    parser.add_argument("--dir", type=str, default="outputs", help="Directory containing figures/loss.dat")
    args = parser.parse_args()
    
    loss_file = os.path.join(args.dir, "figures/loss.dat")
    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found. Make sure the training in this directory has completed.")
        return
        
    try:
        # 加载数据，跳过第一行表头
        data = np.loadtxt(loss_file, skiprows=1)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        epochs = data[:, 0]
        pde_losses = data[:, 1]
        bc_losses = data[:, 2]
        total_losses = pde_losses + bc_losses
        
        setup_nature_style()
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        
        # 配色方案 (Nature restrained palette)
        c_pde = '#3B4992' # Deep Blue
        c_bc = '#EE0000'  # Deep Red
        c_tot = '#008B45' # Deep Green
        
        # 1. 绘制原始数据的淡背景 (Raw data background)
        ax.semilogy(epochs, pde_losses, color=c_pde, alpha=0.15, linewidth=0.5)
        ax.semilogy(epochs, bc_losses, color=c_bc, alpha=0.15, linewidth=0.5)
        ax.semilogy(epochs, total_losses, color=c_tot, alpha=0.15, linewidth=0.5)
        
        # 2. 绘制平滑后的主曲线 (Smoothed Hero lines)
        alpha_smooth = 0.05
        ax.semilogy(epochs, ema(pde_losses, alpha_smooth), label='PDE Loss', color=c_pde, alpha=0.9, linewidth=1.5)
        ax.semilogy(epochs, ema(bc_losses, alpha_smooth), label='BC Loss', color=c_bc, alpha=0.9, linewidth=1.5)
        ax.semilogy(epochs, ema(total_losses, alpha_smooth), label='Total Loss', color=c_tot, alpha=1.0, linewidth=2.0)
        
        # 3. 美化边框与坐标轴 (Spine manipulation)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', length=4, width=1.0, direction='out')
        ax.tick_params(axis='both', which='minor', length=2, width=0.5, direction='out')
        
        ax.set_xlabel('Epoch', fontweight='bold', labelpad=6)
        ax.set_ylabel('Loss (Log Scale)', fontweight='bold', labelpad=6)
        ax.set_title('PINN Navier-Stokes Solver Convergence History', fontsize=11, fontweight='bold', pad=12)
        
        # 最小化网格线
        ax.grid(True, which="major", axis='y', linestyle='-', alpha=0.15)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        out_image = os.path.join(args.dir, "figures/loss_curve.png")
        fig.savefig(out_image, bbox_inches='tight', dpi=300, transparent=False)
        plt.close(fig)
        print(f"[OK] Loss curve successfully plotted and saved to: {out_image}")
    except Exception as e:
        print(f"Error loading or plotting loss: {e}")

if __name__ == "__main__":
    main()
