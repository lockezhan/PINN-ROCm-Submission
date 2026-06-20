import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_ns_results(net, funcs, X_test, out_dir="outputs"):
    """
    使用传入 of X_test (测试点坐标) 和 net (已训练好的网络，已解除 DDP 包装并开启 eval) 来绘制 2D 流场。
    funcs: (u_func, v_func, p_func) 解析解函数
    """
    print("Generating Flow Field visualizations...")
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    
    # 提取函数
    u_func, v_func, p_func = funcs
    
    # 因为 x_test 是一维离散点，为了画 2D 热力图，我们需要生成规则的网格 (Grid)
    x = np.linspace(-0.5, 1.0, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 拉平成 Nx2 供网络推理
    grid_pts = np.vstack((np.ravel(X), np.ravel(Y))).T
    
    # 转移到 GPU 并进行推理
    device = next(net.parameters()).device
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        y_pred = net(grid_tensor).cpu().numpy()
        
    u_pred = y_pred[:, 0]
    v_pred = y_pred[:, 1]
    p_pred = y_pred[:, 2]
    
    # 计算真实解
    u_true = u_func(grid_pts).flatten()
    v_true = v_func(grid_pts).flatten()
    
    # 速度幅值 magnitude = sqrt(u^2 + v^2)
    mag_pred = np.sqrt(u_pred**2 + v_pred**2).reshape(100, 100)
    mag_true = np.sqrt(u_true**2 + v_true**2).reshape(100, 100)
    p_pred_map = p_pred.reshape(100, 100)
    error_map = np.abs(mag_pred - mag_true)

    # 绘图四象限
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pred Vel
    c1 = axs[0, 0].contourf(X, Y, mag_pred, 50, cmap='jet')
    axs[0, 0].set_title('Predicted Velocity Magnitude')
    fig.colorbar(c1, ax=axs[0, 0])
    
    # True Vel
    c2 = axs[0, 1].contourf(X, Y, mag_true, 50, cmap='jet')
    axs[0, 1].set_title('True Velocity Magnitude')
    fig.colorbar(c2, ax=axs[0, 1])
    
    # Pred Pressure
    c3 = axs[1, 0].contourf(X, Y, p_pred_map, 50, cmap='coolwarm')
    axs[1, 0].set_title('Predicted Pressure Field')
    fig.colorbar(c3, ax=axs[1, 0])
    
    # Absolute Error
    c4 = axs[1, 1].contourf(X, Y, error_map, 50, cmap='magma')
    axs[1, 1].set_title('Absolute Error in Velocity')
    fig.colorbar(c4, ax=axs[1, 1])

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figures/ns_flow_field.png', dpi=300)
    plt.close()

def plot_loss_curve(loss_history, out_dir="outputs"):
    """
    绘制并保存 Loss 随 Epoch 变化的收敛曲线。
    """
    if not loss_history:
        return
        
    epochs = [item[0] for item in loss_history]
    pde_losses = [item[1] for item in loss_history]
    bc_losses = [item[2] for item in loss_history]
    total_losses = [p + b for p, b in zip(pde_losses, bc_losses)]
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.semilogy(epochs, pde_losses, label='PDE Loss', color='#1f77b4', alpha=0.8, linewidth=1.5)
    plt.semilogy(epochs, bc_losses, label='BC Loss', color='#ff7f0e', alpha=0.8, linewidth=1.5)
    plt.semilogy(epochs, total_losses, label='Total Loss', color='#2ca02c', alpha=0.9, linewidth=2.0)
    
    plt.xlabel('Epoch', fontsize=11, fontweight='semibold')
    plt.ylabel('Loss (Log Scale)', fontsize=11, fontweight='semibold')
    plt.title('PINN Navier-Stokes Solver Convergence History', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(fontsize=10, loc='upper right')
    
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    plt.savefig(f"{out_dir}/figures/loss_curve.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Loss curve saved to {out_dir}/figures/loss_curve.png")
