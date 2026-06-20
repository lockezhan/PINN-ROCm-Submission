import deepxde as dde
import numpy as np
import torch

def get_ns_equation_data(scale_factor="small"):
    nu = 0.05
    lam = 10.0 - np.sqrt(100.0 + 4 * (np.pi ** 2))
    
    def pde(x, y):
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        # 使用原生 PyTorch 一次性求出 u 对 x, y 的偏导数
        grad_u = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        du_x = grad_u[:, 0:1]
        du_y = grad_u[:, 1:2]
        
        # 使用原生 PyTorch 一次性求出 v 对 x, y 的偏导数
        grad_v = torch.autograd.grad(
            v, x, 
            grad_outputs=torch.ones_like(v),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        dv_x = grad_v[:, 0:1]
        dv_y = grad_v[:, 1:2]
        
        # 使用原生 PyTorch 一次性求出 p 对 x, y 的偏导数
        grad_p = torch.autograd.grad(
            p, x, 
            grad_outputs=torch.ones_like(p),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        dp_x = grad_p[:, 0:1]
        dp_y = grad_p[:, 1:2]
        
        # 二阶导数计算
        grad_ux = torch.autograd.grad(
            du_x, x, 
            grad_outputs=torch.ones_like(du_x),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        du_xx = grad_ux[:, 0:1]
        
        grad_uy = torch.autograd.grad(
            du_y, x, 
            grad_outputs=torch.ones_like(du_y),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        du_yy = grad_uy[:, 1:2]
        
        grad_vx = torch.autograd.grad(
            dv_x, x, 
            grad_outputs=torch.ones_like(dv_x),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        dv_xx = grad_vx[:, 0:1]
        
        grad_vy = torch.autograd.grad(
            dv_y, x, 
            grad_outputs=torch.ones_like(dv_y),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        dv_yy = grad_vy[:, 1:2]
        
        eq_u = u * du_x + v * du_y + dp_x - nu * (du_xx + du_yy)
        eq_v = u * dv_x + v * dv_y + dp_y - nu * (dv_xx + dv_yy)
        eq_mass = du_x + dv_y
        
        return [eq_u, eq_v, eq_mass]
        
    def u_func(x):
        return 1 - np.exp(lam * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])

    def v_func(x):
        return lam / (2 * np.pi) * np.exp(lam * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])

    def p_func(x):
        return 0.5 * (1 - np.exp(2 * lam * x[:, 0:1]))

    geom = dde.geometry.Rectangle([-0.5, -0.5], [1.0, 1.5])
    
    if scale_factor == "small":
        num_domain, num_boundary, num_test = 2000, 200, 2000
    elif scale_factor == "large":
        num_domain, num_boundary, num_test = 400000, 20000, 50000
    elif scale_factor == "extreme":
        num_domain, num_boundary, num_test = 1000000, 100000, 200000
    else:
        num_domain, num_boundary, num_test = 2000, 200, 2000
    
    return geom, pde, (u_func, v_func, p_func), num_domain, num_boundary, num_test
