import deepxde as dde

def build_network(scale_factor="small", precision="float32"):
    """
    构建 2D N-S 方程的 FNN 网络
    输入维度：2 (x, y)
    输出维度：3 (u, v, p)
    """
    # 全面使用用户指定的精度（防范 Hessian Underflow 陷阱）
    # 在命令行中若指定 bfloat16，DeepXDE 会将其配置到底层
    dde.config.set_default_float(precision)
    
    # 根据 scale 膨胀网络，配合点数构成 3D 组合拳压榨显存
    if scale_factor == "small":
        layer_size = [2, 64, 64, 64, 64, 3]
    elif scale_factor == "large":
        layer_size = [2, 256, 256, 256, 256, 256, 3]
    elif scale_factor == "extreme":
        # 极限宽深模型，8 个 512 的隐藏层，参数量激增，显存占用极大
        layer_size = [2, 512, 512, 512, 512, 512, 512, 512, 512, 3]
    else:
        layer_size = [2, 64, 64, 64, 64, 3]
        
    activation = "tanh"
    initializer = "Glorot normal"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    return net
