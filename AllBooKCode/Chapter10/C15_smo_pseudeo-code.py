



初始化所有 alpha_i = 0, b = 0, passes = 0
while (passes < max_passes)
    num_changed_alphas = 0
    for i = 1,...,m
        计算 E_i
        if ((y_i*E_i < -tol and a_i < C)||(y_i*E_i > tol and alpha_i > 0))
            随机选择j，且j不等于i
            计算 E_j
            保存：alpha_i_old = alpha_i,alpha_j_old = alpha_j
            计算 L 和 H
            if (L == H):
                continue
            计算 eta
            if (eta >- 0):
                continue
            计算 alpha_j并裁剪
            if (|alpha_j - alpha_j_old| < 10e-5):
                continue
            分别计算alpha_i, b_1, b_2
            计算b
            num_changed_alphas += 1
    if (num_changed_alphas == 0):
        passes += 1
    else:
        passes = 0
