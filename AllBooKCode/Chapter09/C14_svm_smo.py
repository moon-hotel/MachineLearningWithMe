import numpy as np


def compute_w(data_x, data_y, alphas):
    p1 = data_y.reshape(-1, 1) * data_x
    p2 = alphas.reshape(-1, 1) * p1
    return np.sum(p2, axis=0)

    # r = np.sum(data_x*alphas*data_y,axis=0)
    # return r


def kernel(x1, x2):
    return np.dot(x1, x2)


def f_x(data_x, data_y, alphas, x, b):
    """

    :param data_x:  shape (m,n)
    :param data_y:  shape (m,)
    :param alphas:  shape (m,)
    :param x:       shape (n,)
    :param b:       shape (1,)
    :return:
    """
    k = kernel(data_x, x)
    r = alphas * data_y * k
    return np.sum(r) + b


def compute_L_H(C, alpha_i, alpha_j, y_i, y_j):
    L = np.max((0., alpha_j - alpha_i))
    H = np.min((C, C + alpha_j - alpha_i))
    if y_i == y_j:
        L = np.max((0., alpha_i + alpha_j - C))
        H = np.min((C, alpha_i + alpha_j))
    return L, H


def compute_eta(x_i, x_j):
    return 2 * kernel(x_i, x_j) - kernel(x_i, x_i) - kernel(x_j, x_j)


def compute_E_k(f_x_k, y_k):
    return f_x_k - y_k


def clip_alpha_j(alpha_j, H, L):
    if alpha_j > H:
        return H
    if alpha_j < L:
        return L
    return alpha_j


def compute_alpha_j(alpha_j, E_i, E_j, y_j, eta):
    return alpha_j - (y_j * (E_i - E_j) / eta)


def compute_alpha_i(alpha_i, y_i, y_j, alpha_j, alpha_old_j):
    return alpha_i + y_i * y_j * (alpha_old_j - alpha_j)


def compute_b1(b, E_i, y_i, alpha_i, alpha_old_i, x_i, y_j, alpha_j, alpha_j_old, x_j):
    p1 = b - E_i - y_i * (alpha_i - alpha_old_i) * kernel(x_i, x_i)
    p2 = y_j * (alpha_j - alpha_j_old) * kernel(x_i, x_j)
    return p1 - p2


def compute_b2(b, E_j, y_i, alpha_i, alpha_old_i, x_i, x_j, y_j, alpha_j, alpha_j_old):
    p1 = b - E_j - y_i * (alpha_i - alpha_old_i) * kernel(x_i, x_j)
    p2 = y_j * (alpha_j - alpha_j_old) * kernel(x_j, x_j)
    return p1 - p2


def clip_b(alpha_i, alpha_j, b1, b2, C):
    if alpha_i > 0 and alpha_i < C:
        return b1
    if alpha_j > 0 and alpha_j < C:
        return b2
    return (b1 + b2) / 2


def select_j(i, m):
    j = np.random.randint(m)
    while i == j:
        j = np.random.randint(m)
    return j


def smo(C, tol, max_passes, data_x, data_y):
    m, n = data_x.shape
    b, passes = 0., 0
    alphas = np.zeros(shape=(m))
    alphas_old = np.zeros(shape=(m))
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            x_i, y_i, alpha_i = data_x[i], data_y[i], alphas[i]
            f_x_i = f_x(data_x, data_y, alphas, x_i, b)
            E_i = compute_E_k(f_x_i, y_i)
            if ((y_i * E_i < -tol and alpha_i < C) or (y_i * E_i > tol and alpha_i > 0.)):
                j = select_j(i, m)
                x_j, y_j, alpha_j = data_x[j], data_y[j], alphas[j]
                f_x_j = f_x(data_x, data_y, alphas, x_j, b)
                E_j = compute_E_k(f_x_j, y_j)
                alphas_old[i], alphas_old[j] = alpha_i, alpha_j
                L, H = compute_L_H(C, alpha_i, alpha_j, y_i, y_j)
                if L == H:
                    continue
                eta = compute_eta(x_i, x_j)
                if eta >= 0:
                    continue
                alpha_j = compute_alpha_j(alpha_j, E_i, E_j, y_j, eta)
                alpha_j = clip_alpha_j(alpha_j, H, L)
                alphas[j] = alpha_j
                if np.abs(alpha_j - alphas_old[j]) < 10e-5:
                    continue
                alpha_i = compute_alpha_i(alpha_i, y_i, y_j, alpha_j, alphas_old[j])
                b1 = compute_b1(b, E_i, y_i, alpha_i, alphas_old[i], x_i, y_i, alpha_j, alphas_old[j], x_j)
                b2 = compute_b2(b, E_j, y_i, alpha_i, alphas_old[i], x_i, x_j, y_j, alpha_j, alphas_old[j])
                b = clip_b(alpha_i, alpha_j, b1, b2, C)
                num_changed_alphas += 1
                alphas[i] = alpha_i

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alphas, b


if __name__ == '__main__':
    data_x = np.array([[5, 1], [0, 2], [1, 5], [3., 2], [1, 2], [3, 5], [1.5, 6], [4.5, 6], [0, 7]])
    data_y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])
    alphas, b = smo(C=.2, tol=0.001, max_passes=200, data_x=data_x, data_y=data_y)
    print(alphas)  # [0.   0.   0.2   0.142   0.   0.2   0.142   0.   0.]
    print(b)  # 2.66
    w = compute_w(data_x, data_y, alphas)
    print(w)  # [-0.186,  -0.569]
