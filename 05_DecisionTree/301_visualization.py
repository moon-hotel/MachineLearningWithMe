import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0, 1, 5000)
q = 1 - p
h = -1. * (p * np.log2(p) + q * np.log2(q))
entropy = 0.5 * h
gini = 2 * p * (1 - p)
error = np.linspace(0, 0.5, 150)
plt.plot(error, error, 'black', label='分类误差')
error = np.linspace(0.5, 1, 150)
plt.plot(error, 1 - error, 'black')
plt.plot(p, entropy, label='熵之半')
plt.plot(p, gini, label='基尼指数')
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来  正常显示中文标签
plt.legend(fontsize=14)
# plt.tight_layout()  # 调整子图间距
plt.show()
