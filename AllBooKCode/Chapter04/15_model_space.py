import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 5, 100).reshape(-1, 1)
y1 = x ** 2 + 0.5 * x ** 3 + 0.2 * x ** 4 - 0.1 * x ** 5
y2 = 0.1 * x ** 2 + 0.05 * x ** 3 + 0.02 * x ** 4 - 0.01 * x ** 5
plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
plt.plot(x, y1, linestyle='--', label=r'$y_1=1.0x^2+0.5x^3+0.2x^4-0.1x^5$', c='black')
plt.plot(x, y2, label=r'$y_1=0.1x^2+0.05x^3+0.02x^4-0.01x^5$', c='black')
plt.legend(fontsize=12)
plt.show()
