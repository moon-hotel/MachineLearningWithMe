import numpy as np
import matplotlib.pyplot as plt


def l2_reg():
    w = np.linspace(-0.5, 2.5, 100)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    J1 = (w - 1) ** 2
    J2 = (w - 1) ** 2 + w ** 2
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.plot(w, J1, label=r'$J_1=(w-1)^2$', c='black')
    plt.scatter(1, 0, s=40, c='black')
    plt.plot(w, J2, linestyle='--', label=r'$J_2=(w-1)^2+w^2$', c='black')
    plt.xlabel('w', fontsize=15)
    plt.scatter(0.5, 0.5, s=40, c='black')
    plt.legend(fontsize=15)

    plt.subplot(1, 2, 2)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    w = np.linspace(-0.5, 2.5, 100)
    J1 = 2 * (w - 1) ** 4
    J2 = 2 * (w - 1) ** 4 + w ** 2
    plt.plot(w, J1, label=r'$J_1=2(w-1)^4$', c='black')
    plt.scatter(1, 0, s=40, c='black')
    plt.plot(w, J2, linestyle='--', label=r'$J_2=2(w-1)^4+w^2$', c='black')
    plt.scatter(0.5, 3 / 8, s=40, c='black')
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.show()


if __name__ == '__main__':
    l2_reg()
