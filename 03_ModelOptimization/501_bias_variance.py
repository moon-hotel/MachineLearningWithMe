import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

#创建画布
fig = plt.figure()


#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)

#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
#"-|>"代表实心箭头："->"代表空心箭头

ax.axis["bottom"].set_axisline_style("->", size = 2)
ax.axis["left"].set_axisline_style("->", size = 2)


#通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)
plt.xlabel("Model Complexity", fontsize=12)
plt.ylabel("Error", fontsize=12)

a = b = np.arange(0, 3, .02)
c = 0.9 * np.exp(a)
d = c[::-1]

ax.plot(a, c, 'k', c='b')
ax.plot(a, d, 'k', c='r')
ax.plot(a, c + d, 'k', c='black')

plt.annotate('Variance', xy=(2.5, 9), fontsize=15, c='b')
plt.annotate(r'$(Bias)^2$', xy=(2.5, 2), fontsize=15, c='r')
plt.annotate('Total Error', xy=(2, 15), fontsize=15, c='black')
plt.vlines(1.5, 0, 20, linestyles='--')
plt.annotate('Optimal Model', xy=(1.3, 10.2), fontsize=15, c='black', rotation=90)

plt.xticks([])
plt.yticks([])
plt.show()
