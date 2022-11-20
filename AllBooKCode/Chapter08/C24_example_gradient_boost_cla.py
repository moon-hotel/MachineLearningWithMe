"""
文件名: AllBooKCode/Chapter08/C24_example_gradient_boost_cla.py
创建时间: 2022/11/6 8:47 上午 星期日
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from C22_manual_gradient_boost_cla import negative_gradient
from C22_manual_gradient_boost_cla import objective_function
import numpy as np


def example():
    y = np.array([2, 1, 2, 0, 0])
    K = 3
    M = 3
    learning_rate = 0.5
    Y = np.zeros((y.shape[0], K), dtype=np.float64)
    for k in range(K):
        Y[:, k] = y == k
    print(f"原始标签的one-hot编码结果为:\n {Y}")
    raw_predictions = np.array([[0, 0.1, 0.3],
                                [0.2, 0.2, 0.1],
                                [0.3, 0.2, 0.1],
                                [0, 0, 0.4],
                                [0, 0.1, 0]])  # [n_samples,n_classes]
    print(f"初始化所有样本的预测概率为:\n {raw_predictions}")
    print(f"此时的损失值为: {objective_function(y, raw_predictions, K)}")
    for m in range(M):
        for k in range(K):
            grad = negative_gradient(y, raw_predictions, k)
            print(f"第{m + 1}次提升时类别{k}下所有样本对应的负梯度为{grad}")
            raw_predictions[:, k] += learning_rate * grad  # 梯度下降更新预测概率
        print(f"第{m + 1}次提升后的损失值为: {objective_function(y, raw_predictions, K)}")
        print(f"第{m + 1}次提升后的预测概率为:\n {raw_predictions}")
        print(f"第{m + 1}次提升后的预测值为: {np.argmax(raw_predictions, 1)}")


if __name__ == '__main__':
    example()
