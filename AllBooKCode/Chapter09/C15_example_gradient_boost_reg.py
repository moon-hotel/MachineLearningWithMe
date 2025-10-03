"""
文件名: AllBooKCode/Chapter08/C23_example_gradient_boost_reg.py
创建时间: 2022/11/6 9:06 上午 星期日
作 者: @空字符 
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from C17_manual_gradient_boost_reg import objective_function
from C17_manual_gradient_boost_reg import negative_gradient
import numpy as np


def example():
    y = np.array([2., 4, 2.5, 1.8, 2.4])
    raw_predictions = np.array([1.6, 2.7, .5, 1.1, 0.9])
    M = 3
    learning_rate = 0.5
    print(f"初始化所有样本的预测结果为:\n {raw_predictions}")
    print(f"此时的损失值为: {objective_function(y, raw_predictions)}")
    for m in range(M):
        grad = negative_gradient(y, raw_predictions)
        print(f"第{m + 1}次提升时所有样本对应的负梯度为{grad}")
        raw_predictions += learning_rate * grad  # 梯度下降更新预测结果
        print(f"第{m + 1}次提升后的损失值为: {objective_function(y, raw_predictions)}")
        print(f"第{m + 1}次提升后的预测值为: {raw_predictions}")
        print("=" * 20)


if __name__ == '__main__':
    example()
    # 初始化所有样本的预测结果为:
    #  [1.6 2.7 0.5 1.1 0.9]
    # 此时的损失值为: [0.08  0.845 2.    0.245 1.125]
    # 第1次提升时所有样本对应的负梯度为[0.4 1.3 2.  0.7 1.5]
    # 第1次提升后的损失值为: [0.02    0.21125 0.5     0.06125 0.28125]
    # 第1次提升后的预测值为: [1.8  3.35 1.5  1.45 1.65]
    # ====================
    # 第2次提升时所有样本对应的负梯度为[0.2  0.65 1.   0.35 0.75]
    # 第2次提升后的损失值为: [0.005     0.0528125 0.125     0.0153125 0.0703125]
    # 第2次提升后的预测值为: [1.9   3.675 2.    1.625 2.025]
    # ====================
    # 第3次提升时所有样本对应的负梯度为[0.1   0.325 0.5   0.175 0.375]
    # 第3次提升后的损失值为: [0.00125    0.01320313 0.03125    0.00382813 0.01757812]
    # 第3次提升后的预测值为: [1.95   3.8375 2.25   1.7125 2.2125]
    # ====================
