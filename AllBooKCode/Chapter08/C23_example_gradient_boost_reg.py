"""
文件名: AllBooKCode/Chapter08/C23_example_gradient_boost_reg.py
创建时间: 2022/11/6 9:06 上午 星期日
作 者: @空字符 
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from C21_manual_gradient_boost_reg import objective_function
from C21_manual_gradient_boost_reg import negative_gradient
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
