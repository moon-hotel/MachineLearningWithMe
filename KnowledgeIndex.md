# 知识点与代码索引

## [0 环境配置](./00_Configuration/README.md)

- [如何才能入门机器学习？ ](https://mp.weixin.qq.com/s/fhWdDOmJWRb0_cUhR8NU0A)
  - 一个合适的方法往往能够事半功倍
- [优雅的安装和使用Anaconda](https://mp.weixin.qq.com/s/KOFvW5UpAzqJKchCkfv7JA)
  - Windows和Linux下安装Anaconda
  - 使用`Conda`来进行环境的创建与管理
  - 源替换与导出Python列表
- [Pycharm安装与使用](https://mp.weixin.qq.com/s/MY0B6tIF9jcONmc0puARig)
  - 配置运行环境

## [1 线性回归](./01_LinearRegression/README.md)

- [线性回归（模型的建立与求解）](https://mp.weixin.qq.com/s/pSr5EFNK2Lu9CvRZnFCGKQ)
  - 什么是线性回归
  - 模型的误差之目标函数
  - `pip install`命令的使用
  - `sklearn`建模房价预测 [示例1](01_LinearRegression/102_train.py)
- [线性回归（多变量与多项式回归）](https://mp.weixin.qq.com/s/QqjCzRIRkOydlwAsvWWLyQ)
  - 多变量回归 [示例1](01_LinearRegression/201_train.py)
  - 多项式回归与`PolynomialFeatuires`用法 [示例1](01_LinearRegression/202_train_pol.py)
- [线性回归（模型的评估）](https://mp.weixin.qq.com/s/Y4QMqv9EfdDDpQOo85y1aA)
  - 模型评估（MAE,MSE,RMSE,MAPE) [示例1](01_LinearRegression/302_metrics.py)
- [线性回归（梯度下降）](https://mp.weixin.qq.com/s/6Z4vcs_CtQYZxrU_JmSzqQ)
  - 梯度下降原理与实现 [示例1](01_LinearRegression/401_visualization.py)，[示例2](01_LinearRegression/402_init_weight.py)，[示例3](01_LinearRegression/403_gradient_descent.py)
- [神说，要有正态分布，于是就有了正态分布](https://mp.weixin.qq.com/s/1DiBXYGRCXZgmmEADsLI-w)
  - 正态分布的来历与作用
- [线性回归（目标函数的推导）](https://mp.weixin.qq.com/s/gzMEjhRNCHRekg7BY3VoXA)
  - 线性回归的推导与实现 [示例1](01_LinearRegression/405_normal_distribution.py)，[示例2](01_LinearRegression/406_boston_house_prediction.py)

## [2 逻辑回归](./02_LogisticRegression/README.md)

- [逻辑回归（模型的建立与求解）](https://mp.weixin.qq.com/s/QNds12K8v9tuHnOFcY-0ag)
  - 逻辑回归原理
  - 决策边界[示例1](02_LogisticRegression/102_decision_boundary.py)
- [逻辑回归（混淆矩阵与评估指标）](https://mp.weixin.qq.com/s/COz9WNXSIBij2x1-dtEoJA)
  - 混淆矩阵与`classification_report` [示例1](02_LogisticRegression/202_one_vs_all.py)，[示例2](02_LogisticRegression/203_one_vs_all_train.py)
  - One-vs-all与多分类思想
  - `sklearn`建模患癌预测
- `sklearn`建模患癌预测 [示例1](02_LogisticRegression/201_metrics.py)
- [逻辑回归（目标函数推导与实现）](https://mp.weixin.qq.com/s/g6x2o_FN3Ndi_RhDpxCJWg)
  - 目标函数与梯度推导
  - 动手实现二分类与多分类 [示例1](02_LogisticRegression/302_implementation.py)，[示例2](02_LogisticRegression/303_implementation_multi_class.py)
- [任务一（利用逻辑回归完成学生是否能被录取的二分类任务）](https://mp.weixin.qq.com/s/VTCAMZhuxhtwM-pl59c9oA)
  - `Pandas`读取数据集[示例1](02_LogisticRegression/304_task1.py)

## [3 模型的改善与泛化](./03_ModelOptimization/README.md)

- [模型的改善与泛化（标准化与特征映射）](https://mp.weixin.qq.com/s/DTddhHYrlehoaorDRxquJA)
  - 等高线与特征标准化
  - 特征组合与特征映射
  - 动手实现非线性分类器 [示例1](03_ModelOptimization/108_visualization_pol_cla.py)

- [模型的改善与泛化（梯度与等高线）](https://mp.weixin.qq.com/s/Eo-S8jm25TiZW82yRmsg4Q)
  - 梯度与等高线

- [模型的改善与泛化（过拟合）](https://mp.weixin.qq.com/s/uF2Zp90FayUu-YqHUcKuGw)
  - 过拟合与欠拟合
  - 训练集与测试集

- [模型的改善与泛化（正则化）](https://mp.weixin.qq.com/s/rvmdb16QTDi4euyanoC2-w)
  - 正则化原理

- [模型的改善与泛化（偏差方差与交叉验证）](https://mp.weixin.qq.com/s/REYf_FcHvNSASCuybeSenQ)
  - 偏差方差与超参数
  - 模型选择与交叉验证

- [模型的改善与泛化（手写体识别）](https://mp.weixin.qq.com/s/Hx0EmnvJqShuaCckIiYvew)
  - `StandardScaler()`与`KFold`的使用 [示例1](03_ModelOptimization/601_digits_classification.py)
  - `LogisticRegression`与`SGDClassifier`

## [4 K最近邻与朴素贝叶斯](./04_KNNAndNaiveBayes/README.md)

- [K近邻算法](https://mp.weixin.qq.com/s/HpgyaOffKNKw768sqxA6gQ)
  - KNN原理与sklearn建模 [示例1](04_KNNAndNaiveBayes/103_train.py)
  - 距离的度量方式
  - 网格搜索与并行搜索
  - `GridSearch`
- kd树
- [朴素贝叶斯算法](https://mp.weixin.qq.com/s/QOWBh77f_Dv0Fmj_mAk-Cw)
  - 朴素贝叶斯原理
  - 先验概率与后验概率
  - 拉普拉斯平滑
  - 贝叶斯估计

- [文本特征提取之词袋模型](https://mp.weixin.qq.com/s/oCql-n_B1zrjC5RCi0Vpew)
  - 词袋模型原理
  - 分词与词频统计
  - `jieba`与`Counter`

- [基于词袋模型的垃圾邮件分类](https://mp.weixin.qq.com/s/6J5vZSb53rEJ-ToPcMpYKQ)
  - `CountVectorizer`与文本数据预处理
  - 朴素贝叶斯分类示例 [示例1](04_KNNAndNaiveBayes/302_bag_of_word_cla.py)
  - `classification_report`

- [TF-IDF文本表示方法与词云图](https://mp.weixin.qq.com/s/ULyWHF2hJq3IMQrBCR7FzQ)
  - TFIDF原理与计算示例
  - `TfidfVectorizer`与停用词
  - `word cloud`与词云图
- [任务二（基于贝叶斯算法的新闻分类））](https://mp.weixin.qq.com/s/S4X2anVmhL4-CwP2Mo-T0Q)

## [5 决策树与随机森林](./05_DecisionTree/README.md)

- [这就是决策树的思想](https://mp.weixin.qq.com/s/y1-4Vs6v7zUjRMTrRhQiYA)
  - 决策树思想
  - 信息熵与信息增益
- [决策树的生成之ID3与C4.5](https://mp.weixin.qq.com/s/-Rirj5DiEV3CeoZomcE0MQ)
  - ID3与C4.5原理示例
- [决策树的建模与剪枝](https://mp.weixin.qq.com/s/42aNTQ1qth-XXfWKXU4H7Q)
  - sklearn接口介绍 
  - 决策树建模与可视化 [示例1](05_DecisionTree/201_decision_tree_ID3.py)
  - 剪枝思想
- [决策树的生成与剪枝CART](https://mp.weixin.qq.com/s/EIhE897boqlEItmmRo6tzA)
  - 基尼指数
  - CART分类决策树原理
- [集成模型：Bagging、Boosting和Stacking](https://mp.weixin.qq.com/s/fCP0u6kYnqvAaIsqkiQKQg)   
  - 集成学习思想
  - `BaggingClassifier`的使用 [示例1](05_DecisionTree/401_ensemble_bagging_knn.py)[示例2](05_DecisionTree/402_ensemble_bagging_dt.py)
- `StackingClassifier`的使用 [示例1](05_DecisionTree/403_ensemble_stacking.py)
- [随机森林在sklearn中的使用](https://mp.weixin.qq.com/s/56g3IQIPvCoHJ2NVeIFZFA)
  - `RandomForestClassifier`介绍
  - 特征重要性评估 [示例1](05_DecisionTree/501_random_forests.py)
- [泰坦尼克号沉船生还预测](https://mp.weixin.qq.com/s/K2i54YjoUGNWxgwJqSTQwg)
  - 缺失值补充
  - 特征值转换
  - `GridSearchCV`的使用 [示例1](05_DecisionTree/601_titanic.py)
- [多分类下的召回率与F值](https://mp.weixin.qq.com/s/pB7RX0Wnr2iLN7X-iAQgHg)
  - 准确率、精确率、召回率与F值

## [6 支持向量机](./06_SupportVectorMachine/README.md)

- [原来这就是支持向量机](https://mp.weixin.qq.com/s/R03hUHGDBgVLhOnXLmx2QA)
  - 函数间隔与几何间隔
  - 最大间隔分类器
- [从另一个角度看支持向量机](https://mp.weixin.qq.com/s/Vj0ZsrC-jEfoYy2Z04Vg_Q)
- [SVM中关于函数间隔为什么可以设为1](https://mp.weixin.qq.com/s/GgNOxmZKwMLHZyE9EifUHQ)
- [SVM之sklear建模与核技巧](https://mp.weixin.qq.com/s/33tdsLYz9vjB-oMaVMDnkQ)
  - `SVC`的使用 [示例1](06_SupportVectorMachine/301_linear_svm.py)
  - 线性不可分与特征映射
  - 核技巧与无穷维
- [SVM之软间隔最大化](https://mp.weixin.qq.com/s/5l10ErpurIdpEP6fRPE0rg)
  - 误差与惩罚

- [好久不见的拉格朗日乘数法](https://mp.weixin.qq.com/s/ZdPbghhWat9HFGV7AOh8-A)
  - 拉格朗日乘数法
- [对偶性与KKT条件](https://mp.weixin.qq.com/s/rhUIG5Yrkwqj83TuAki51w)
- [SVM之目标函数求解](https://mp.weixin.qq.com/s/lGlXBlt3X_Vedmpgf4M-Ig)

## [7 聚类](./07_Clustering/README.md)

- [Kmeans聚类算法](https://mp.weixin.qq.com/s/29KQDvxPNtosJ_gt6WgRkQ)
  - 有监督与无监督学习
  - 聚类算法的思想
  - Kmeans聚类原理
  - sklearn建模Kmeans
  - Kmeans目标函数
- [聚类与分类的区别是什么](https://mp.weixin.qq.com/s/l8jQHLjRj-1-c-_1GHcxiQ)
- [Kmeans聚类算法的优缺点以及改进方法](https://mp.weixin.qq.com/s/TuXkQPKjtUixrSdMMNHKww)
- [几种常见的聚类评估指标](https://mp.weixin.qq.com/s/A2HLa8ei2Ob84zVZq9CXmA)
  - Acc、F-score、ARI与NMI [示例1](07_Clustering/402_metrics.py)

- [Kmeans聚类算法求解与实现](https://mp.weixin.qq.com/s/luC4s4ZFLL2bLFc2u29vmA)
  - [封装实现](https://github.com/moon-hotel/Kmeans)
- [Kmeans++原理与实现](https://mp.weixin.qq.com/s/nQEtA5fPvGOnSHjvjc9VNg)
  - 簇中心初始化 [示例1](./303_kmeanspp.py)

- [WKmeans一种基于特征权重的聚类算法](https://mp.weixin.qq.com/s/OUussxPI6UERmsnFqPM4Sw)
  - 加权Kmean聚类算法 [封装实现](https://github.com/moon-hotel/WKmeans)



### [<主页>](./README.md) 