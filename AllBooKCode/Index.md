# 知识点与代码索引

### [第一章：机器学习环境安装](./Chapter01)

- `Conda`安装与虚拟环境管理
- `pip`源与`Conda`源替换
- `PyCharm`安装与使用

### [第二章：从零认识线性回归](./Chapter02/README-zh-CN.md)

- `sklearn`库安装与线性回归建模`LinearRegression`模块
- 多变量回归、多项式回归与`PolynomialFeatures`模块
- 常见回归评价指标与实现（MAE,MSE,RMSE,MAPE,R2)
- 方向导数与梯度下降可视化
- 正态分布与目标函数
- 从零实现线性回归
- 数据集与`load_boston`

### [第三章：从零认识逻辑回归](./Chapter03/README-zh-CN.md)

- 理解逻辑回归`LogisticRegression`
- `load_iris`与`make_blobs`构造模拟数据集
- 可视化二分类决策边界
- 多变量与多分类逻辑回归
- 准确率、精确率、召回率与F值
- 逻辑回归目标函数
- 从零实现逻辑回归yu与`load_breast_cancer`
- `classification_report`与`confusion_matrix`

### [第四章：模型的改善与泛化](./Chapter04/README-zh-CN.md)

- 机器学习概念
- 等高线与特征标准化
- 过拟合与正则化
- 偏差方差与交叉验证
- 准确率与召回率区别
- Precision-Recall曲线与AUC
- 等高线与梯度下降可视化

### [第五章：K近邻算法与原理](./Chapter05/README-zh-CN.md)

- K值选取与可视化
- sklearn接口介绍与`KNeighborsClassifier`
- 特征标准化方法与`StandardScaler`
- kd树原理与构造
- 交叉验证`train_test_split`与网格搜索`GridSearchCV`
- 最近邻与K近邻搜索原理
- kd树实现与搜索`KDTree`
- 从零实现KNN算法

### [第六章：文本特征提取与模型复用](Chapter06/README-zh-CN.md)

- 分词`jieba`与词频统计`Counter`
- 词袋模型`CountVectorizer`与TFIDF`TfidfTransformer`
- 基于朴素贝叶斯的垃圾邮件分类模型
- sklearn中模型的复用
- 词云图`WordCloud`

### [第七章：朴素贝叶斯算法](Chapter07/README-zh-CN.md)

- 先验概率后验概率辨析
- 朴素的含义
- 极大后验概率与平滑处理
- 从零实现`CategoricalNB`贝叶斯与`LabelBinarizer`
- 从零实现`MultinomialNB`贝叶斯与`TfidfVectorizer`
- 从零实现`GaussianNB`贝叶斯

### [第八章：决策树与集成模型](./Chapter08/README-zh-CN.md)

- 信息熵与条件熵
- ID3与C4.5生成算法`DecisionTreeClassifier`
- ID3与C4.5剪枝与可视化`graphviz`
- CART分类树生成与剪枝
- Bagging、Boosting和Stacking集成学习
- `BaggingClassifier`,`AdaBoostClassifier`和`StackingClassifier`
- 随机森林与特征重要性
- 泰坦尼克号生还预测
- 从零实现决策树
- 从零实现CART决策树算法
- 从零实现AdaBoost算法
- 从零实现AdaBoost之SAMME算法
- 从零实现GradientBoost
- 梯度提升树原理介绍
- 从零实现GradientBoost Regression
- 从零实现GradientBoost Classification

### [第九章：支持向量机](./Chapter09/README-zh-CN.md)

- 支持向量机的思想与原理
- 最大间隔分类器与sklearn示例`sklearn.svm.SVC`
- 从低维高维再到无穷维核函数
- 条件极值与拉格朗日乘子法
- 对偶性与KKT条件
- SVM优化问题与SMO算法
- 从零实现支持向量机

### [第十章：聚类算法](./Chapter10/README-zh-CN.md)

- Kmeans聚类算法原理与实现
- Kmeans聚类Sklearn示例
- Kmeans聚类算法弊端可视化
- Kmeans++原理与从零实现
- 4种常见聚类外部评价指标原理与实现
- WKmeans聚类算法原理与从零实现
- 3种常见聚类内部评价指标原理与实现
- elbow方法分析K值及其弊端
- silhouette方法分析K值
- elbow和silhouette发综合分析
- 类Kmeans聚类算法弊端可视化
- 基于密度的聚类算法实现
- Single-Link、Complete-Link和Ward层次聚类原理与实现

### [第十一章：降维算法](./Chapter11/README-zh-CN.md)

- 主成分分析原理与实现
- PCA可视化
- sklearn中PCA使用示例
- 从零实现PCA降维算法
- Kernel主成分分析原理与实现
- PCA与KPCA对比可视化
- KPCA算法思想可视化
- KPCA算法sklearn示例
- KPCA算法实现 代码

### [第十二章：自训练与标签传播算法](./Chapter12/README-zh-CN.md)

- 半监督学习之self-training讲解
- self-training示例
- self-training从零实现
- 半监督学习之label-propagations讲解
- label-propagations示例
- label-propagation从零实现
- label-propagation从零实现(论文中非迭代版本)
- 半监督学习之label-spreading讲解
- label-spreading示例
- label-propagation和label-spreading对比
- label-spreading从零实现

### [返回主页](../README-zh-CN.md)


