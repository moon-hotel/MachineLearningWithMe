import numpy as np

np.set_printoptions(3)


def estimator(n_samples, sample_weight):
    # 在真实情况中，需要sample_weight参与模型的训练过程，这里是模拟所以没用到
    y_pred = np.random.uniform(0, 1, n_samples) < 0.5
    return np.array(y_pred, dtype=np.int32)


def single_boost(n_samples, sample_weight, y, iboost):
    y_predict = estimator(n_samples, sample_weight)
    incorrect = y_predict != y
    estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
    # estimator_error = sum(incorrect * sample_weight) / sum(sample_weight)
    estimator_weight = np.log((1. - estimator_error) / estimator_error)

    sample_weight *= np.exp(estimator_weight * incorrect)
    print(f"estimator: {iboost}")
    print("sample_weight: ", sample_weight)
    print("y_predict: ", y_predict)
    print("estimator_error: ", estimator_error)
    print("estimator_weight: ", estimator_weight)
    print("accuracy:", np.average(y_predict == y))
    print("I(*) incorrect: ", incorrect)

    print("=" * 20)
    return sample_weight, estimator_weight, estimator_error, y_predict


def boost(y, n_samples, n_estimators):
    sample_weight = np.ones(n_samples, dtype=np.float64)  # 初始化权重全为1
    sample_weight /= sample_weight.sum()  # 标准化权重
    estimator_weights_ = np.zeros(n_estimators, dtype=np.float64)
    estimator_errors_ = np.zeros(n_estimators, dtype=np.float64)
    y_predicts_ = []
    for iboost in range(n_estimators):
        sample_weight, estimator_weight, estimator_error, y_predict \
            = single_boost(n_samples, sample_weight, y, iboost)
        estimator_weights_[iboost] = estimator_weight
        estimator_errors_[iboost] = estimator_error
        y_predicts_.append(y_predict)
        if estimator_error == 0:
            break
        sample_weight_sum = np.sum(sample_weight)
        if sample_weight_sum <= 0:
            break
        # sample_weight /= sample_weight_sum # 标准化样本权重
    y_predicts_ = np.vstack(y_predicts_)
    return estimator_weights_, y_predicts_


def predict(estimator_weights, y_predicts):
    K = 2  # 分类数
    estimator_weights = np.reshape(estimator_weights, [1, -1])
    result = []
    for k in range(K):
        correct = (y_predicts == k)
        result.append(np.matmul(estimator_weights, correct))
    result = np.vstack(result)
    print("计算分类概率值: \n", result)
    y_pred = np.argmax(result, axis=0)
    return y_pred


if __name__ == '__main__':
    n_samples = 10  # 371
    np.random.seed(370)
    y = np.random.randint(0, 2, 10)  # 生成正确标签
    n_estimators = 4
    print("正确标签: ", y)
    estimator_weights, y_predicts = boost(y, n_samples, n_estimators)
    print("estimator_weights:", estimator_weights)
    print("每个分类器的预测结果:\n", y_predicts)
    y_pred = predict(estimator_weights, y_predicts)
    print("预测结果：", y_pred)
    print("AdaBoost准确率:", np.average(y == y_pred))

    # 正确标签:  [0 1 1 0 1 0 1 1 0 0]
    # estimator: 0
    # sample_weight:  [0.1 0.1 0.1 0.4 0.4 0.1 0.1 0.1 0.1 0.1]
    # y_predict:  [0 1 1 1 0 0 1 1 0 0]
    # estimator_error:  0.2
    # estimator_weight:  1.3862943611198906
    # accuracy: 0.8
    # I(*) incorrect:  [False False False  True  True False False False False False]
    # ====================
    # estimator: 1
    # sample_weight:  [0.1 0.1 0.1 0.4 0.4 0.1 0.1 0.7 0.1 0.7]
    # y_predict:  [0 1 1 0 1 0 1 0 0 1]
    # estimator_error:  0.125
    # estimator_weight:  1.9459101490553132
    # accuracy: 0.8
    # I(*) incorrect:  [False False False False False False False  True False  True]
    # ====================
    # estimator: 2
    # sample_weight:  [0.1   0.1   0.133 0.4   0.533 0.1   0.1   0.933 0.1   0.7  ]
    # y_predict:  [0 1 0 0 0 0 1 0 0 0]
    # estimator_error:  0.42857142857142866
    # estimator_weight:  0.2876820724517807
    # accuracy: 0.7
    # I(*) incorrect:  [False False  True False  True False False  True False False]
    # ====================
    # estimator: 3
    # sample_weight:  [0.1   0.153 0.204 0.4   0.533 0.1   0.153 1.425 0.1   0.7  ]
    # y_predict:  [0 0 0 0 1 0 0 0 0 0]
    # estimator_error:  0.3958333333333333
    # estimator_weight:  0.42285685082003377
    # accuracy: 0.6
    # I(*) incorrect:  [False  True  True False False False  True  True False False]
    # ====================
    # estimator_weights: [1.386 1.946 0.288 0.423]
    # 每个分类器的预测结果:
    #  [[0 1 1 1 0 0 1 1 0 0]
    #  [0 1 1 0 1 0 1 0 0 1]
    #  [0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 0]]
    # 计算分类概率值:
    #  [[4.043 0.423 0.711 2.656 1.674 4.043 0.423 2.656 4.043 2.097]
    #  [0.    3.62  3.332 1.386 2.369 0.    3.62  1.386 0.    1.946]]
    # 预测结果： [0 1 1 0 1 0 1 0 0 0]
    # AdaBoost准确率: 0.9
