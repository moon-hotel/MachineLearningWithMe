import numpy as np

np.set_printoptions(3)


def estimator(n_samples):
    y_pred = np.random.uniform(0, 1, n_samples) < 0.5
    return np.array(y_pred, dtype=np.int32)


def single_boost(n_samples, sample_weight, y, iboost):
    y_predict = estimator(n_samples)
    incorrect = y_predict != y
    estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
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
    y_predicts_ = np.vstack(y_predicts_)
    return estimator_weights_, y_predicts_


def predict(estimator_weights, y_predicts):
    K = 2
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
    print("正确标签: ", y)
    estimator_weights, y_predicts = boost(y, n_samples, 4)
    print("estimator_weights:", estimator_weights)
    print("所有没有预测结果:\n", y_predicts)
    y_pred = predict(estimator_weights, y_predicts)
    print("AdaBoost准确率:", np.average(y == y_pred))
