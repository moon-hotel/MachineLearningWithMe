from sklearn.feature_extraction.text import TfidfVectorizer

s = ['文本 分词 工具 可 用于 对 文本 进行 分词 处理',
     '常见 的 用于 处理 文本 的 分词 处理 工具 有 很多']
tfidf = TfidfVectorizer(stop_words=None,
                        token_pattern=r"(?u)\b\w\w+\b", max_features=6)  # 默认值
weight = tfidf.fit_transform(s).toarray()
word = tfidf.get_feature_names()
print('vocabulary list:')
vocab = tfidf.vocabulary_.items()
vocab = sorted(vocab, key=lambda x: x[1])
print(vocab)
print('IFIDF词频矩阵:')
print(weight)

for i in range(len(weight)):
    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，
    # 第二个for便利某一类文本下的词语权重
    print(u"-------这里输出第", i, u"个文本的词语tf-idf权重------")
    for j in range(len(word)):
        print(word[j], weight[i][j])  # 第i个文本中，第j个次的tfidf值
