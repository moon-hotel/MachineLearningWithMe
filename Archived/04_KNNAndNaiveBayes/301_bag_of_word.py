from sklearn.feature_extraction.text import CountVectorizer

s = ['文本 分词 工具 可 用于 对 文本 进行 分词 处理',
     '常见 的 用于 处理 文本 的 分词 处理 工具 有 很多']
count_vec = CountVectorizer()
x = count_vec.fit_transform(s).toarray()
vocab = count_vec.vocabulary_
vocab = sorted(vocab.items(),key=lambda x:x[1])
print(vocab)
print(x)
