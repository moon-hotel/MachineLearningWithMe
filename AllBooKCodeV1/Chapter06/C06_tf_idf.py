corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    count = CountVectorizer(vocabulary=vocabulary)
    count_matrix = count.fit_transform(corpus).toarray()
    tfidf_trans = TfidfTransformer(norm=None)
    tfidf_matrix = tfidf_trans.fit_transform(count_matrix)
    idf_vec = tfidf_trans.idf_
    print(tfidf_matrix.toarray())

    print("\n\nTfidfVectorizer转换结果：")
    tfidf = TfidfVectorizer(norm=None, vocabulary=vocabulary)
    print(tfidf.fit_transform(corpus).toarray())
