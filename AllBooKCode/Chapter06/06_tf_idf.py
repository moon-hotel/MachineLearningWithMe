corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    count = CountVectorizer(vocabulary=vocabulary)
    count_matrix = count.fit_transform(corpus).toarray()
    tfidf_trans = TfidfTransformer(norm=None)
    tfidf_matrix = tfidf_trans.fit_transform(count_matrix)
    idf_vec = tfidf_trans.idf_
    print(tfidf_matrix.toarray())
