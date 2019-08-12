import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm

from sklearn.metrics import confusion_matrix
from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1


corpus = 'corpus-26'

train_src, _, train_tgt, _ = prepare_train_data_for_task_1(corpus= corpus)
dev_src, dev_tgt = prepare_dev_data_for_task_1(corpus= corpus)

count_vec =CountVectorizer(analyzer='char', ngram_range=(2, 5), max_df=0.95)
X_train_counts = count_vec.fit_transform(train_src)
X_dev_counts = count_vec.transform(dev_src)

tfidf_vect = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)
X_train_tfidf = tfidf_vect.fit_transform(X_train_counts)
X_dev_tfidf = tfidf_vect.transform(X_dev_counts)

clf = svm.SVC(kernel='linear', C=300, )
clf.fit(X_train_tfidf, train_tgt)

#
# my_C = 100
#
# clf = svm.SVC()
# clf.fit(X_train, train_tgt)
# train_pred = clf.predict(X_train)
# dev_pred = clf.predict(X_dev)
#
# clf_linear = svm.LinearSVC(C= 0.3)
# clf_linear.fit(X_train, train_tgt)
# train_pred = clf_linear.predict(X_train)
# dev_pred = clf_linear.predict(X_dev)
#
train_pred = clf.predict(X_train_tfidf)
dev_pred = clf.predict(X_dev_tfidf)

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')