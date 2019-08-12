import codecs
from collections import Counter
import random
import numpy as np

from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1


from scipy.optimize import minimize


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm

from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

train_src, _, train_tgt, _ = prepare_train_data_for_task_1()
dev_src, dev_tgt = prepare_dev_data_for_task_1()

tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(train_src)
X_dev = tfidf_vect.transform(dev_src)


clf = SGDClassifier()
clf.fit(X_train, train_tgt)
train_pred = clf.predict(X_train)
dev_pred = clf.predict(X_dev)


with open('corpus-26/result/run_01.G', mode='w') as gold:
    for rs in dev_tgt:
        gold.write(f'{rs}\n')

with open('corpus-26/result/run_01.P', mode='w') as prediction:
    for rs in dev_pred:
        prediction.write(f'{rs}\n')

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')


#
# my_C = 100
#
# # clf = svm.SVC()
# # clf.fit(X_train, label_train)
# # train_pred = clf.predict(X_train)
# # y_pred = clf.predict(X_test)
#
# clf_linear = svm.LinearSVC(C= 0.3)
# clf_linear.fit(X_train, label_train)
# train_pred = clf_linear.predict(X_train)
# y_pred = clf_linear.predict(X_test)
#
# print 'Training Acc: ',np.around(np.mean(train_pred == label_train)*100,2), '%'
# print 'Testing Acc: ',np.around(np.mean(y_pred == label_test)*100,2), '%'