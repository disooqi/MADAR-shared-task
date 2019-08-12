import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1

corpus = 'corpus-26'

train_src, _, train_tgt, _ = prepare_train_data_for_task_1(corpus= corpus)
dev_src, dev_tgt = prepare_dev_data_for_task_1(corpus= corpus)


tfidf_vect = TfidfVectorizer(analyzer= 'char',lowercase=False, max_df=0.95,ngram_range=(2, 5), smooth_idf=False,
                             sublinear_tf=True)

# vectorizer.fit(sentences)
# X_train = vectorizer.transform(dataset_train)
# X_test = vectorizer.transform(dataset_test)
# # X_removed = vectorizer.transform(removed_sentences)
#
# sgd_clf_02 = SGDClassifier()
# sgd_clf_02.fit(X_train, label_train)
#
# pred_train = sgd_clf_02.predict(X_train)
# pred_test = sgd_clf_02.predict(X_test)
# # pred_removed = sgd_clf_02.predict(X_removed)
#
# print 'Training Acc: ',np.around(np.mean(pred_train == label_train)*100,2), '%'
# print 'Testing Acc: ',np.around(np.mean(pred_test == label_test)*100,2), '%'

# tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(train_src)
X_dev = tfidf_vect.transform(dev_src)


clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-8, n_jobs=-1)
clf.fit(X_train, train_tgt)
train_pred = clf.predict(X_train)
dev_pred = clf.predict(X_dev)


with open(f'{corpus}/result/run_04.G', mode='w') as gold:
    for rs in dev_tgt:
        gold.write(f'{rs}\n')

with open(f'{corpus}/result/run_04.P', mode='w') as prediction:
    for rs in dev_pred:
        prediction.write(f'{rs}\n')

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')