import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1


corpus = 'corpus-26'

train_src, _, train_tgt, _ = prepare_train_data_for_task_1(corpus= corpus)
dev_src, dev_tgt = prepare_dev_data_for_task_1(corpus= corpus)

tfidf_vect = TfidfVectorizer(ngram_range=(1, 5))
X_train = tfidf_vect.fit_transform(train_src)
X_dev = tfidf_vect.transform(dev_src)


clf = MultinomialNB()
clf.fit(X_train, train_tgt)
train_pred = clf.predict(X_train)
dev_pred = clf.predict(X_dev)


with open(f'{corpus}/result/run_02.G', mode='w') as gold:
    for rs in dev_tgt:
        gold.write(f'{rs}\n')

with open(f'{corpus}/result/run_02.P', mode='w') as prediction:
    for rs in dev_pred:
        prediction.write(f'{rs}\n')

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')