import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1, prepare_test_data_for_task_1




corpus = 'corpus-26'

train_src, _, train_tgt, _ = prepare_train_data_for_task_1(corpus= corpus)
dev_src, dev_tgt = prepare_dev_data_for_task_1(corpus= corpus)
test_src = prepare_test_data_for_task_1(corpus= corpus)
print(len(test_src))
# analyzer= 'char',lowercase=False, max_df=0.95,ngram_range=(2, 5), smooth_idf=False,
#                              sublinear_tf=True

labels = ['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS', 'BEI', 'BEN',
              'CAI', 'DAM', 'DOH', 'FES', 'JED', 'JER', 'KHA', 'MOS', 'MSA',
              'MUS', 'RAB', 'RIY', 'SAL', 'SAN', 'SFX', 'TRI', 'TUN']

_y_train = [labels.index(tgt.replace(' ', '')) for tgt in train_tgt]
_y_dev = [labels.index(tgt.replace(' ', '')) for tgt in dev_tgt]

count_vec = CountVectorizer(analyzer='word',  min_df=1, ngram_range=(1, 1), encoding='utf-8', )
X_train_counts = count_vec.fit_transform(train_src)
X_dev_counts = count_vec.transform(dev_src)
X_test_counts = count_vec.transform(test_src)

tf_vec = TfidfTransformer(use_idf=False)
X_train_tf = tf_vec.fit_transform(X_train_counts)
X_dev_tf = tf_vec.transform(X_dev_counts)

tfidf_vect = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)
X_train_tfidf = tfidf_vect.fit_transform(X_train_counts)
X_dev_tfidf = tfidf_vect.transform(X_dev_counts)
X_test_tfidf = tfidf_vect.transform(X_test_counts)


# print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'مكسورة'))


clf = MultinomialNB(alpha=0.31).fit(X_train_tfidf, train_tgt)

train_pred = clf.predict(X_train_tfidf)
dev_pred = clf.predict(X_dev_tfidf)
test_pred = clf.predict(X_test_tfidf)

with open(f'{corpus}/result/run_test_np_01.P', mode='w') as prediction:
    for rs in test_pred:
        prediction.write(f'{rs}\n')

with open(f'{corpus}/result/run_dev_np_01.P', mode='w') as prediction:
    for rs in dev_pred:
        prediction.write(f'{rs}\n')


with open(f'{corpus}/result/run_dev_np_01.G', mode='w') as prediction:
    for rs in dev_tgt:
        prediction.write(f'{rs}\n')


with open(f'{corpus}/result/run_dev_np_01.A', mode='w') as prediction:
    for src, tgt, pred in zip(dev_src, dev_tgt, dev_pred):
        prediction.write(f'{src} T:{tgt} P:{pred}\n')

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')