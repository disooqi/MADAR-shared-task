import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from preparedata import prepare_train_data_for_task_1, prepare_dev_data_for_task_1
from sklearn.pipeline import Pipeline


text_clf = Pipeline([
    ('vect', CountVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_df=0.95,
    )),
    ('tfidf', TfidfTransformer()),
    # ('clf', MultinomialNB()),
    ('clf', svm.SVC(kernel='linear', C=300)),
    # ('clf', SGDClassifier(
    #     loss='hinge', penalty='l2',
    #     alpha=1e-3, random_state=42,
    #     max_iter=5, tol=None)),
])

# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
parameters = {
    # 'vect__analyzer': ('char', 'word'),
    # 'vect__ngram_range': [(1, 3), (2, 5)],
    # 'vect__max_df': (0.9, 0.95, 1.0),
    # 'tfidf__smooth_idf': (False, True),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__sublinear_tf': (True, False),
    # 'clf__alpha': (1e-1, 1e-2, 1e-3),
    # 'clf__kernel':('linear', 'rbf'),
    # 'clf__C':[1, 10]
}


corpus = 'corpus-26'

train_src, _, train_tgt, _ = prepare_train_data_for_task_1(corpus= corpus)
dev_src, dev_tgt = prepare_dev_data_for_task_1(corpus= corpus)


# gs = grid search
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
# train_src.extend(dev_src)
# train_tgt.extend(dev_tgt)
gs_clf.fit(train_src, train_tgt)

best_clf = gs_clf.best_estimator_
train_pred = best_clf.predict(train_src)
dev_pred = best_clf.predict(dev_src)

print('Training Acc: ',np.around(np.mean(train_pred == train_tgt)*100,2), '%')
print('Testing Acc: ',np.around(np.mean(dev_pred == dev_tgt)*100,2), '%')