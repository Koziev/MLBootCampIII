# -*- coding: utf-8 -*-
'''
Решение задачи для http://mlbootcamp.ru/championship/10/
(c) Илья Козиев 2017 inkoziev@gmail.com
Идея:
1) случайно выбираем набор параметров для xgboost.
2) используем StratifiedKFold для разбиения обучающего набора на N_SPLIT порций
3) обучаем N_SPLIT моделей, при обучании каждой модели используем early_stopping
   по тестовой порции в используемом фолде.
4) делаем предсказание по каждой модели, затем усредняем предсказания
'''

import os
import pandas as pd
import numpy as np
import codecs
import math
import random
import numpy as np
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics
import sklearn.utils
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost
import sklearn.preprocessing
import sklearn.decomposition


# Папка с исходными данными задачи
#data_folder = r'e:\MLBootCamp\III'
data_folder = r'/home/eek/polygon/MLBootCamp/III'

# Папка для сохранения результатов
results_folder = r'./results'

# Выполнять ли dimension reduction для исходных данных
REDUCE_DIM = False

N_SPLIT = 8

AVERAGING = 'arith' # 'arith' или 'geom'




# -----------------------------------------------------------------------
# ЧАСТЬ 1 - ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

x_train = pd.read_csv(os.path.join(data_folder, 'x_train.csv'), delimiter=';', skip_blank_lines=True)
y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv'), delimiter=';', skip_blank_lines=True, header=None)
x_test = pd.read_csv(os.path.join(data_folder, 'x_test.csv'), delimiter=';', skip_blank_lines=True)

ntrain = x_train.shape[0]
ntest = x_test.shape[0]
nfeatures0 = x_train.shape[1]

print('ntrain={}'.format(ntrain))
print('ntest={}'.format(ntest))

y_data = np.zeros((ntrain), dtype=np.float)
for irow in range(ntrain):
    y_data[irow] = float(y_train.iloc[irow][0])

# соединим X обучающего и тестового датасета и сделаем преобразование с получившимся датасетом
X_all = np.zeros( ( ntrain+ntest, nfeatures0 ) )
X_all[ :ntrain, : ] = x_train
X_all[ ntrain:, : ] = x_test

# нормализуем данные
print( 'Data normalization...' )
xnormalizer = sklearn.preprocessing.StandardScaler()
xnormalizer.fit(X_all)
X_normal = xnormalizer.transform(X_all)
X_train_normal = xnormalizer.transform(x_train)
X_test_normal = xnormalizer.transform(x_test)


if REDUCE_DIM:
    # уменьшаем размерность данных
    n_components = 10
    print( 'Dimensionality reduction to n_components={}...'.format(n_components) )
    reductor = sklearn.decomposition.PCA(n_components)
    #reductor = sklearn.decomposition.KernelPCA(n_components,kernel='poly',degree=2)
    reductor.fit(X_normal)
    X_train_reduced = reductor.transform(X_train_normal)
    X_test_reduced = reductor.transform(X_test_normal)
else:
    X_train_reduced = X_train_normal
    X_test_reduced = X_test_normal


X_data = X_train_reduced # просто алиас
X_submission = X_test_reduced

D_submission = xgboost.DMatrix(X_submission)

nb_features = X_data.shape[1]
nb_test = X_submission.shape[0]

# --------------------------------------------------------------------

print('\n\nFitting XGBClassifier...')

wrt_log = codecs.open( os.path.join( results_folder, 'search.csv' ), 'w' )
wrt_records = codecs.open( os.path.join( results_folder, 'records.log' ), 'w' )

best_acc  = 0.0 # текущий абсолютный рекорд
iprobe=0

for k in range(10000):

    # случайный выбор параметров
    _n_estimators = random.randint(80,300)
    _subsample=0.45 #random.uniform(0.50,0.85)
    _max_depth=2 #random.randint(1,6)
    _seed = random.randint(1,2000000000)
    _min_child_weight = 8 #random.randint(1,6)
    _colsample_bytree = 0.6 #random.uniform(0.50,0.90)
    _colsample_bylevel = 0.6
    _learning_rate = random.uniform(0.03,0.10)
    _gamma = 5 #random.uniform(0.0,0.10)

    xgb_params = { 
                   'booster': 'gbtree', #'dart',
                   'n_estimators': _n_estimators,
                   'subsample': _subsample,
                   'max_depth': _max_depth,
                   'seed': _seed,
                   'min_child_weight': _min_child_weight,
                   'eta': _learning_rate,
                   'gamma': _gamma,
                   'colsample_bytree': _colsample_bytree,
                   'colsample_bylevel': _colsample_bylevel,
                   'scale_pos_weight': 1,
                   'silent': 1,
                   'eval_metric': 'logloss',
                   'objective': 'binary:logistic'
                 }
                 
    if xgb_params['booster']=='dart':
        xgb_params['rate_drop'] = 0.1
        xgb_params['one_drop'] = 1

    # на каждой итерации делаем новый случайный фолдинг
    kf = StratifiedKFold( N_SPLIT, random_state=_seed, shuffle=True)

    acc = 0.0
    classifiers = []
    
    acc_sum = 0.0
    for train_index, test_index in kf.split(X_data,y_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        D_train = xgboost.DMatrix(X_train,label=y_train)
        D_test = xgboost.DMatrix(X_test,label=y_test)

        cl = xgboost.train( xgb_params, D_train,
                            num_boost_round=1000,#_n_estimators,
                            evals=[(D_test,'eval')],
                            early_stopping_rounds=20,
                            verbose_eval=False
                          )

        y_pred = cl.predict(D_test,ntree_limit=cl.best_ntree_limit)
        acci = -sklearn.metrics.log_loss( y_test, y_pred )
        
        acc_sum += acci
        classifiers.append(cl)
    
    acc = acc_sum/N_SPLIT
        
    print( 'k={} acc={}'.format(k,acc))
    submission_path = os.path.join( results_folder, 'y_submission({}).csv'.format(iprobe) )
    
    if iprobe==0 or acc>best_acc:
        print( "\nNew best_acc={}, storing submission y's...".format(acc) )
        print( 'n_estimators={} subsample={} max_depth={} seed={} _min_child_weight={} colsample_bytree={} _learning_rate={}'.format(_n_estimators,_subsample,_max_depth,_seed,_min_child_weight,_colsample_bytree,_learning_rate) )
        print( '='*60 )
        wrt_records.write( 'new acc={:7.5f} previous best_acc={:7.5f} iprobe={} ==> {}\n'.format(acc,best_acc,iprobe,submission_path) )
        wrt_records.flush()
        best_acc = acc
        
    wrt_log.write( '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(iprobe,acc,submission_path,_n_estimators,_subsample,_max_depth,_seed,_min_child_weight,_colsample_bytree,_learning_rate) )
    wrt_log.flush()

    if acc<0.383:
        y_submisson = None
        ncl = float(len(classifiers))
        for i,cl in enumerate(classifiers):
            if i==0:
                y_submission = cl.predict(D_submission,ntree_limit=cl.best_ntree_limit)
            elif AVERAGING=='geom':
                y_submission *= cl.predict(D_submission,ntree_limit=cl.best_ntree_limit)
            else:
                y_submission += cl.predict(D_submission,ntree_limit=cl.best_ntree_limit)

        if AVERAGING == 'geom':
            # todo переделать на vectorize etc ...
            y_pred = np.zeros( (nb_test,2), dtype=np.float )
            for i in range(nb_test):
                y_pred[i] = math.pow( y_submission[i], 1.0/ncl )
        else:
            y_pred = y_submission/float(ncl)

        with codecs.open( submission_path, 'w' ) as wrt:
            for idata in range(nb_test):
                y = y_pred[idata]
                wrt.write( '{}\n'.format(y) )

    iprobe += 1

wrt_log.close()
wrt_records.close()
