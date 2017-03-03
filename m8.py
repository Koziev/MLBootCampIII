# -*- coding: utf-8 -*-
'''
Решение задачи для http://mlbootcamp.ru/championship/10/
(c) Илья Козиев 2017 inkoziev@gmail.com
Model#8 - случайный подбор параметров xgboost, бэггинг.
TODO: возможно, переделать на http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
PS: overfitting
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
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import xgboost

col_ranges = dict()

# colnames = ['maxPlayerLevel', 'numberOfAttemptedLevels', 'attemptsOnTheHighestLevel',
#             'totalNumOfAttempts', 'averageNumOfTurnsPerCompletedLevel', 'doReturnOnLowerLevels',
#             'numberOfBoostersUsed', 'fractionOfUsefullBoosters', 'totalScore', 'totalBonusScore',
#             'totalStarsCount', 'numberOfDaysActuallyPlayed']


colnames = ['maxPlayerLevel', 'numberOfAttemptedLevels', 'attemptsOnTheHighestLevel',
            'totalNumOfAttempts', 'averageNumOfTurnsPerCompletedLevel', 'doReturnOnLowerLevels',
            'numberOfBoostersUsed', 'fractionOfUsefullBoosters', 'totalScore', 'totalBonusScore',
            'totalStarsCount', 'numberOfDaysActuallyPlayed']

log_scale_cols = { 'maxPlayerLevel', 'numberOfAttemptedLevels', 'attemptsOnTheHighestLevel', 'totalNumOfAttempts',
                   'averageNumOfTurnsPerCompletedLevel', 'numberOfBoostersUsed',
                   'totalScore', 'totalBonusScore', 'totalStarsCount', 'numberOfDaysActuallyPlayed' }

# ------------------------------------------------------------------------------------------------------


def get_x( colname, cell_value):
    if colname=='doReturnOnLowerLevels':
        return int(cell_value)
    elif colname in log_scale_cols:
        denom = float(col_ranges[colname][1] - col_ranges[colname][0])
        cell_val = math.log( float(xrow[colname]+1.0) ) / math.log( denom )
        return cell_val
    else:
        denom = float(col_ranges[colname][3])
        cell_val = float(xrow[colname]-col_ranges[colname][2]) / denom
        return cell_val

data_folder = r'e:\MLBootCamp\III'
#data_folder = r'/home/eek/polygon/MLBootCamp/III'

x_train = pd.read_csv(os.path.join(data_folder, 'x_train.csv'), delimiter=';', skip_blank_lines=True)
y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv'), delimiter=';', skip_blank_lines=True, header=None)
x_test = pd.read_csv(os.path.join(data_folder, 'x_test.csv'), delimiter=';', skip_blank_lines=True)



for colname in colnames:
    dt = x_train[colname]
    col_min = dt.min()
    col_max = dt.max()
    mean = dt.mean()
    std = dt.std()
    col_ranges[colname] = (col_min, col_max, mean, std)

ntrain = x_train.shape[0]
nb_patterns = ntrain
nb_features = len(colnames)

print( 'ntrain={}'.format(ntrain) )
print( 'nb_features={}'.format(nb_features) )

# --------------------------------------------------------------

X_data = np.zeros((nb_patterns, nb_features), dtype=np.float32)
y_data = np.zeros((nb_patterns), dtype=np.float)

print('Vectorization...')
patterns = []
for irow in range(ntrain):
    xrow = x_train.iloc[irow]
    for icol,colname in enumerate(colnames):
        X_data[ irow, icol ] = get_x( colname, xrow[colname] )

    y = y_train.iloc[irow][0]
    y_data[irow] = float(y)

   
# --------------------------------------------------------------------

nb_test = x_test.shape[0]

X_testdata = np.zeros((nb_test, nb_features), dtype=np.float32)

print('\n\nVectorization of x_test...')
for irow in range(nb_test):
    xrow = x_test.iloc[irow]
    for icol,colname in enumerate(colnames):
        X_testdata[ irow, icol ] = get_x( colname, xrow[colname] )

# --------------------------------------------------------------------

if False:
    print( 'Start grid search for GridSearchCV with XGBClassifier...' )

    param_grid2 = [
      { 'n_estimators': [100,200,300,400,500,600,700,800,900],
        'max_depth': [1,2,3,4,5],
        'subsample': np.linspace( 0.1, 1.0, 10 )
        #'base_score': np.linspace(0.3,0.6,6),
        #'colsample_bylevel':[1,2,3],
        #'colsample_bytree':np.linspace(0.7,1.0,3)        
        }
     ]

    estimator = xgboost.XGBClassifier()
    grid = GridSearchCV( estimator, param_grid=param_grid2, cv=5, verbose=3, scoring='neg_log_loss' )
    grid.fit(X_data1, y_data)
    print( 'best score=', 1 - grid.best_score_ )

    for pname,pval in grid.best_params_.iteritems():
        print( 'best {}={}'.format(pname,pval) )

    raw_input('Press a key to start fitting...')
    
# --------------------------------------------------------------------

print('\n\nFitting XGBClassifier...')

wrt_log = codecs.open( 'search.log', 'w' )

best_acc = 0.0
iprobe=0

_learning_rate = 0.1


for k in range(10000):

    _n_estimators = random.randint(100,700)
    _subsample=random.uniform(0.2,0.95)
    _max_depth=random.randint(1,5)
    _seed = random.randint(1,2000000000)
    _min_child_weight = random.randint(1,6)

    acc_sum = 0.0

    N_SPLIT = 5
    kf = KFold( N_SPLIT, random_state=_seed, shuffle=False)
    
    classifiers = []
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        cl = xgboost.XGBClassifier(n_estimators=_n_estimators,
                                   subsample=_subsample,
                                   max_depth=_max_depth,
                                   seed=_seed,
                                   learning_rate=_learning_rate,
                                   silent=True
                                  )
        cl.fit(X_train, y_train)
        
        y_pred = cl.predict_proba(X_test)
        acci = -sklearn.metrics.log_loss( y_test, y_pred )
        
        acc_sum += acci
        classifiers.append(cl)
    
    acc = acc_sum/N_SPLIT

    print( 'k={} acc={}'.format(k,acc))
    if iprobe==0 or acc>best_acc:
        print( "New best_acc={}, storing submission y's...".format(acc) )
        print( 'n_estimators={} subsample={} max_depth={} seed={} _min_child_weight={}'.format(_n_estimators,_subsample,_max_depth,_seed,_min_child_weight) )
        print( '='*60 )
        wrt_log.write( 'iprobe={}\tn_estimators={} subsample={} max_depth={} seed={} min_child_weight={}\tacc={}\n'.format(iprobe,_n_estimators,_subsample,_max_depth,_seed,_min_child_weight,acc) )
        wrt_log.flush()
        best_acc = acc
        
        y_submisson = None
        for i,cl in enumerate(classifiers):
            if i==0:
                y_submission = cl.predict_proba(X_testdata)
            else:
                y_submission += cl.predict_proba(X_testdata)
                
        y_pred = y_submission/N_SPLIT
    
        with codecs.open( 'y_test({})_kfold.csv'.format(iprobe), 'w' ) as wrt:
            for idata in range(nb_test):
                y = y_pred[idata][1]
                wrt.write( '{}\n'.format(y) )

        cl0 = xgboost.XGBClassifier(n_estimators=_n_estimators,
                                   subsample=_subsample,
                                   max_depth=_max_depth,
                                   seed=_seed,
                                   learning_rate=_learning_rate,
                                   silent=True )
        cl0.fit(X_data,y_data)
        y_pred = cl0.predict_proba(X_testdata)
        
        with codecs.open( 'y_test({})_whole.csv'.format(iprobe), 'w' ) as wrt:
            for idata in range(nb_test):
                y = y_pred[idata][1]
                wrt.write( '{}\n'.format(y) )

    iprobe += 1

wrt_log.close()
