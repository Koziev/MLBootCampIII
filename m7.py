# -*- coding: utf-8 -*-
'''
Компоненты входного вектора нормализуются и приводятся к диапазону 0...1
Случайный подбор варианта ансамбля деревьев.
'''

import os
import pandas as pd
import numpy as np
import codecs
import math
import random
import numpy as np
import scipy.sparse
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.metrics
import sklearn.utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.ensemble
import scipy as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


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


def get_x1( colname, cell_value):
    if colname=='doReturnOnLowerLevels':
        return int(cell_value)
    elif colname in log_scale_cols:
        denom = float(col_ranges[colname][1] - col_ranges[colname][0])
        cell_val = math.log( float(xrow[colname]+1.0) ) / math.log( denom )
        return cell_val
    else:
        denom = float(col_ranges[colname][3])
        cell_val = float(xrow[colname]) / denom
        return cell_val

def get_x2( colname, cell_value):
    if colname=='doReturnOnLowerLevels':
        return int(cell_value)
    else:
        denom = float(col_ranges[colname][3])
        cell_val = float(xrow[colname]-col_ranges[colname][2]) / denom
        return cell_val

data_folder = r'e:\MLBootCamp\III'
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

X_data1 = np.zeros((nb_patterns, nb_features), dtype=np.float32)
X_data2 = np.zeros((nb_patterns, nb_features), dtype=np.float32)
y_data = np.zeros((nb_patterns), dtype=np.float)

print('Vectorization...')
patterns = []
for irow in range(ntrain):
    xrow = x_train.iloc[irow]
    for icol,colname in enumerate(colnames):
        X_data1[ irow, icol ] = get_x1( colname, xrow[colname] )
        X_data2[ irow, icol ] = get_x2( colname, xrow[colname] )

    y = y_train.iloc[irow][0]
    y_data[irow] = float(y)

X_data_12 = [ X_data1, X_data2 ]    
    
# --------------------------------------------------------------------

nb_test = x_test.shape[0]

X_testdata1 = np.zeros((nb_test, nb_features), dtype=np.float32)
X_testdata2 = np.zeros((nb_test, nb_features), dtype=np.float32)

print('\n\nVectorization of x_test...')
for irow in range(nb_test):
    xrow = x_test.iloc[irow]
    for icol,colname in enumerate(colnames):
        X_testdata1[ irow, icol ] = get_x1( colname, xrow[colname] )
        X_testdata2[ irow, icol ] = get_x2( colname, xrow[colname] )

X_testdata12 = [ X_testdata1, X_testdata2 ]

# --------------------------------------------------------------------

print('\n\nFitting GradientBoostingClassifier...')

wrt_log = codecs.open( 'search.log', 'w' )

best_acc = 0.0
iprobe=0


for k in range(1000):

    _n_estimators = random.randint(20,800)
    _subsample=random.uniform(0.3,1.0)
    _max_depth=random.randint(1,5)
    cl_type = random.randint(0,1) # 0 - GradientBoostingClassifier, 1 - RandomForestClassifier

    for idata,X_data in enumerate(X_data_12):
    
        acc_sum = 0.0

        N_SPLIT = 5
        kf = KFold(n_splits=N_SPLIT, random_state=None, shuffle=False)
        
        classifiers = []
        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            if cl_type==0:
                cl = sklearn.ensemble.GradientBoostingClassifier(n_estimators=_n_estimators,
                                                          subsample=_subsample,
                                                          max_depth=_max_depth)
            else:
                cl = sklearn.ensemble.RandomForestClassifier(n_estimators=_n_estimators,
                                                          max_depth=_max_depth)
            cl.fit(X_train, y_train)
            
            y_pred = cl.predict_proba(X_test)
            acci = -sklearn.metrics.log_loss( y_test, y_pred )
            
            acc_sum += acci
            classifiers.append(cl)
        
        acc = acc_sum/N_SPLIT

        print( 'k={} cl_type={} acc={}'.format(k,cl_type,acc))
        if iprobe==0 or acc>best_acc:
            print( "New best_acc={}, storing submission y's...".format(acc) )
            
            wrt_log.write( 'iprobe={}\tidata={}\tcl_type={}\tn_estimators={} subsample={} max_depth={}\tacc={}\n'.format(iprobe,idata,cl_type,_n_estimators,_subsample,_max_depth,acc) )
            wrt_log.flush()
            best_acc = acc
            
            X_testdata = X_testdata12[idata]
            y_submisson = None
            for i,cl in enumerate(classifiers):
                if i==0:
                    y_submission = cl.predict_proba(X_testdata)
                else:
                    y_submission += cl.predict_proba(X_testdata)
                    
            y_pred = y_submission/N_SPLIT
        
            with codecs.open( 'y_test({}).csv'.format(iprobe), 'w' ) as wrt:
                for idata in range(nb_test):
                    y = y_pred[idata][1]
                    wrt.write( '{}\n'.format(y) )

        iprobe += 1

wrt.close()
