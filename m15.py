# -*- coding: utf-8 -*-
'''
Решение задачи для http://mlbootcamp.ru/championship/10/
(c) Илья Козиев 2017 inkoziev@gmail.com
Используется xgboost
Подбор параметров выполняется через beam search
Оценка параметров - cross-validation
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
import json


col_ranges = dict()

colnames = ['maxPlayerLevel', 'numberOfAttemptedLevels', 'attemptsOnTheHighestLevel',
            'totalNumOfAttempts', 'averageNumOfTurnsPerCompletedLevel', 'doReturnOnLowerLevels',
            'numberOfBoostersUsed', 'fractionOfUsefullBoosters', 'totalScore', 'totalBonusScore',
            'totalStarsCount', 'numberOfDaysActuallyPlayed']

log_scale_cols = {}
# 'maxPlayerLevel', 'numberOfAttemptedLevels', 'attemptsOnTheHighestLevel', 'totalNumOfAttempts',
#                   'averageNumOfTurnsPerCompletedLevel', 'numberOfBoostersUsed',
#                   'totalScore', 'totalBonusScore', 'totalStarsCount', 'numberOfDaysActuallyPlayed'}


# ------------------------------------------------------------------------------------------------------

'''
Функция для нормализация исходных данных.
В зависимости от имени столбца может быть использована линейное масштабирование
либо логарифмическое сжатие.
'''
def get_x(colname, cell_value):
    if colname == 'doReturnOnLowerLevels':
        return int(cell_value)
    elif colname in log_scale_cols:
        denom = float(col_ranges[colname][1] - col_ranges[colname][0])
        cell_val = math.log(float(xrow[colname] + 1.0)) / math.log(denom)
        return cell_val
    else:
        denom = float(col_ranges[colname][3])
        cell_val = float(xrow[colname] - col_ranges[colname][2]) / denom
        return cell_val

# Грузим исходные данные
#data_folder = r'e:\MLBootCamp\III'
data_folder = r'/home/eek/polygon/MLBootCamp/III'

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

print('ntrain={}'.format(ntrain))
print('nb_features={}'.format(nb_features))

# --------------------------------------------------------------

# Нормализуем данные и создаем датасеты для обучения и для сабмита.
X_data = np.zeros((nb_patterns, nb_features), dtype=np.float32)
y_data = np.zeros((nb_patterns), dtype=np.float)

print('Vectorization...')
patterns = []
for irow in range(ntrain):
    xrow = x_train.iloc[irow]
    for icol, colname in enumerate(colnames):
        X_data[irow, icol] = get_x(colname, xrow[colname])

    y = y_train.iloc[irow][0]
    y_data[irow] = float(y)




# --------------------------------------------------------------------

nb_test = x_test.shape[0]

X_testdata = np.zeros((nb_test, nb_features), dtype=np.float32)

print('\n\nVectorization of x_test...')
for irow in range(nb_test):
    xrow = x_test.iloc[irow]
    for icol, colname in enumerate(colnames):
        X_testdata[irow, icol] = get_x(colname, xrow[colname])

# ---------------------------------------------------------------------------------------------------------

# Количество наборов параметров в ходе поиска.
BEAM_SIZE = 10

# Здесь храним наборы параметров
params_beam = []

iprobe = 0

# При первом запуске будет сформированы чисто случайные параметры.
# При последующих запусках будет загружен с диска последний список наборов параметров
# и поиск продолжится с него.
best_acc = 0.0
if os.path.exists('params_beam.json'):
    print( 'Loading previous params_beam...' )
    with open( 'params_beam.json', 'rt') as cfg:
        params_beam = json.load(cfg)
        
    iprobe = random.randint(1,1000000)
    best_acc = max( [ a for (p,a) in params_beam ] )
    print( 'previous best_acc={}'.format( best_acc ) )
else:
    for i in range(BEAM_SIZE):
        px = {
              'n_estimators': random.randint(100, 700),
              'subsample': random.uniform(0.2, 0.85),
              'max_depth': random.randint(1, 4),
              'seed': random.randint(1, 2000000000),
              'min_child_weight': random.randint(1, 6),
              'colsample_bytree': random.uniform(0.1, 1.0)
             }
        params_beam.append( (px, -10000) )

# --------------------------------------------------------------------------------------------------

print('\n\nFitting XGBClassifier...')

# Сюда будем писать параметры для моделей, которые бьют текущий рекорд.
wrt_log = codecs.open('search.log', 'w')

_learning_rate = 0.05
decay_rate = 0.999
_gamma = 0.01

# Бесконечный цикл подбора
for k in range(10000):

    _learning_rate = _learning_rate*decay_rate
    print( '_learning_rate={:6.4f}'.format(_learning_rate) )
    
    # На каждом раунде будем подмешивать в поиск несколько новых
    # случайных наборов параметров, чтобы избежать потенциального
    # вырождения гипероптимизации.
    for irnd in range( random.randint(0,3) ):
        print( 'Adding random param[{}] group to beam'.format(irnd) )
        px = {
              'n_estimators': random.randint(100, 700),
              'subsample': random.uniform(0.2, 0.85),
              'max_depth': random.randint(1, 4),
              'seed': random.randint(1, 2000000000),
              'min_child_weight': random.randint(1, 6),
              'colsample_bytree': random.uniform(0.1, 1.0)
             }
        params_beam.append( (px, -10000) )
        
        # также модифицируем seed у лучшего из имеющихся наборов параметров
        # и добавляем его.
        max_acc = max( [ a for (p,a) in params_beam ] )
        cur_best_param = [ p for p in params_beam if p[1]==max_acc ][0]
        params_new = dict( cur_best_param[0].iteritems() )
        params_new['seed'] = random.randint(1, 2000000000)
        params_beam.append( (params_new,-1000000) )
        
        

    # Обходим текущие наборы параметров, мутируем их, проверяем на рекорд.
    beam_size = len(params_beam)
    for params0 in params_beam[0:beam_size]:
        params_new = dict( params0[0].iteritems() )
        params_string = str.join( ' ', [ p+'='+str(v) for (p,v) in params_new.iteritems() ] )
        ichange = random.randint(1,5)
        if ichange == 1:
            params_new['n_estimators'] = random.randint(100, 700)
        elif ichange == 2:
            params_new['subsample'] = random.uniform(0.2, 0.95)
        elif ichange == 3:
            params_new['max_depth'] = random.randint(1, 7)
        elif ichange == 4:
            params_new['seed'] = random.randint(1, 2000000000)
        elif ichange == 5:
            params_new['min_child_weight'] = random.randint(1, 6)
        elif ichange == 6:
            params_new['colsample_bytree'] = random.uniform(0.1, 1.0)

        acc_sum = 0.0

        # Используем каждый раз случайный cross-validation.
        # Возможен вариант, когда split выполняется один раз до цикла и
        # частные датасеты сохраняются и использются внутри цикла. Но случайная
        # перегенерация в теории должна работать лучше как дополнительный случайный
        # фактов в stochastic gradient search гипероптимизации.
        N_SPLIT = 7
        kf = KFold(N_SPLIT, random_state=params_new['seed'], shuffle=False)

        classifiers = []
        for train_index, test_index in kf.split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            cl = xgboost.XGBClassifier(
                                       learning_rate=_learning_rate,
                                       gamma=_gamma,
                                       silent=True,
                                       **params_new
                                      )
            cl.fit(X_train, y_train)

            y_pred = cl.predict_proba(X_test)
            acci = -sklearn.metrics.log_loss(y_test, y_pred)

            acc_sum += acci
            classifiers.append(cl)

        # Оценка достигнутой точности как среднее арифметическое лосса для каждого классификатора.
        acc = acc_sum / N_SPLIT

        if sum( 1 for p in params_beam if p[1]!=acc )>0:
            params_beam.append((params_new, acc))

        print('acc={:7.5f}\t{}'.format(acc,params_string))
        if iprobe == 0 or acc > best_acc:
            # Есть новый рекорд точности.
            print("New best_acc={}, storing submission y's...".format(acc))
            print( params_string )
            print('=' * 60)
            wrt_log.write('iprobe={}\t{}\tacc={}\n'.format(iprobe, params_string, acc))
            wrt_log.flush()
            best_acc = acc

            # сгенерируем данные для сабмита как среднее от результатов каждого из классификаторов.
            y_submisson = None
            for i, cl in enumerate(classifiers):
                if i == 0:
                    y_submission = cl.predict_proba(X_testdata)
                else:
                    y_submission += cl.predict_proba(X_testdata)

            y_pred = y_submission / N_SPLIT

            # сохраняем сабмит в файл.
            with codecs.open('y_test({})_kfold.csv'.format(iprobe), 'w') as wrt:
                for idata in range(nb_test):
                    y = y_pred[idata][1]
                    wrt.write('{}\n'.format(y))

        iprobe += 1

    # оставляем только top лучших наборов параметров.
    params_beam = sorted( params_beam, key=lambda z: -z[1] )[0:BEAM_SIZE]
    min_acc = min( [ a for (p,a) in params_beam ] )
    max_acc = max( [ a for (p,a) in params_beam ] )
    print( 'Storing the current beam search with min_acc={} max_acc={}'.format(min_acc,max_acc) )
    with codecs.open( 'params_beam.json','w','utf-8') as cfg:
        json.dump( params_beam, cfg )

wrt_log.close()
