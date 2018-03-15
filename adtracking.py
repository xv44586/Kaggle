"""
__author__ = 'xumingming'
"""
import gc
import pandas as pd
from functools import reduce
import itertools
import time
import numpy as np
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
import lightgbm as lgb

from utils import *


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10,
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.01,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric': metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics + ":", evals_results['valid'][metrics][n_estimators - 1])

    return bst1


path = '/Users/xumingming/Kaggle/'
train_file_name = 'train.csv'
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
min_interval = 10

print('load train...')
train_df = pd.read_csv(path + train_file_name, nrows=40000000, dtype=dtypes,
                       usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time',
                                'is_attributed'])
print('load test...')
test_df = pd.read_csv(path + "test.csv", dtype=dtypes,
                      usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df = train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
# train_df['interval_click2attr'] = (pd.to_datetime(train_df.attributed_time) - pd.to_datetime(
#     train_df.click_time)) / np.timedelta64(1, 's')
gc.collect()

print('group by...')
# gp = train_df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
#     ['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})

# ip_device_channel = train_df[['ip', 'device', 'day', 'hour', 'channel']].groupby(by=['ip', 'device', 'day', 'hour'])[
#     ['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_channel'})

# ip_app_device_channel = train_df[['ip', 'device', 'app', 'day', 'hour', 'channel']].groupby(
#     by=['ip', 'device', 'app', 'day', 'hour']
# )[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_device_channel'})
#
# ip_app_device_os_channel = train_df[['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']].groupby(
#     by=['ip', 'app', 'device', 'os', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={
#     'channel': 'ip_app_device_os_channel'})
# gp.info()
feature_matrix = [
    ['ip', 'channel'],
    ['device', 'channel'],
    ['app', 'channel'],
    ['os', 'channel'],
    ['channel', 'os'],

    ['ip', 'device', 'channel'],
    ['ip', 'os', 'channel'],
    ['ip', 'app', 'channel'],
    ['ip', 'channel', 'os'],

    ['ip', 'device', 'os', 'channel'],
    ['ip', 'device', 'app', 'channel'],
    ['ip', 'device', 'channel', 'app'],

    ['ip', 'device', 'app', 'os', 'channel'],
    ['ip', 'device', 'app', 'channel', 'os'],

    ['ip', 'device', 'app', 'os', 'channel', 'click_time'],
]

# group_features = ['ip', 'device', 'os', 'channel', 'app']
# feature_matrix = []
# for g_feature in itertools.product(group_features, group_features):
#     if len(set(g_feature)) > 1:
#         feature_matrix.append(list(g_feature))

# for g_feature in itertools.product(group_features, group_features, group_features):
#     if len(set(g_feature)) > 2:
#         feature_matrix.append(list(g_feature))
#
# for g_feature in itertools.product(group_features, group_features, group_features, group_features):
#     if len(set(g_feature)) > 3:
#         feature_matrix.append(list(g_feature))
#
# for g_feature in itertools.product(group_features, group_features, group_features, group_features, group_features):
#     if len(set(g_feature)) > 4:
#         feature_matrix.append(list(g_feature))

group_result = (count_group_by(train_df, x) for x in feature_matrix)

print('merge...')
_index = 0

for gp in group_result:
    result, feature_name = gp
    print(result is gp[0])
    train_df = train_df.merge(result, on=feature_matrix[_index][:-1], how='left')
    _index += 1

    del result
    gc.collect()

group_result_feature_names = [g[1] for g in group_result]

del group_result
del _index
gc.collect()
# train_df = train_df.merge(gp, on=['ip', 'day', 'hour'], how='left')
# train_df = train_df.merge(ip_app_device_os_channel, on=['ip', 'app', 'device', 'os', 'day', 'hour'], how='left')
# train_df = train_df.merge(ip_device_channel, on=['ip', 'device', 'day', 'hour'], how='left')
print("vars and data type: ")
train_df.info()

gc.collect()

split_count = 3000000
test_df = train_df[len_train:]
val_df = train_df[(len_train - split_count):len_train]
train_df = train_df[:(len_train - split_count)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
predictors = ['app', 'device', 'os', 'channel', 'hour']

predictors.extend(group_result_feature_names)
print(predictors)
categorical = ['app', 'device', 'os', 'channel', 'hour']

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
params = {
    'learning_rate': 0.1,
    # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 400,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': .7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99  # because training data is extremely unbalanced
}
bst = lgb_modelfit_nocv(params,
                        train_df,
                        val_df,
                        predictors,
                        target,
                        objective='binary',
                        metrics='auc',
                        early_stopping_rounds=30,
                        verbose_eval=True,
                        num_boost_round=500,
                        categorical_features=categorical)

del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced.csv', index=False)
print("done...")
print(sub.info())
