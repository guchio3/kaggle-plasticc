import numpy as np
import pandas as pd
import sys, os, gc
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
#import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count

#import utils, utils_metric
import tools.objective_function as utils_metric


X = pd.read_pickle('./features/onodera_feats/X_train_1_1217-1.pkl.gz')
#y = utils.load_target().target
y = pd.read_csv('/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set_metadata.csv').target

target_dict = {}
target_dict_r = {}
for i,e in enumerate(y.sort_values().unique()):
    target_dict[e] = i
    target_dict_r[i] = e

y = y.replace(target_dict)

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()




SEED = np.random.randint(9999)
np.random.seed(SEED)
print('SEED:', SEED)

NFOLD = 5


param = {
         'objective': 'multiclass',
         'num_class': 14,
         'metric': 'multi_logloss',
         
         'learning_rate': 0.5,
         'max_depth': 3,
         'num_leaves': 63,
         'max_bin': 127,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.5,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         }




dtrain = lgb.Dataset(X, y.values, #categorical_feature=CAT, 
                 free_raw_data=False)

gc.collect()
param['seed'] = np.random.randint(9999)
ret, models = lgb.cv(param, dtrain, 99999, nfold=NFOLD,
                     fobj=utils_metric.wloss_objective, 
                     feval=utils_metric.wloss_metric,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)
