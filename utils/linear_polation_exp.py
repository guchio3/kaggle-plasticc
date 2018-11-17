import numpy as np
import pandas as pd

import pickle


test_set_metadata_df = pd.read_csv('/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/test_set_metadata.csv')
object_ids = test_set_metadata_df.object_id

with open('../temp/Booster_weight-multi-logloss-0.612193_2018-11-11-04-49-01_res.csv', 'rb') as fin:
    test_reses = pickle.load(fin)


lin_pure_res = np.clip(test_reses[-1], 10**(-15), 1 - 10**(-15))
lin_pure_preds_99 = np.ones((lin_pure_res.shape[0]))
for i in range(lin_pure_res.shape[1]):
    lin_pure_preds_99 *= (1 - lin_pure_res[:, i])
lin_pure_preds_99 = 0.14 * lin_pure_preds_99 / np.mean(lin_pure_preds_99)
lin_pure_res_df = pd.DataFrame(lin_pure_res, columns=[
    'class_6',
    'class_15',
    'class_16',
    'class_42',
    'class_52',
    'class_53',
    'class_62',
    'class_64',
    'class_65',
    'class_67',
    'class_88',
    'class_90',
    'class_92',
    'class_95',
])
lin_pure_res_df['class_99'] = lin_pure_preds_99
lin_pure_res_df['object_id'] = object_ids
lin_pure_res_df.to_csv('../temp/Booster_weight-multi-logloss-0.612193_2018-11-11-04-49-01_res_lin_pure.csv', index=False)



all_mean_res = np.clip(np.mean(test_reses, axis=0), 10**(-15), 1 - 10**(-15))
all_mean_preds_99 = np.ones((all_mean_res.shape[0]))
for i in range(all_mean_res.shape[1]):
    all_mean_preds_99 *= (1 - all_mean_res[:, i])
preds_99 = 0.14 * all_mean_preds_99 / np.mean(all_mean_preds_99)

all_mean_res_df = pd.DataFrame(all_mean_res, columns=[
    'class_6',
    'class_15',
    'class_16',
    'class_42',
    'class_52',
    'class_53',
    'class_62',
    'class_64',
    'class_65',
    'class_67',
    'class_88',
    'class_90',
    'class_92',
    'class_95',
])
all_mean_res_df['class_99'] = all_mean_preds_99
all_mean_res_df['object_id'] = object_ids
all_mean_res_df.to_csv('../temp/Booster_weight-multi-logloss-0.612193_2018-11-11-04-49-01_res_all_mean.csv', index=False)
