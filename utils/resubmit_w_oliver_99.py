import pandas as pd
import numpy as np

df = pd.read_csv('../submissions/LGBMClassifier_weight-multi-logloss-0.935157_2018-10-28-13-14-25.csv')

_df = df.drop(['object_id', 'class_99'], axis=1).values

preds_99 = np.ones(_df.shape[0])
for i in range(_df.shape[1]):
    preds_99 *= (1 - _df[:, i])

df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)

df.to_csv('../submissions/LGBMClassifier_weight-multi-logloss-0.935157_2018-10-28-13-14-25_ovliver-99.csv', index=False)
