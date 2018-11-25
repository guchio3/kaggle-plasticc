import pandas as pd

filename='../../plasticc-2018/submissions/Booster_weight-multi-logloss-0.612193_2018-11-10-22-58-58.csv'
df = pd.read_csv(filename)

df.class_99 = 1/9

cols = list(df.columns)
cols.remove('class_99')
cols.remove('object_id')
df[cols] *= 8/9

df.to_csv('../submissions/' + filename.split('/')[-1][:-4] + '_class_99_1above9.csv', index=False)
