import numpy as np
import pandas as pd

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

df = pd.read_csv('../submissions/LGBMClassifier_weight-multi-logloss-0.890562_2018-11-06-13-06-21.csv')
df.class_99 = sigmoid(df.class_99 / np.max(df.class_99) * 4 - 2)
df.to_csv('../submissions/LGBMClassifier_weight-multi-logloss-0.890562_2018-11-06-13-06-21_sigmoid.csv', index=False)
