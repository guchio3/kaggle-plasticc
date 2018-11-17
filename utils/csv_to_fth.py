import pandas as pd

df = pd.read_csv('/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/test_set.csv')
df.to_feather('/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/test_set.fth')
