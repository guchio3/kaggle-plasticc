import pandas as pd
import numpy as np

from tqdm import tqdm


def unpack_passbands(set_df):
    res_df = pd.DataFrame(np.unique(set_df[['object_id', 'mjd']], axis=1))
    res_df.columns = ['object_id', 'mjd']
    for i in tqdm([0, 1, 2, 3, 4, 5]):
        res_df = res_df.merge(
            set_df[set_df.passband == i].drop('passband', axis=1).rename(
                columns={
                    'flux': 'flux_{}'.format(i),
                    'flux_err': 'flux_err_{}'.format(i),
                    'detected': 'detected_{}'.format(i)}),
            on=['object_id', 'mjd'],
            how='left')
    return res_df
