import numpy as np
import pandas as pd

import cesium.featurize as featurize

import warnings
from multiprocessing import Pool
from logging import getLogger
from tqdm import tqdm
import sys
sys.path.append('../tools/')

from feature_tools import load_test_set_dfs, split_dfs
from my_logging import logInit

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)


def normalise(ts):
    return (ts - ts.mean()) / ts.std()


def get_phase(set_df):
    groups = set_df.groupby(['object_id', 'passband'])
    times = groups.apply(lambda block: block['mjd'].values).\
        reset_index().\
        rename(columns={0: 'seq'})
    flux = groups.apply(lambda block: normalise(block['flux']).values).\
        reset_index().\
        rename(columns={0: 'seq'})
    times_list = times.groupby('object_id').\
        apply(lambda x: x['seq'].tolist()).\
        tolist()
    flux_list = flux.groupby('object_id').\
        apply(lambda x: x['seq'].tolist()).\
        tolist()
    warnings.simplefilter('ignore', RuntimeWarning)
    if np.prod(np.isnan(np.array(times_list))) * np.prod(np.isnan(np.array(flux_list))) > 0:
        freq_df = featurize.featurize_time_series(times=times_list,
                                              values=flux_list,
                                              features_to_use=['freq1_freq'],
                                              scheduler=None)
        freqs = pd.DataFrame(freq_df.median(axis=1)).rename(columns={0: 'freq_median'})
        freqs['object_id'] = set_df.object_id.unique()
        set_df = set_df.merge(
            freqs,
            on='object_id',
            how='left').reset_index(drop=True)
        set_df['phase'] = set_df['mjd'] * set_df['freq_median'] % 1
        set_df.drop(['freq_median'], axis=1, inplace=True)
    else:
        set_df['phase'] = np.nan
    return set_df


def _main(nthread, test_flg):
    logger = getLogger(__name__)
    logInit(logger, log_dir='../log/', log_filename='add_phase.log')

    if test_flg:
        set_dfs = load_test_set_dfs(nthread, logger)
        for i, df in tqdm(list(enumerate(set_dfs))):
            df = get_phase(df)
            df.reset_index(drop=True).to_feather('./test_dfs/{}.fth'.format(i))
    else:
        set_df = pd.read_csv(
            '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv')
            #'/Users/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv')
        phase_df = get_phase(set_df)
        phase_df.to_csv(
            '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv',
            #'/Users/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv',
            index=False)


def main(nthread, test_flg):
    logger = getLogger(__name__)
    logInit(logger, log_dir='../log/', log_filename='add_phase.log')

    if test_flg:
        set_dfs = load_test_set_dfs(nthread, logger)
    else:
        set_df = pd.read_csv(
            #'/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv')
            '/Users/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv')
        set_dfs = split_dfs(set_df, nthread, logger)

    logger.info('start multiprocessing')
    p = Pool(nthread)
    phase_df_list = p.map(get_phase, set_dfs)
    p.close()
    p.join()
    logger.info('done multiprocessing')

    if test_flg:
        for i, df in tqdm(list(enumerate(phase_df_list))):
            df.reset_index(drop=True).to_feather('/home/naoya.taguchi/workspace/kaggle/plasticc-2018/test_dfs/{}.fth'.format(i))
    else:
        phase_df = pd.concat(phase_df_list, axis=0).reset_index(drop=True)
        phase_df.to_csv(
            #'/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv',
            '/Users/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set.csv',
            index=False)


if __name__ == '__main__':
    main(62, True)
    #main(62, False)
    #main(2, False)
