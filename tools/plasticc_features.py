import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import kurtosis

import gc
from multiprocessing import Pool
from tqdm import tqdm
import warnings

import cesium.featurize as featurize

from features import featureCreator, MulHelper, toapply


warnings.simplefilter('ignore', RuntimeWarning)
warnings.filterwarnings('ignore')
np.random.seed(71)


# =======================================
# feature functions
# =======================================
def weighted_mean(flux, dflux):
    return np.sum(flux * (flux / dflux)**2) /\
        np.sum((flux / dflux)**2)


def normalized_flux_std(flux, wMeanFlux):
    return np.std(flux / wMeanFlux, ddof=1)


def normalized_amplitude(flux, wMeanFlux):
    return (np.max(flux) - np.min(flux)) / wMeanFlux


def normalized_MAD(flux, wMeanFlux):
    return np.median(np.abs((flux - np.median(flux)) / wMeanFlux))


def beyond_1std(flux, wMeanFlux):
    return sum(np.abs(flux - wMeanFlux) > np.std(flux, ddof=1)) / len(flux)


def get_starter_features(_id_grouped_df):
    f = _id_grouped_df.flux
    df = _id_grouped_df.flux_err
    m = weighted_mean(f, df)
    std = normalized_flux_std(f, df)
    amp = normalized_amplitude(f, m)
    mad = normalized_MAD(f, m)
    beyond = beyond_1std(f, m)
    return m, std, amp, mad, beyond

def get_flux_mjd_diff(df):
    return df.flux.diff()/df.mjd.diff()

def get_flux_mjd_diff_mean(df):
    return get_flux_mjd_diff(df).mean()

def get_flux_mjd_diff_max(df):
    return get_flux_mjd_diff(df).max()

def get_flux_mjd_diff_min(df):
    return get_flux_mjd_diff(df).min()

def get_flux_mjd_diff_std(df):
    return get_flux_mjd_diff(df).std()

def get_flux_mjd_diff_var(df):
    return get_flux_mjd_diff(df).var()

def diff_mean(x):
    return x.diff().mean()

def diff_max(x):
    return x.diff().max()

def diff_std(x):
    return x.diff().std()

def diff_var(x):
    return x.diff().var()

def diff_sum(x):
    return x.diff().sum()

def get_max_min_diff(x):
    return x.max() - x.min()

def quantile10(x):
    return x.quantile(0.10)

def quantile25(x):
    return x.quantile(0.25)

def quantile75(x):
    return x.quantile(0.75)

def quantile90(x):
    return x.quantile(0.90)

def quantile95(x):
    return x.quantile(0.95)

def minmax_range(x):
    return x.max() - x.min()

def quantile2575_range(x):
    return quantile75(x) - quantile25(x)

def quantile1090_range(x):
    return quantile90(x) - quantile10(x)



# =======================================
# feature creator
# =======================================
class featureCreatorPreprocess(featureCreator):
    def __init__(self, load_dir, save_dir, 
            src_df_dict=None, logger=None, nthread=1, train=True):
        super(featureCreatorPreprocess, self).\
                __init__(load_dir=load_dir,
                        save_dir=save_dir,
                        src_df_dict=src_df_dict,
                        logger=logger,
                        nthread=nthread)
        self.train = train

    def _load(self):
        if self.train:
            path_dict = {
                    'set_df': self.load_dir + 'training_set.csv',
                    'set_metadata_df': self.load_dir + 'training_set_metadata.csv'}
        else:
            path_dict = {'set_metadata_df': self.load_dir + 'test_set_metadata.csv'}
            for i in tqdm([i for i in range(62)]):
                path_dict[f'test_set_{i}_df'] = f'../test_dfs/{i}.fth'
        self._load_dfs_from_paths(path_dict=path_dict)

    def _split_dfs(self, df, nthread, save_flg=False):
        self._log_print('calculating uniq object_id num')
        object_ids = df.object_id.unique()
        self._log_print('getting groups')
        groups = np.array_split(object_ids, nthread)
        self._log_print('splitting df')
        dfs = []
        for group in tqdm(list(groups)):
            dfs.append(df[df.object_id.isin(set(group))])
        if save_flg:
            self._log_print('saving the split dfs...')
            for i, df in tqdm(list(enumerate(dfs))):
                df.reset_index().to_feather('./test_dfs/{}.fth'.format(i))
        return dfs

    def _add_corrected_flux(self, set_df, set_metadata_df):
        # _set_metadata_df = set_metadata_df[
        #         (set_metadata_df.hostgal_photoz_err < 0.5) &
        #         (set_metadata_df.hostgal_photoz_err > 0.)]
        set_metadata_df['lumi_dist'] = 10**((set_metadata_df.distmod+5)/5)
        _set_metadata_df = set_metadata_df
        set_df = set_df.merge(
            _set_metadata_df[['object_id', 'hostgal_photoz', 'lumi_dist']],
            on='object_id',
            how='left')
        set_df['corrected_flux'] = set_df.flux / (set_df.hostgal_photoz**2)
        set_df['normed_flux'] = (set_df.flux - set_df.flux.min()) / set_df.flux.max()
        set_df['luminosity'] = 4*np.pi*(set_df.lumi_dist**2)*set_df.flux
        return set_df
    
    def _create_features(self):
        if self.train:
            set_dfs = self._split_dfs(self.src_df_dict['set_df'], self.nthread)
            for i in tqdm([i for i in range(62)]):
                splitted_set_df = set_dfs[i]
                set_df_name = f'test_set_{i}_df'
                self.src_df_dict[set_df_name] = splitted_set_df

        # flux の補正を入れる
        self._log_print('adding corrected flux...')
        for i in tqdm([i for i in range(62)]):
            set_df_name = f'test_set_{i}_df'
            self.src_df_dict[set_df_name] = \
                    self._add_corrected_flux(
                        self.src_df_dict[set_df_name],
                        self.src_df_dict['set_metadata_df']
                    )

        self._log_print('pre-processing set dfs ...')
        for i in tqdm([i for i in range(62)]):
            set_df_name = f'test_set_{i}_df'
            _set_df = self.src_df_dict[set_df_name]
            # preprocess
            _set_df['flux_ratio_to_flux_err'] = _set_df['flux'] / _set_df['flux_err']
            _set_df['flux_ratio_sq'] = np.power(
                _set_df['flux'] / _set_df['flux_err'], 2.0)
            _set_df['flux_by_flux_ratio_sq'] = _set_df['flux'] * \
                _set_df['flux_ratio_sq']
            _set_df['corrected_flux_ratio_sq'] = np.power(
                _set_df['corrected_flux'] / _set_df['flux_err'], 2.0)
            _set_df['corrected_flux_by_flux_ratio_sq'] = _set_df['corrected_flux'] * \
                _set_df['flux_ratio_sq']
            # replace
            self.src_df_dict[set_df_name] = _set_df


def fe_set_df_base(corrected_set_df):
    aggregations = {
        # 'passband': ['mean', 'std', 'var'],
        # 'mjd': ['max', 'min', 'var'],
        # 'mjd': [diff_mean, diff_max],
        # 'phase': [diff_mean, diff_max],
        'flux': ['min', 'max', 'mean', 'median',
                 'std', 'var', 'skew', 'count', kurtosis],
        'corrected_flux': ['min', 'max', 'mean', 'median',
                           'std', 'var', 'skew', ],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', kurtosis],
        'flux_ratio_to_flux_err': ['min', 'max', ],
        'detected': ['mean', ],
        'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis],
        'flux_by_flux_ratio_sq': ['sum', 'skew', ],
        'corrected_flux_ratio_sq': ['sum', 'skew', ],
        'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
        # 'luminosity': ['median', 'var', 'skew', kurtosis],
        #        'minused_flux': ['min', 'max', 'mean', 'median',
        #                         'std', 'var', 'skew'],
        #        'normed_flux': ['mean', 'median', 'skew'],
        # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
    }

    fe_set_df = corrected_set_df.groupby('object_id').agg({**aggregations})
    fe_set_df.columns = pd.Index( [e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
    return fe_set_df


def fe_set_df_detected(corrected_set_df): 
    detected_aggregations = {
        'mjd': [get_max_min_diff, 'skew'],
    }
    fe_set_df = corrected_set_df.groupby('object_id').agg({**detected_aggregations})
    fe_set_df.columns = pd.Index( [e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
    return fe_set_df


def fe_set_df_detected(corrected_set_df): 
    detected_aggregations = {
        'mjd': [get_max_min_diff, 'skew'],
    }
    fe_set_df = corrected_set_df.groupby('object_id').agg({**detected_aggregations})
    fe_set_df.columns = pd.Index( [e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
    return fe_set_df



class featureCreatorSet(featureCreator):
    def __init__(self, fe_set_df, set_res_df_name, load_dir, save_dir, 
            src_df_dict=None, logger=None, nthread=1):
        super(featureCreatorSet, self).\
                __init__(load_dir=load_dir,
                        save_dir=save_dir,
                        src_df_dict=src_df_dict,
                        logger=logger,
                        nthread=nthread)
        self.fe_set_df = fe_set_df
        self.set_res_df_name = set_res_df_name

    def _load(self, ):
        None

#    def _fe_set_df(self, set_df):
#        aggregations = {
#            # 'passband': ['mean', 'std', 'var'],
#            # 'mjd': ['max', 'min', 'var'],
#            # 'mjd': [diff_mean, diff_max],
#            # 'phase': [diff_mean, diff_max],
#            'flux': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis],
#            'corrected_flux': ['min', 'max', 'mean', 'median',
#                               'std', 'var', 'skew', ],
#            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', kurtosis],
#            'flux_ratio_to_flux_err': ['min', 'max', ],
#            'detected': ['mean', ],
#            'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis],
#            'flux_by_flux_ratio_sq': ['sum', 'skew', ],
#            'corrected_flux_ratio_sq': ['sum', 'skew', ],
#            'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
#            # 'luminosity': ['median', 'var', 'skew', kurtosis],
#            #        'minused_flux': ['min', 'max', 'mean', 'median',
#            #                         'std', 'var', 'skew'],
#            #        'normed_flux': ['mean', 'median', 'skew'],
#            # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
#        }
#
#        fe_set_df = set_df.groupby('object_id').agg({**aggregations})
#        fe_set_df.columns = pd.Index(
#            [e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
#
#        return fe_set_df

    def _create_features(self):
        self._log_print('creating base features ...')
        set_df_name = [f'test_set_{i}_df' for i in range(62)]
        set_dfs = [self.src_df_dict[f] for f in set_df_name]
        with Pool(self.nthread) as p:
            self._log_print('start fature engineering ...')
            #set_res_list = p.map(self._fe_set_df_base, set_dfs)
            set_res_list = p.map(self.fe_set_df, set_dfs)
            #set_res_list = p.map(MulHelper(self, '_fe_set_df'), set_dfs)
            #set_res_list = p.apply(toapply, (self, '_fe_set_df', set_dfs))
            p.close()
            p.join()
            set_res_df = pd.concat(set_res_list, axis=0)
            gc.collect()

        # set the result in df_dict
        set_res_df.reset_index(inplace=True)
        self.df_dict[self.set_res_df_name] = set_res_df
