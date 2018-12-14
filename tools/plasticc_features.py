import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import kurtosis

import gc
from multiprocessing import Pool
from tqdm import tqdm
import warnings

import cesium.featurize as featurize
from tsfresh.feature_extraction import extract_features

from features import featureCreator, MulHelper, toapply

from astropy.cosmology import FlatLambdaCDM


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

def calc_flux_mjd_skewness(df):
    mjd = df.mjd
    flux = df.flux.clip(0., None)
    mean = (df.mjd * flux).sum() / flux.sum()
    std = np.abs(np.sqrt(((mjd - mean)**2 * flux).sum() / flux.sum()))
    fm_skew = ((((mjd - mean) * flux).sum())/flux.sum())**3 / std**3
    return fm_skew

def calc_flux_mjd_kurtosis(df):
    mjd = df.mjd
    flux = df.flux.clip(0., None)
    mean = (df.mjd * flux).sum() / flux.sum()
    std = np.abs(np.sqrt(((mjd - mean)**2 * flux).sum() / flux.sum()))
    fm_kurt = ((((mjd - mean) * flux).sum())/flux.sum())**4 / std**4
    return fm_kurt


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
        # cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        # distance_modulus = cosmo.distmod(set_metadata_df.hostgal_specz)
        # set_metadata_df['z_distmod'] = distance_modulus

        set_metadata_df['lumi_dist'] = 10**((set_metadata_df.distmod+5)/5)
        # set_metadata_df['z_lumi_dist'] = 10**((set_metadata_df.distmod+5)/5)
        _set_metadata_df = set_metadata_df
        set_df = set_df.merge(
            _set_metadata_df[['object_id', 'hostgal_photoz', 'lumi_dist', 'distmod', 'hostgal_specz']],
            on='object_id',
            how='left')
        set_df['corrected_flux'] = set_df.flux / (set_df.hostgal_photoz**2)
        set_df['z_corrected_flux'] = set_df.flux / (set_df.hostgal_specz**2)
        set_df['normed_flux'] = (set_df.flux - set_df.flux.min()) / set_df.flux.max()
#        set_df['luminosity'] = 4*np.pi*(set_df.lumi_dist**2)*set_df.flux
        # set_df['z_luminosity'] = 4*np.pi*(set_df.z_lumi_dist**2)*set_df.flux
#        set_df['magnitude'] = -2.5*np.log10(set_df.flux)
#        set_df['abs_magnitude'] = set_df.magnitude - set_df.distmod
        del set_df['distmod'], set_df['hostgal_specz'], set_df['lumi_dist']#, set_df['z_lumi_dist'], set_df['magnitude']
        gc.collect()
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
            _set_df['z_corrected_flux_ratio_sq'] = np.power(
                _set_df['z_corrected_flux'] / _set_df['flux_err'], 2.0)
            _set_df['z_corrected_flux_by_flux_ratio_sq'] = _set_df['z_corrected_flux'] * \
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
#        'abs_magnitude': ['min', 'max', 'mean', 'median',
#                 'std', 'var', 'skew', 'count', kurtosis],
        'corrected_flux': ['min', 'max', 'mean', 'median',
                           'std', 'var', 'skew', ],
        'z_corrected_flux': ['min', 'max', 'mean', 'median',
                           'std', 'var', 'skew', ],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', kurtosis],
        'flux_ratio_to_flux_err': ['min', 'max', ],
        'detected': ['mean', ],
        'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis, 'max'],
        'flux_by_flux_ratio_sq': ['sum', 'skew', ],
        'corrected_flux_ratio_sq': ['sum', 'skew', ],
        'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
        'z_corrected_flux_ratio_sq': ['sum', 'skew', ],
        'z_corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
#        'luminosity': ['min', 'max', 'mean', 'median',
#                 'std', 'var', 'skew', 'count', kurtosis],
#        'z_luminosity': ['min', 'max', 'mean', 'median',
#                 'std', 'var', 'skew', 'count', kurtosis],
        #        'minused_flux': ['min', 'max', 'mean', 'median',
        #                         'std', 'var', 'skew'],
        #        'normed_flux': ['mean', 'median', 'skew'],
        # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
    }

    fe_set_df = corrected_set_df.groupby('object_id').agg({**aggregations})
    fe_set_df.columns = pd.Index([e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
    return fe_set_df


def fe_set_df_detected(corrected_set_df): 
    detected_corrected_set_df = corrected_set_df[corrected_set_df.detected == 1]

    detected_aggregations = {
        'mjd': [get_max_min_diff, 'skew'],
    }

    fe_set_df = detected_corrected_set_df.groupby('object_id').agg({**detected_aggregations})
    fe_set_df.columns = pd.Index(['detected_' + e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])
    return fe_set_df


def fe_set_df_std_upper_and_lower(corrected_set_df): 
    object_flux_std_df = corrected_set_df[['object_id', 'flux']].\
        groupby('object_id').\
        std().\
        rename(columns={'flux': 'flux_std'})
    object_flux_mean_df = corrected_set_df[['object_id', 'flux']].\
        groupby('object_id').\
        mean().\
        rename(columns={'flux': 'flux_mean'})
    corrected_set_df = corrected_set_df.merge(
            object_flux_std_df, on='object_id', how='left')
    corrected_set_df = corrected_set_df.merge(
            object_flux_mean_df, on='object_id', how='left')

    std_upper_corrected_set_df = corrected_set_df[corrected_set_df.flux > 
                                    corrected_set_df.flux_std + corrected_set_df.flux_mean]
    std_lower_corrected_set_df = corrected_set_df[corrected_set_df.flux < 
                                    corrected_set_df.flux_std - corrected_set_df.flux_mean]

    std_upper_aggregations = {
        'mjd': [get_max_min_diff, 'var', 'skew', ],
#        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', kurtosis],
        'flux': ['count', 'min'],
        # 'mjd': ['min', 'max', 'var', ],
    }
    std_lower_aggregations = {
        'mjd': [get_max_min_diff, 'var', 'skew', ],
        'flux': ['count', 'max'],
    }

    std_upper_fe_set_df = std_upper_corrected_set_df.groupby('object_id').agg({**std_upper_aggregations})
    std_upper_fe_set_df.columns = pd.Index(['std_upper_' + e[0] + "_" + e[1] for e in std_upper_fe_set_df.columns.tolist()])
    std_lower_fe_set_df = std_lower_corrected_set_df.groupby('object_id').agg({**std_lower_aggregations})
    std_lower_fe_set_df.columns = pd.Index(['std_lower_' + e[0] + "_" + e[1] for e in std_lower_fe_set_df.columns.tolist()])

    fe_set_df = std_upper_fe_set_df.merge(std_lower_fe_set_df, on='object_id', how='left')

    return fe_set_df


def fe_set_df_passband(corrected_set_df):
    passband_aggregations = {
        # 'mjd': [diff_mean, diff_max],
        # 'phase': [diff_mean, diff_max],
        'flux': ['min', 'max', 'count', 'var', 'mean', 'skew', kurtosis, quantile10,quantile25, quantile75, quantile90, quantile2575_range, quantile1090_range, get_max_min_diff],
        'normed_flux': [diff_mean, ],
        #'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', kurtosis],
#        'flux_err': ['var'],
        'detected': ['mean', ],
        'flux_ratio_sq': ['sum', 'skew', 'max', 'min', get_max_min_diff],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
#        'luminosity': ['max', kurtosis],
        'corrected_flux': ['min', 'max', 'mean', 'median',
                           'std', 'var', 'skew', 
                            diff_var, get_max_min_diff],
        'z_corrected_flux': ['min', 'max', 'mean', 'median',
                           'std', 'var', 'skew', 
                            diff_var, get_max_min_diff],
    }

    fe_set_df = pd.DataFrame()
    passbands = [0, 1, 2, 3, 4, 5]
    for passband in passbands:
        _passband_set_df = corrected_set_df[corrected_set_df.passband == passband]

        # starter kit type fe
        starter_fe_series = _passband_set_df.\
            groupby('object_id').\
            apply(get_starter_features)
        starter_fe_df = starter_fe_series.\
            apply(lambda x: pd.Series(x)).\
            rename(columns={
                0: 'band-{}_wmean'.format(passband),
                1: 'band-{}_normed_std'.format(passband),
                2: 'band-{}_normed_amp'.format(passband),
                3: 'band-{}_normed_mad'.format(passband),
                4: 'band-{}_beyond_1std'.format(passband),
            })

        # the other aggregations
        band_fe_set_df = _passband_set_df.\
            groupby('object_id').\
            agg({**passband_aggregations})
        band_fe_set_df.columns = pd.Index(
            ['band-{}_'.format(passband) + e[0] + "_" + e[1]
             for e in band_fe_set_df.columns.tolist()])

        if fe_set_df.shape[0] != 0:
            fe_set_df = fe_set_df.merge(
                starter_fe_df, on='object_id', how='left')
        else:
            fe_set_df = starter_fe_df
        fe_set_df = fe_set_df.merge(
            band_fe_set_df, on='object_id', how='left')

    return fe_set_df


def fe_set_df_passband_std_upper(corrected_set_df):
    band_std_upper_flux_aggregations = {
        'mjd': [get_max_min_diff, 'var', 'skew', diff_mean],
        'flux': ['count', diff_mean, quantile10, quantile25, quantile75, quantile90, quantile2575_range, quantile1090_range],
    }

    fe_set_df = pd.DataFrame()
    passbands = [0, 1, 2, 3, 4, 5]
    for passband in passbands:
        _passband_set_df = corrected_set_df[corrected_set_df.passband == passband]

        band_object_flux_std_df = _passband_set_df[['object_id', 'flux']].\
            groupby('object_id').\
            std().\
            rename(columns={'flux': 'flux_std'})
        band_object_flux_mean_df = _passband_set_df[['object_id', 'flux']].\
            groupby('object_id').\
            mean().\
            rename(columns={'flux': 'flux_mean'})
        _passband_set_df = _passband_set_df.merge(
            band_object_flux_std_df, on='object_id', how='left')
        _passband_set_df = _passband_set_df.merge(
            band_object_flux_mean_df, on='object_id', how='left')
        band_std_upper_flux_df = _passband_set_df[_passband_set_df.flux >
                                                _passband_set_df.flux_std + 
                                                _passband_set_df.flux_mean]
        band_fe_std_upper_flux_df = band_std_upper_flux_df.groupby('object_id').\
            agg({**band_std_upper_flux_aggregations})
        band_fe_std_upper_flux_df.columns = pd.Index(
            ['band-{}_std_upper_'.format(passband) + e[0] + "_" + e[1]
                for e in band_fe_std_upper_flux_df.columns.tolist()])

        if fe_set_df.shape[0] != 0:
            fe_set_df = fe_set_df.merge(
                band_fe_std_upper_flux_df,  on='object_id', how='left')
        else:
            fe_set_df = band_fe_std_upper_flux_df

    return fe_set_df


def fe_set_df_passband_detected(corrected_set_df):
    band_detected_aggregations = {
        'mjd': [get_max_min_diff, 'var', 'skew', diff_mean],
    }

    fe_set_df = pd.DataFrame()
    passbands = [0, 1, 2, 3, 4, 5]
    for passband in passbands:
        _passband_set_df = corrected_set_df[corrected_set_df.passband == passband]

        band_detected_df = _passband_set_df[_passband_set_df.detected == 1]
        band_fe_detected_df = band_detected_df.groupby('object_id').\
            agg({**band_detected_aggregations})
        band_fe_detected_df.columns = pd.Index(
            ['band-{}_detected_'.format(passband) + e[0] + "_" + e[1]
                for e in band_fe_detected_df.columns.tolist()])

        if fe_set_df.shape[0] != 0:
            fe_set_df = fe_set_df.merge(
                band_fe_detected_df,  on='object_id', how='left')
        else:
            fe_set_df = band_fe_detected_df

    return fe_set_df


def _get_peak_mjd(df):
    return df[df.flux == df.flux.max()].iloc[0].mjd
     

def fe_set_df_peak_around(corrected_set_df):
    date_lwidths = [14, 14, 0, 30, 0, 30, 90, 0, 90]
    date_rwidths = [14, 0, 14, 30, 30, 0, 90, 90, 0]

    # detected しないと overfit する
    det_corrected_set_df = corrected_set_df.query('detected == 1')

    fe_set_df = pd.DataFrame(det_corrected_set_df.object_id.unique(), columns=['object_id'])
    for date_lwidth, date_rwidth in zip(date_lwidths, date_rwidths):
        peak_df = det_corrected_set_df.\
                merge(det_corrected_set_df.groupby('object_id').
                apply(_get_peak_mjd).
                reset_index().
                rename(columns={0: 'peak_mjd'}),
                on='object_id', how='left')
        peak_df = peak_df[
                (peak_df.mjd <= peak_df.peak_mjd + date_rwidth) &
                (peak_df.mjd >= peak_df.peak_mjd - date_lwidth)]
    
        peak_aggregations = {
            # 'passband': ['mean', 'std', 'var'],
            # 'mjd': ['max', 'min', 'var'],
            # 'mjd': [diff_mean, diff_max],
            # 'phase': [diff_mean, diff_max],
            'flux': ['min', 'max', 'mean', 'median',
                     'std', 'var', 'skew', 'count', kurtosis,
                     diff_var, get_max_min_diff],
#            'abs_magnitude': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis, 
#                     diff_var, get_max_min_diff],
            'corrected_flux': ['min', 'max', 'mean', 'median',
                               'std', 'var', 'skew', 
                               diff_var, get_max_min_diff],
            'flux_ratio_to_flux_err': ['min', 'max', ],
            'detected': ['mean', ],
            'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis, ],
            'flux_by_flux_ratio_sq': ['sum', 'skew', ],
            'corrected_flux_ratio_sq': ['sum', 'skew', ],
            'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
#            'luminosity': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis],
            #        'minused_flux': ['min', 'max', 'mean', 'median',
            #                         'std', 'var', 'skew'],
            #        'normed_flux': ['mean', 'median', 'skew'],
            # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
        }
    
        _fe_set_df = peak_df.groupby('object_id').agg({**peak_aggregations})
        _fe_set_df.columns = pd.Index([f'peak-{date_lwidth}-{date_rwidth}_' + e[0] + "_" + e[1] for e in _fe_set_df.columns.tolist()])
        fe_set_df = fe_set_df.merge(_fe_set_df, on='object_id', how='left')
    return fe_set_df


def fe_set_df_passband_peak_around(corrected_set_df):
    passbands = [0, 1, 2, 3, 4, 5]
    date_lwidths = [14, 14, 0, 30, 0, 30, 90, 0, 90]
    date_rwidths = [14, 0, 14, 30, 30, 0, 90, 90, 0]

    fe_set_df = pd.DataFrame(corrected_set_df.object_id.unique(), columns=['object_id'])
    for date_lwidth, date_rwidth in zip(date_lwidths, date_rwidths):
        for passband in passbands:
            print('a')
        peak_df = corrected_set_df.\
                merge(corrected_set_df.groupby('object_id').
                apply(_get_peak_mjd).
                reset_index().
                rename(columns={0: 'peak_mjd'}),
                on='object_id', how='left')
        peak_df = peak_df[
                (peak_df.mjd <= peak_df.peak_mjd + date_rwidth) &
                (peak_df.mjd >= peak_df.peak_mjd - date_lwidth)]
    
        passband_peak_aggregations = {
            # 'passband': ['mean', 'std', 'var'],
            # 'mjd': ['max', 'min', 'var'],
            # 'mjd': [diff_mean, diff_max],
            # 'phase': [diff_mean, diff_max],
            'flux': ['min', 'max', 'mean', 'median',
                     'std', 'var', 'skew', 'count', kurtosis,
                     diff_var],
#            'abs_magnitude': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis, 
#                     diff_var],
            'corrected_flux': ['min', 'max', 'mean', 'median',
                               'std', 'var', 'skew', 
                               diff_var],
            'z_corrected_flux': ['min', 'max', 'mean', 'median',
                               'std', 'var', 'skew', 
                               diff_var],
            'flux_ratio_to_flux_err': ['min', 'max', ],
            'detected': ['mean', ],
            'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis],
            'flux_by_flux_ratio_sq': ['sum', 'skew', ],
            'corrected_flux_ratio_sq': ['sum', 'skew', ],
            'z_corrected_flux_ratio_sq': ['sum', 'skew', ],
            'z_corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
#            'luminosity': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis],
            #        'minused_flux': ['min', 'max', 'mean', 'median',
            #                         'std', 'var', 'skew'],
            #        'normed_flux': ['mean', 'median', 'skew'],
            # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
        }
    
        _fe_set_df = peak_df.groupby('object_id').agg({**passband_peak_aggregations})
        _fe_set_df.columns = pd.Index([f'peak-{date_lwidth}-{date_rwidth}_' + e[0] + "_" + e[1] for e in _fe_set_df.columns.tolist()])
        fe_set_df = fe_set_df.merge(_fe_set_df, on='object_id', how='left')
    return fe_set_df


def _get_ratsq_peak_mjd(df):
    return df[df.flux_ratio_sq == df.flux_ratio_sq.max()].iloc[0].mjd
     
def fe_set_df_ratsq_peak_around(corrected_set_df):
    date_lwidths = [14, 14, 0, 30, 0, 30, 90, 0, 90]
    date_rwidths = [14, 0, 14, 30, 30, 0, 90, 90, 0]

    fe_set_df = pd.DataFrame(corrected_set_df.object_id.unique(), columns=['object_id'])
    for date_lwidth, date_rwidth in zip(date_lwidths, date_rwidths):
        peak_df = corrected_set_df.\
                merge(corrected_set_df.groupby('object_id').
                apply(_get_ratsq_peak_mjd).
                reset_index().
                rename(columns={0: 'peak_mjd'}),
                on='object_id', how='left')
        peak_df = peak_df[
                (peak_df.mjd <= peak_df.peak_mjd + date_rwidth) &
                (peak_df.mjd >= peak_df.peak_mjd - date_lwidth)]
    
        peak_aggregations = {
            # 'passband': ['mean', 'std', 'var'],
            # 'mjd': ['max', 'min', 'var'],
            # 'mjd': [diff_mean, diff_max],
            # 'phase': [diff_mean, diff_max],
            'flux': ['min', 'max', 'mean', 'median',
                     'std', 'var', 'skew', 'count', kurtosis,
                     diff_var, get_max_min_diff],
#            'abs_magnitude': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis, 
#                     diff_var, get_max_min_diff],
            'corrected_flux': ['min', 'max', 'mean', 'median',
                               'std', 'var', 'skew', 
                               diff_var, get_max_min_diff],
            'flux_ratio_to_flux_err': ['min', 'max', ],
            'detected': ['mean', ],
            'flux_ratio_sq': ['sum', 'skew', 'mean', kurtosis, 'var', get_max_min_diff],
            'flux_by_flux_ratio_sq': ['sum', 'skew', ],
            'corrected_flux_ratio_sq': ['sum', 'skew', ],
            'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
#            'luminosity': ['min', 'max', 'mean', 'median',
#                     'std', 'var', 'skew', 'count', kurtosis],
            #        'minused_flux': ['min', 'max', 'mean', 'median',
            #                         'std', 'var', 'skew'],
            #        'normed_flux': ['mean', 'median', 'skew'],
            # 'diff_flux_by_diff_mjd': ['min', 'max', 'var', ],
        }
    
        _fe_set_df = peak_df.groupby('object_id').agg({**peak_aggregations})
        _fe_set_df.columns = pd.Index([f'ratsq-peak-{date_lwidth}-{date_rwidth}_' + e[0] + "_" + e[1] for e in _fe_set_df.columns.tolist()])
        fe_set_df = fe_set_df.merge(_fe_set_df, on='object_id', how='left')
    return fe_set_df


def fe_set_df_my_skew_kurt(corrected_set_df):
    skew_df = corrected_set_df.groupby('object_id').\
            apply(calc_flux_mjd_skewness).\
            rename('my_skew')
    skew_df = (skew_df * 1e40).reset_index()
    kurt_df = corrected_set_df.groupby('object_id').\
            apply(calc_flux_mjd_kurtosis).\
            rename('my_kurt')
    kurt_df = (kurt_df * 1e55).reset_index()
    fe_set_df = skew_df.merge(kurt_df, on='object_id', how='left')

    # detected type
    det_skew_df = corrected_set_df.query('detected==1').groupby('object_id').\
            apply(calc_flux_mjd_skewness).\
            rename('det_my_skew')
    det_skew_df = (det_skew_df * 1e40).reset_index()
    det_kurt_df = corrected_set_df.query('detected==1').groupby('object_id').\
            apply(calc_flux_mjd_kurtosis).\
            rename('det_my_kurt')
    det_kurt_df = (det_kurt_df * 1e55).reset_index()
    fe_set_df = fe_set_df.merge(det_skew_df, on='object_id', how='left')
    fe_set_df = fe_set_df.merge(det_kurt_df, on='object_id', how='left')

    for passband in range(6):
        band_df = corrected_set_df[corrected_set_df.passband == passband]
        band_skew_df = band_df.\
            groupby('object_id').\
            apply(calc_flux_mjd_skewness).\
            rename(f'band-{passband}_my_skew')
        band_skew_df = (band_skew_df * 1e40).reset_index()
        band_kurt_df = band_df.\
            groupby('object_id').\
            apply(calc_flux_mjd_kurtosis).\
            rename(f'band-{passband}_my_kurt')
        band_kurt_df = (band_kurt_df * 1e55).reset_index()
        fe_set_df = fe_set_df.merge(band_skew_df, on='object_id', how='left')
        fe_set_df = fe_set_df.merge(band_kurt_df, on='object_id', how='left')
    return fe_set_df


def fe_set_df_deficits(corrected_set_df):
    det_mjd_diff = corrected_set_df[corrected_set_df['detected']==1].pivot_table('mjd','object_id',aggfunc=[min,max])
    det_mjd_diff.columns = ['min_mjd', 'max_mjd']
    # detected==1の前後の間隔を追加
    mjd_diff_ = corrected_set_df[['object_id','mjd']].merge(right=det_mjd_diff, on=['object_id'], how='left')
    max_mjd_bf_det1 = mjd_diff_[mjd_diff_.mjd < mjd_diff_.min_mjd].groupby('object_id')[['object_id','mjd', 'min_mjd']].max().rename(columns={'mjd': 'max_mjd_bf_det1'})
    mjd_diff_bf_det1 = max_mjd_bf_det1['min_mjd'] - max_mjd_bf_det1['max_mjd_bf_det1']
    mjd_diff_bf_det1 = mjd_diff_bf_det1.rename('mjd_diff_bf_det1').reset_index()
    min_mjd_af_det1 = mjd_diff_[mjd_diff_.mjd > mjd_diff_.max_mjd].groupby('object_id')[['object_id','mjd', 'max_mjd']].min().rename(columns={'mjd': 'min_mjd_af_det1'})
    mjd_diff_af_det1 = min_mjd_af_det1['min_mjd_af_det1'] - min_mjd_af_det1['max_mjd'] 
    mjd_diff_af_det1 = mjd_diff_af_det1.rename('mjd_diff_af_det1').reset_index()

    fe_set_df = mjd_diff_bf_det1.merge(mjd_diff_af_det1, on ='object_id', how='left')
    fe_set_df['mjd_diff_ab_sum'] = fe_set_df['mjd_diff_af_det1'] + fe_set_df['mjd_diff_bf_det1']
    return fe_set_df.set_index('object_id')


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
        set_df_name = [f'test_set_{i}_df' for i in range(62)]
        set_dfs = [self.src_df_dict[f].copy() for f in set_df_name]
        #set_dfs = [self.src_df_dict[f] for f in set_df_name]
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


def fe_meta(meta_df):
    # band feature engineerings 
    passbands = [0, 1, 2, 3, 4, 5]
    for passband in passbands:
        meta_df[f'band-{passband}_flux_count_ratio'] = \
            meta_df[f'band-{passband}_flux_count'] / meta_df['flux_count']
        meta_df[f'band-{passband}_std_upper_flux_count_ratio'] = \
            meta_df[f'band-{passband}_std_upper_flux_count'] / meta_df['flux_count']
        meta_df[f'band-{passband}_flux_ratio_sq_max_ratio'] = \
            meta_df[f'band-{passband}_flux_ratio_sq_max'] / meta_df['flux_ratio_sq_max']

        # starter type fe
        lpb = passband
        rpb = (lpb + 1) % 6
        lMean = meta_df['band-{}_wmean'.format(lpb)]
        rMean = meta_df['band-{}_wmean'.format(rpb)]
        lstd = meta_df['band-{}_normed_std'.format(lpb)]
        rstd = meta_df['band-{}_normed_std'.format(rpb)]
        lamp = meta_df['band-{}_normed_amp'.format(lpb)]
        ramp = meta_df['band-{}_normed_amp'.format(rpb)]
        lmad = meta_df['band-{}_normed_mad'.format(lpb)]
        rmad = meta_df['band-{}_normed_mad'.format(rpb)]
        l1std = meta_df['band-{}_beyond_1std'.format(lpb)]
        r1std = meta_df['band-{}_beyond_1std'.format(rpb)]
        ldmgmmd = meta_df[f'band-{lpb}_detected_mjd_get_max_min_diff']
        rdmgmmd = meta_df[f'band-{rpb}_detected_mjd_get_max_min_diff']
        lskew = meta_df[f'band-{lpb}_flux_skew']
        rskew = meta_df[f'band-{rpb}_flux_skew']
        lkurt = meta_df[f'band-{lpb}_flux_kurtosis']
        rkurt = meta_df[f'band-{rpb}_flux_kurtosis']
        lq2575_rng = meta_df[f'band-{lpb}_flux_quantile2575_range']
        rq2575_rng = meta_df[f'band-{rpb}_flux_quantile2575_range']
        lmax = meta_df['band-{}_flux_max'.format(lpb)]
        rmax = meta_df['band-{}_flux_max'.format(rpb)]
        lratsqmax = meta_df['band-{}_flux_ratio_sq_max'.format(lpb)]
        rratsqmax = meta_df['band-{}_flux_ratio_sq_max'.format(rpb)]
        rcorrmax = meta_df['band-{}_corrected_flux_max'.format(rpb)]
        lcorrmax = meta_df['band-{}_corrected_flux_max'.format(lpb)]
        rzcorrmax = meta_df['band-{}_z_corrected_flux_max'.format(rpb)]
        lzcorrmax = meta_df['band-{}_z_corrected_flux_max'.format(lpb)]
        mean_diff = -2.5 * np.log10(lMean / rMean)
        std_diff = lstd - rstd
        amp_diff = lamp - ramp
        mad_diff = lmad-rmad
        beyond_diff = l1std-r1std
        dmgmmd_diff = ldmgmmd - rdmgmmd
        skew_diff = lskew - rskew
        kurt_diff = lkurt - rkurt
        q2575_rng_diff = lq2575_rng - rq2575_rng
        max_diff = lmax - rmax
        ratsqmax_diff = lratsqmax - rratsqmax
        corrmax_diff = lcorrmax - rcorrmax
        zcorrmax_diff = lzcorrmax - rzcorrmax
        ratsqmax_diff_log = -2.5 * np.log10(lratsqmax/rratsqmax)
        mean_diff_colname = '{}_minus_{}_wmean'.format(lpb, rpb)
        std_diff_colname = '{}_minus_{}_std'.format(lpb, rpb)
        amp_diff_colname = '{}_minus_{}_amp'.format(lpb, rpb)
        mad_diff_colname = '{}_minus_{}_mad'.format(lpb, rpb)
        beyond_diff_colname = '{}_minus_{}_beyond'.format(lpb, rpb)
        dmgmmd_diff_colname = f'{lpb}_minus_{rpb}_dmgmmd'
        skew_diff_colname = f'{lpb}_minus_{rpb}_skew'
        kurt_diff_colname = f'{lpb}_minus_{rpb}_kurt'
        q2575_rng_diff_colname = f'{lpb}_minus_{rpb}_q2575_rng'
        max_diff_colname = f'{lpb}_minus_{rpb}_max'
        ratsqmax_diff_colname = f'{lpb}_minus_{rpb}_ratsqmax'
        ratsqmax_diff_log_colname = f'{lpb}_minus_{rpb}_ratsqmax_log'
        corrmax_diff_colname = f'{lpb}_minus_{rpb}_corrmax_diff'
        zcorrmax_diff_colname = f'{lpb}_minus_{rpb}_zcorrmax_diff'
        meta_df[mean_diff_colname] = mean_diff
        meta_df[std_diff_colname] = std_diff
        meta_df[amp_diff_colname] = amp_diff
        meta_df[dmgmmd_diff_colname] = dmgmmd_diff
        meta_df[skew_diff_colname] = skew_diff
        meta_df[kurt_diff_colname] = kurt_diff
        meta_df[q2575_rng_diff_colname] = q2575_rng_diff
        meta_df[max_diff_colname] = max_diff
        meta_df[ratsqmax_diff_colname] = ratsqmax_diff
        meta_df[ratsqmax_diff_log_colname] = ratsqmax_diff_log
        meta_df[corrmax_diff_colname] = corrmax_diff
        meta_df[zcorrmax_diff_colname] = zcorrmax_diff

    # non band feature engineering
    meta_df['flux_diff'] = meta_df['flux_max'] - meta_df['flux_min']
    meta_df['flux_dif2'] = (meta_df['flux_max'] - meta_df['flux_min'])\
        / meta_df['flux_mean']
    meta_df['flux_w_mean'] = meta_df['flux_by_flux_ratio_sq_sum'] / \
        meta_df['flux_ratio_sq_sum']
    meta_df['flux_dif3'] = (meta_df['flux_max'] - meta_df['flux_min'])\
        / meta_df['flux_w_mean']
    meta_df['corrected_flux_diff'] = meta_df['corrected_flux_max'] - meta_df['corrected_flux_min']
    meta_df['corrected_flux_dif2'] = (meta_df['corrected_flux_max'] - meta_df['corrected_flux_min'])\
        / meta_df['corrected_flux_mean']
    meta_df['corrected_flux_w_mean'] = meta_df['corrected_flux_by_flux_ratio_sq_sum'] / \
        meta_df['corrected_flux_ratio_sq_sum']
    meta_df['corrected_flux_dif3'] = (meta_df['corrected_flux_max'] - meta_df['corrected_flux_min'])\
        / meta_df['corrected_flux_w_mean']
    meta_df['z_corrected_flux_diff'] = meta_df['z_corrected_flux_max'] - meta_df['z_corrected_flux_min']
    meta_df['z_corrected_flux_dif2'] = (meta_df['z_corrected_flux_max'] - meta_df['z_corrected_flux_min'])\
        / meta_df['z_corrected_flux_mean']
    meta_df['z_corrected_flux_w_mean'] = meta_df['z_corrected_flux_by_flux_ratio_sq_sum'] / \
        meta_df['z_corrected_flux_ratio_sq_sum']
    meta_df['z_corrected_flux_dif3'] = (meta_df['z_corrected_flux_max'] - meta_df['z_corrected_flux_min'])\
        / meta_df['z_corrected_flux_w_mean']

    meta_df['std_upper_rat'] = meta_df['std_upper_flux_count'] / meta_df['flux_count']

    passband_flux_maxes = \
        ['band-{}_flux_max'.format(i) for i in passbands]
    # meta_df['passband_flux_maxes_var'] = \
    #     meta_df[passband_flux_maxes].var(axis=1)
    for passband_flux_max in passband_flux_maxes:
        meta_df[passband_flux_max + '_ratio_to_the_max'] = \
            meta_df[passband_flux_max] / meta_df['flux_max']
#    passband_maxes = meta_df[passband_flux_maxes].values
#    passband_maxes_argmaxes = np.argmax(passband_maxes, axis=1)
#    meta_df['passband_maxes_argmaxes'] = passband_maxes_argmaxes
#        meta_df[passband_flux_max + '_from_the_max'] = \
#             meta_df['flux_max'] - meta_df[passband_flux_max]
#    passband_flux_maxes_from_the_max = \
#        ['band-{}_flux_max_from_the_max'.format(i) for i in passbands]
#    passband_flux_maxes_from_the_max_value = meta_df[passband_flux_maxes_from_the_max].values
#    passband_flux_maxes_from_the_max_value.sort(axis=1)
#    meta_df['2nd_passband_flux_max_diff'] = passband_flux_maxes_from_the_max_value[:,1]
#    meta_df['3rd_passband_flux_max_diff'] = passband_flux_maxes_from_the_max_value[:,2]
#    meta_df['2nd_passband_flux_max_diff_rat'] = meta_df['2nd_passband_flux_max_diff'] / meta_df.flux_max
#    meta_df['3rd_passband_flux_max_diff_rat'] = meta_df['3rd_passband_flux_max_diff'] / meta_df.flux_max
    passband_flux_mins = \
        ['band-{}_flux_min'.format(i) for i in passbands]
    meta_df['passband_flux_min_var'] = \
        meta_df[passband_flux_mins].var(axis=1)
    # for passband_flux_min in passband_flux_mins:
    #     meta_df[passband_flux_min + '_ratio_to_the_min'] = \
    #          meta_df[passband_flux_min] / meta_df['flux_min']
    passband_flux_means = \
        ['band-{}_flux_mean'.format(i) for i in passbands]
    meta_df['passband_flux_means_var'] = \
        meta_df[passband_flux_means].var(axis=1)
    passband_flux_counts = \
        ['band-{}_flux_count_ratio'.format(i) for i in passbands]
    meta_df['passband_flux_counts_var'] = \
        meta_df[passband_flux_counts].var(axis=1)
    passband_detected_means = \
        ['band-{}_detected_mean'.format(i) for i in passbands]
    meta_df['passband_detected_means_var'] = \
        meta_df[passband_detected_means].var(axis=1)
    # passband_flux_ratio_sq_sum = \
    #    ['band-{}_flux_ratio_sq_sum'.format(i) for i in passbands]
    # meta_df['passband_flux_ratio_sq_sum_var'] = \
    #    meta_df[passband_flux_ratio_sq_sum].var(axis=1)
    # passband_flux_ratio_sq_skew = \
    #    ['band-{}_flux_ratio_sq_skew'.format(i) for i in passbands]
    # meta_df['passband_flux_ratio_sq_skew_var'] = \
    #    meta_df[passband_flux_ratio_sq_skew].var(axis=1)
    # band $B$N7gB;N($N(B var $B$H$+$bNI$5$=$&(B
    passband_flux_vars = \
        ['band-{}_flux_var'.format(i) for i in passbands]
    passband_flux_diffs = \
        ['band-{}_flux_get_max_min_diff'.format(i) for i in passbands]
    meta_df['band_flux_diff_max'] = meta_df[passband_flux_diffs].max(axis=1)
    meta_df['band_flux_diff_min'] = meta_df[passband_flux_diffs].min(axis=1)
    meta_df['band_flux_diff_diff'] = meta_df['band_flux_diff_max'] - meta_df['band_flux_diff_min']
    meta_df['band_flux_diff_diff_rat'] = meta_df['band_flux_diff_diff'] / meta_df['band_flux_diff_max']
    meta_df['band_flux_max_min_rat'] = meta_df['band_flux_diff_min'] / meta_df['band_flux_diff_max']
    meta_df['internal'] = meta_df.hostgal_photoz == 0.
    meta_df['lumi_dist'] = 10**((meta_df.distmod+5)/5)

    # peak around features
    meta_df['peak_kurt_14to30'] = meta_df['peak-14-14_flux_kurtosis'] - meta_df['peak-30-30_flux_kurtosis']
    meta_df['peak_kurt_14to90'] = meta_df['peak-14-14_flux_kurtosis'] - meta_df['peak-90-90_flux_kurtosis']
    meta_df['peak_kurt_30to90'] = meta_df['peak-30-30_flux_kurtosis'] - meta_df['peak-90-90_flux_kurtosis']
    meta_df['peak_skew_14to30'] = meta_df['peak-14-14_flux_skew'] - meta_df['peak-30-30_flux_skew']
    meta_df['peak_skew_14to90'] = meta_df['peak-14-14_flux_skew'] - meta_df['peak-90-90_flux_skew']
    meta_df['peak_skew_30to90'] = meta_df['peak-30-30_flux_skew'] - meta_df['peak-90-90_flux_skew']

    return meta_df


class featureCreatorMeta(featureCreator):
    def __init__(self, fe_set_df, set_res_df_name, load_dir, save_dir, 
            src_df_dict=None, logger=None, nthread=1, train=True):
        super(featureCreatorMeta, self).\
                __init__(load_dir=load_dir,
                        save_dir=save_dir,
                        src_df_dict=src_df_dict,
                        logger=logger,
                        nthread=nthread)
        self.fe_set_df = fe_set_df
        self.set_res_df_name = set_res_df_name
        self.train = train
        if self.train:
            self.meta_file = '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/training_set_metadata.csv'
        else:
            self.meta_file = '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/test_set_metadata.csv'

    def _load(self):
        path_dict = {
                'meta_features': self.meta_file,
                'set_base_features': self.save_dir + 'set_base_features.ftr',
                'set_passband_std_upper_features': self.save_dir + 'set_passband_std_upper_features.ftr',
                'set_passband_detected_features': self.save_dir + 'set_passband_detected_features.ftr',
                'set_detected_features': self.save_dir + 'set_detected_features.ftr',
                'set_std_upper_and_lower_features': self.save_dir + 'set_std_upper_and_lower_features.ftr',
                'set_passband_features': self.save_dir + 'set_passband_features.ftr',
                'set_tsfresh_features': self.save_dir + 'set_tsfresh_features.ftr',
                'set_peak_around_features': self.save_dir + 'set_peak_around_features.ftr',
                'set_ratsq_peak_around_features': self.save_dir + 'set_ratsq_peak_around_features.ftr',
                'set_skkt_features': self.save_dir + 'set_skkt_features.ftr',
                'set_deficits_features': self.save_dir + 'set_deficits_features.ftr',
        }
        self._load_dfs_from_paths(path_dict=path_dict)

        self.src_df_dict['merged_meta_df'] = self.src_df_dict['meta_features']
        self._log_print('merging meta dfs ...')
        for key in tqdm(self.src_df_dict.keys()):
            print(key)
            if key == 'meta_features' or key == 'merged_meta_df':
                continue
            self.src_df_dict['merged_meta_df'] = self.src_df_dict['merged_meta_df'].\
                    merge(self.src_df_dict[key], on='object_id', how='left')

        if self.train:
            okumura_df1 = pd.read_pickle('../lcfit/LCfit_feature_train_v4_20181205.pkl.gz', compression='gzip')
            self.src_df_dict['merged_meta_df'] = self.src_df_dict['merged_meta_df'].\
                    merge(okumura_df1, on='object_id', how='left')
            del okumura_df1
            okumura_df2 = pd.read_pickle('../lcfit/okumurasan_feats/LCfit_feature_allSN_i_train_v2_20181213.pkl.gz', compression='gzip')
            self.src_df_dict['merged_meta_df'] = self.src_df_dict['merged_meta_df'].\
                    merge(okumura_df2, on='object_id', how='left')
            del okumura_df2
            gc.collect()
        else:
            okumura_df1 = pd.read_pickle('../lcfit/LCfit_feature_test_v4_20181205.pkl.gz', compression='gzip')
            self.src_df_dict['merged_meta_df'] = self.src_df_dict['merged_meta_df'].\
                    merge(okumura_df1, on='object_id', how='left')
            del okumura_df1
            okumura_df2 = pd.read_pickle('../lcfit/okumurasan_feats/LCfit_feature_allSN_i_test_v2_20181213.pkl.gz', compression='gzip')
            self.src_df_dict['merged_meta_df'] = self.src_df_dict['merged_meta_df'].\
                    merge(okumura_df2, on='object_id', how='left')
            del okumura_df2
            gc.collect()

    def _create_features(self):
        object_ids = self.src_df_dict['merged_meta_df'].object_id.unique()
        meta_dfs = [self.src_df_dict['merged_meta_df'][
            self.src_df_dict['merged_meta_df'].object_id.isin(obj_id_grp)]
            for obj_id_grp in np.array_split(object_ids, 62)]
        with Pool(self.nthread) as p:
            self._log_print('start fature engineering ...')
            set_res_list = p.map(self.fe_set_df, meta_dfs)
            p.close()
            p.join()
            set_res_df = pd.concat(set_res_list, axis=0)
            gc.collect()

        # set the result in df_dict
        set_res_df.reset_index(inplace=True, drop=True)
        self._log_print(set_res_df.columns.tolist())
        self.df_dict[self.set_res_df_name] = set_res_df


class featureCreatorTsfresh(featureCreator):
    def __init__(self, load_dir, save_dir, src_df_dict=None, logger=None, nthread=1, train=True):
        super(featureCreatorTsfresh, self).\
                __init__(load_dir=load_dir,
                        save_dir=save_dir,
                        src_df_dict=src_df_dict,
                        logger=logger,
                        nthread=nthread)
        self.train = train

    def _load(self):
        if self.train:
            path_dict = {
                    'set_df': self.load_dir + 'training_set.csv'}
        else:
            path_dict = {'set_df': self.load_dir + 'test_set.fth'}
        self._load_dfs_from_paths(path_dict=path_dict)

    def _get_tsfresh_feats(self, set_df, nthread):
        # tsfresh features
        fcp = {
            'flux': {
                'longest_strike_above_mean': None,
                'longest_strike_below_mean': None,
                'mean_change': None,
                'mean_abs_change': None,
                'length': None,
    #            'number_peaks': [{'n': 1}],
    #            'fft_coefficient': [
    #                    {'coeff': 0, 'attr': 'abs'}, 
    #                    {'coeff': 1, 'attr': 'abs'}
    #                ],
    #            'binned_entropy': [{'max_bin': 20}],
    #            'agg_linear_trend': None,
    #            'number_cwt_peaks': None,
            },
                    
            'flux_by_flux_ratio_sq': {
                'longest_strike_above_mean': None,
                'longest_strike_below_mean': None,       
            },
                    
            'mjd': {
                'maximum': None, 
                'minimum': None,
                'mean_change': None,
                'mean_abs_change': None,
            },
        }

        # ts_flesh features
        fe_set_df = extract_features(
                set_df, 
                column_id='object_id', 
                column_sort='mjd', 
                column_kind='passband', 
                column_value = 'flux', 
                default_fc_parameters = fcp['flux'], 
                n_jobs=nthread)

        return fe_set_df

    def _create_features(self):
        set_res_df = self._get_tsfresh_feats(self.src_df_dict['set_df'], self.nthread).\
                reset_index().\
                rename(columns={'id': 'object_id'})
        # set the result in df_dict
        set_res_df.reset_index(inplace=True, drop=True)
        self.df_dict['set_tsfresh_features'] = set_res_df
