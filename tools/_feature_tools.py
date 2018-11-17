import pandas as pd
import numpy as np
from scipy import signal

import gc
from multiprocessing import Pool
from tqdm import tqdm
import warnings

import cesium.featurize as featurize
from tsfresh.feature_extraction import extract_features


warnings.simplefilter('ignore', RuntimeWarning)
np.random.seed(71)


# =======================================
# util functions
# =======================================
def split_idxes(df, nthread, logger, nclass=14):
    logger.info('calculating uniq object_id num')
    object_ids = df.object_id.unique()
    logger.info('getting groups')
    groups = np.array_split(object_ids, nclass)
    logger.info('splitting df')
    idxes = [df[df.object_id.isin(group)].index for group in groups]
    return idxes


def get_group_df(df_and_group):
    df, group = df_and_group
    return df[df.object_id.isin(set(group))]


def split_dfs(df, nthread, logger, save_flg=False):
    logger.info('calculating uniq object_id num')
    object_ids = df.object_id.unique()
    logger.info('getting groups')
    groups = np.array_split(object_ids, nthread)
    logger.info('splitting df')
    dfs = []
    for group in tqdm(list(groups)):
        dfs.append(df[df.object_id.isin(set(group))])
    if save_flg:
        logger.info('saving the split dfs...')
        for i, df in tqdm(list(enumerate(dfs))):
            df.reset_index().to_feather('./test_dfs/{}.fth'.format(i))
    return dfs


def load_test_set_dfs(nthread, logger):
    logger.info('loading dfs...')
    dfs_paths = [
        '/home/naoya.taguchi/workspace/kaggle/plasticc-2018/test_dfs/{}.fth'.format(i) for i in range(62)]
    p = Pool(nthread)
    dfs = p.map(pd.read_feather, dfs_paths)
    p.close()
    p.join()
    logger.info('done')
    return dfs


# def normalize_flux(set_df, new_flux_name='flux'):
#     normalize_base_df = set_df.groupby('object_id').\
#         flux.median().\
#         reset_index().\
#         rename(columns={'flux': 'flux_median'})
#     normalize_bases = set_df.merge(
#         normalize_base_df,
#         on='object_id',
#         how='left').flux_median
#     set_df[new_flux_name] = set_df.flux
#     set_df[new_flux_name] /= normalize_bases
#     return set_df


def _normalize_flux(set_df):
    flux_band_stat_df = set_df.groupby(['object_id', 'passband']).\
        agg({'flux': ['mean', 'std']}).\
        reset_index()
    flux_band_stat_df.columns = pd.Index(
        [e[0] + "_" + e[1] for e in flux_band_stat_df.columns.tolist()])
    stats_for_normalize = set_df.merge(
        flux_band_stat_df,
        on=['object_id', 'passband'],
        how='left')
    set_df['flux'] -= stats_for_normalize.flux_mean
    set_df['flux'] /= stats_for_normalize.flux_std
    del flux_band_stat_df, stats_for_normalize
    gc.collect()
    return set_df


def normalise(ts):
    return (ts - ts.mean()) / ts.std()


def get_phase_features(set_df):
    groups = set_df[['object_id', 'passband', 'mjd', 'flux', 'flux_err']].\
        groupby(['object_id', 'passband'])
    # times = groups.apply(lambda block: block['phase'].values).\
    times = groups.apply(lambda block: block['mjd'].values).\
        reset_index().\
        rename(columns={0: 'seq'})
    flux = groups.apply(lambda block: normalise(block['flux']).values).\
        reset_index().\
        rename(columns={0: 'seq'})
    flux_err = groups.apply(lambda block: normalise(block['flux_err']).values).\
        reset_index().\
        rename(columns={0: 'seq'})
    times_list = times.groupby('object_id').\
        apply(lambda x: x['seq'].tolist()).\
        tolist()
    flux_list = flux.groupby('object_id').\
        apply(lambda x: x['seq'].tolist()).\
        tolist()
    flux_err_list = flux_err.groupby('object_id').\
        apply(lambda x: x['seq'].tolist()).\
        tolist()
    warnings.simplefilter('ignore', RuntimeWarning)
    phase_df = featurize.\
        featurize_time_series(times=times_list,
                              values=flux_list,
                              errors=flux_err_list,
                              features_to_use=[
#                                              'amplitude',
                                              'freq1_freq',
#                                               'freq1_signif',
#                                               'freq1_amplitude1',
#                                               'freq2_freq',
#                                               'freq2_amplitude1',
#                                               'percent_beyond_1_std',
#                                               'freq3_freq',
###                                               'flux_percentile_ratio_mid20',
###                                               'max_slope',
#                                               'period_fast'
###                                               'qso_log_chi2_qsonu', 
                                               ],
                              scheduler=None)
#    print(phase_df.head(10))
    phase_df.columns = [str(e[0]) + '_' + str(e[1])
                        for e in phase_df.columns.tolist()]
    phase_df['object_id'] = times.object_id
    del times, flux, times_list, flux_list
    gc.collect()
    return phase_df


def _get_astro_distance(z, c=299790, h=67.15):
    # http://micha072.blog.fc2.com/blog-entry-1378.html
    _pow_z = np.power(z+1, 2)
    v = c * (-1 + _pow_z) / (1 + _pow_z)
    d = v / h
    return d


def _get_pogson_magnitude(flux):
    return 22.5 - 2.5 * np.log10(flux)


def add_corrected_flux(set_df, set_metadata_df):
    # _set_metadata_df = set_metadata_df[
    #         (set_metadata_df.hostgal_photoz_err < 0.5) &
    #         (set_metadata_df.hostgal_photoz_err > 0.)]
    _set_metadata_df = set_metadata_df
    set_df = set_df.merge(
        _set_metadata_df[['object_id', 'hostgal_photoz']],
        on='object_id',
        how='left')
    # set_df['corrected_flux'] = set_df.flux * (set_df.hostgal_photoz.apply(_get_astro_distance)**2)
    set_df['corrected_flux'] = set_df.flux * (set_df.hostgal_photoz**2)
    set_df['pogson_magnitude'] = set_df.flux.apply(_get_pogson_magnitude)
    # set_df['corrected_flux'] = set_df.flux / (set_df.hostgal_photoz**2)
    return set_df


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


def diff_mean(x):
    return x.diff().mean()


def diff_max(x):
    return x.diff().max()


def diff_min(x):
    return x.diff().min()


def diff_std(x):
    return x.diff().std()


def diff_sum(x):
    return x.diff().sum()


def get_max_min_diff(x):
    return x.max() - x.min()


# =======================================
# feature engineering part
# =======================================
def _for_set_df(set_df):
    # set_df = normalize_flux(set_df)
    # min_fluxes = set_df.groupby('object_id').\
    #        flux.min().\
    #        reset_index().\
    #        rename(columns={'flux': '_temp_flux_min'})
    # set_df = set_df.merge(min_fluxes, on='object_id', how='left')
    # set_df['minused_flux'] = set_df.flux - set_df._temp_flux_min
    # set_df.flux -= 0.

    # 25 は大体 train の平均
    #    set_df = set_df[set_df.flux_err < 25]
    set_df['flux_ratio_to_flux_err'] = \
        set_df['flux'] / set_df['flux_err']

    # 'kurtosis' は使えない...？
    aggregations = {
        # 'passband': ['mean', 'std', 'var'],
        # 'mjd': ['max', 'min', 'var'],
        # 'mjd': [diff_mean, diff_max],
        # 'phase': [diff_mean, diff_max],
        'flux': ['min', 'max', 'mean', 'median',
                 'std', 'var', 'skew', 'count'],
####        'corrected_flux': ['min', 'max', 'mean', 'median', 'skew', ],
####        'pogson_magnitude': ['min', 'max', 'mean', 'median', 'skew', ],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew'],
        'flux_ratio_to_flux_err': ['min', 'max', ],
        'detected': ['mean', ],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
####        'corrected_flux_ratio_sq': ['sum', 'skew'],
####        'corrected_flux_by_flux_ratio_sq': ['sum', 'skew'],
        #        'minused_flux': ['min', 'max', 'mean', 'median',
        #                         'std', 'var', 'skew'],
        #        'normed_flux': ['mean', 'median', 'skew'],
    }

    detected_aggregations = {
        'mjd': [get_max_min_diff, 'var', ],
    }

    # non_detected_aggregations = {
    #     'flux': ['var'],
    # }

    mean_upper_flux_aggregations = {
        'mjd': [get_max_min_diff, 'var', ],
        'flux': ['mean', ]
        # 'phase': [get_max_min_diff, 'var', ],
        # 'mjd': ['min', 'max', 'var', ],
    }

    passband_aggregations = {
        'flux': ['min', 'max', 'count', 'var', 'mean', 'skew', ],
        'detected': ['mean', ],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
    }

    # === run aggregations ===
    # fe before agggregations
    set_df['flux_ratio_sq'] = np.power(
        set_df['flux'] / set_df['flux_err'], 2.0)
    set_df['flux_by_flux_ratio_sq'] = set_df['flux'] * \
        set_df['flux_ratio_sq']
####    set_df['corrected_flux_ratio_sq'] = np.power(
####        set_df['corrected_flux'] / set_df['flux_err'], 2.0)
####    set_df['corrected_flux_by_flux_ratio_sq'] = set_df['corrected_flux'] * \
####        set_df['flux_ratio_sq']

    fe_set_df = set_df.groupby('object_id').agg({**aggregations})
    fe_set_df.columns = pd.Index(
        [e[0] + "_" + e[1] for e in fe_set_df.columns.tolist()])

    # === run mean upper aggregation ===
    # 平均値より高い位置にある flux の mjd 的距離を使うために加工。
    # 要は period を表現したい。
    object_flux_mean_df = set_df[['object_id', 'flux']].\
        groupby('object_id').\
        mean().\
        rename(columns={'flux': 'flux_mean'})
    mean_upper_flux_df = set_df.merge(
        object_flux_mean_df, on='object_id', how='left')
    mean_upper_flux_df = mean_upper_flux_df[mean_upper_flux_df.flux >
                                            mean_upper_flux_df.flux_mean]

    fe_mean_upper_flux_df = mean_upper_flux_df.groupby('object_id').\
        agg({**mean_upper_flux_aggregations})
    fe_mean_upper_flux_df.columns = pd.Index(
        ['mean_upper_' + e[0] + "_" + e[1]
            for e in fe_mean_upper_flux_df.columns.tolist()])
#    fe_mean_upper_flux_df['mean_upper_mjd_diff'] = \
#        fe_mean_upper_flux_df['mean_upper_mjd_max'] - \
#        fe_mean_upper_flux_df['mean_upper_mjd_min']
#    fe_mean_upper_flux_df.drop(['mjd_max', 'mjd_min'], axis=1, inplace=True)
    fe_set_df = fe_set_df.merge(
        fe_mean_upper_flux_df,
        on='object_id',
        how='left')
    del object_flux_mean_df, mean_upper_flux_df, fe_mean_upper_flux_df
    gc.collect()

    # === detected aggregation ===
    detected_df = set_df[set_df.detected == 1]
    fe_detected_df = detected_df.groupby('object_id').\
        agg({**detected_aggregations})
    fe_detected_df.columns = pd.Index(
        ['detected_' + e[0] + "_" + e[1]
            for e in fe_detected_df.columns.tolist()])
    fe_set_df = fe_set_df.merge(
        fe_detected_df,
        on='object_id',
        how='left')
    del detected_df, fe_detected_df
    gc.collect()

    # === non_detected aggregation ===
    # non_detected_df = set_df[set_df.detected == 0]
    # fe_non_detected_df = non_detected_df.groupby('object_id').\
    #     agg({**non_detected_aggregations})
    # fe_non_detected_df.columns = pd.Index(
    #     ['non_detected_' + e[0] + "_" + e[1]
    #         for e in fe_non_detected_df.columns.tolist()])
    # fe_set_df = fe_set_df.merge(
    #     fe_non_detected_df,
    #     on='object_id',
    #     how='left')
    # del non_detected_df, fe_non_detected_df
    # gc.collect()

    # === passband ごとに処理 ===
    passband_df = pd.DataFrame(fe_set_df[['flux_count', 'flux_mean']])
    passbands = [0, 1, 2, 3, 4, 5]
    for passband in passbands:
        band_prefix = 'band-{}_'.format(passband)
        # _passband_set_df = normalize_flux(set_df[set_df.passband == passband])
        _passband_set_df = set_df[set_df.passband == passband]
        # starter kit type fe
        starter_fe_series = _passband_set_df.\
            groupby('object_id').\
            apply(get_starter_features)
        starter_fe_df = starter_fe_series.\
            apply(lambda x: pd.Series(x)).\
            rename(columns={
                0: band_prefix + 'wmean',
                1: band_prefix + 'normed_std',
                2: band_prefix + 'normed_amp',
                3: band_prefix + 'normed_mad',
                4: band_prefix + 'beyond_1std',
            })

        # aggregation type fe
        band_fe_set_df = _passband_set_df.\
            groupby('object_id').\
            agg({**passband_aggregations})
        band_fe_set_df.columns = pd.Index(
            ['band-{}_'.format(passband) + e[0] + "_" + e[1]
             for e in band_fe_set_df.columns.tolist()])
        band_fe_set_df[band_prefix + 'flux_diff'] = \
            band_fe_set_df[band_prefix + 'flux_max'] - \
            band_fe_set_df[band_prefix + 'flux_min']

        # feature 過多なので drop
        passband_df = passband_df.merge(
            starter_fe_df, on='object_id', how='left')
        passband_df = passband_df.merge(
            band_fe_set_df, on='object_id', how='left')
#        passband_df['band-{}_flux_count'.format(passband)] = \
#            passband_df['band-{}_flux_count'.format(passband)]\
#            / passband_df['flux_count']

    # feature engineering for passband_df
    for lpb in passbands:
        rpb = (lpb + 1) % 6
        lMean = passband_df['band-{}_wmean'.format(lpb)]
        rMean = passband_df['band-{}_wmean'.format(rpb)]
        lstd = passband_df['band-{}_normed_std'.format(lpb)]
        rstd = passband_df['band-{}_normed_std'.format(rpb)]
        lamp = passband_df['band-{}_normed_amp'.format(lpb)]
        ramp = passband_df['band-{}_normed_amp'.format(rpb)]
        # lmad = passband_df['band-{}_normed_mad'.format(lpb)]
        # rmad = passband_df['band-{}_normed_mad'.format(rpb)]
        # l1std = passband_df['band-{}_beyond_1std'.format(lpb)]
        # r1std = passband_df['band-{}_beyond_1std'.format(rpb)]
        mean_diff = -2.5 * np.log10(lMean / rMean)
        std_diff = lstd - rstd
        amp_diff = lamp - ramp
        # mad_diff = lmad-rmad
        # beyond_diff = l1std-r1std
        mean_diff_colname = '{}_minus_{}_wmean'.format(lpb, rpb)
        std_diff_colname = '{}_minus_{}_std'.format(lpb, rpb)
        amp_diff_colname = '{}_minus_{}_amp'.format(lpb, rpb)
        # mad_diff_colname = '{}_minus_{}_mad'.format(lpb, rpb)
        # beyond_diff_colname = '{}_minus_{}_beyond'.format(lpb, rpb)
        passband_df[mean_diff_colname] = mean_diff
        passband_df[std_diff_colname] = std_diff
        passband_df[amp_diff_colname] = amp_diff # これがないと 0.0001 くらい下がる
        # passband_df[mad_diff_colname] = mad_diff
        # passband_df[beyond_diff_colname] = beyond_diff
        # passband_df[(lMean <= 0) | (rMean <= 0)][mean_diff_colname] = -999

    fe_set_df = fe_set_df.merge(
        passband_df.drop([
            'flux_count',
            'flux_mean',
        ],
            axis=1),
        on='object_id',
        how='left')
    del _passband_set_df, starter_fe_series, starter_fe_df, \
        band_fe_set_df, passband_df
    gc.collect()

    # feature engineering after aggregations
    fe_set_df['flux_diff'] = fe_set_df['flux_max'] - fe_set_df['flux_min']
    fe_set_df['flux_dif2'] = (fe_set_df['flux_max'] - fe_set_df['flux_min'])\
        / fe_set_df['flux_mean']
    fe_set_df['flux_w_mean'] = fe_set_df['flux_by_flux_ratio_sq_sum'] / \
        fe_set_df['flux_ratio_sq_sum']
    fe_set_df['flux_dif3'] = (fe_set_df['flux_max'] - fe_set_df['flux_min'])\
        / fe_set_df['flux_w_mean']
####    fe_set_df['corrected_flux_diff'] = fe_set_df['corrected_flux_max'] - fe_set_df['corrected_flux_min']
####    fe_set_df['corrected_flux_dif2'] = (fe_set_df['corrected_flux_max'] - fe_set_df['corrected_flux_min'])\
####        / fe_set_df['corrected_flux_mean']
####    fe_set_df['corrected_flux_w_mean'] = fe_set_df['corrected_flux_by_flux_ratio_sq_sum'] / \
####        fe_set_df['corrected_flux_ratio_sq_sum']
####    fe_set_df['corrected_flux_dif3'] = (fe_set_df['corrected_flux_max'] - fe_set_df['corrected_flux_min'])\
####        / fe_set_df['corrected_flux_w_mean']

    passband_flux_maxes = \
        ['band-{}_flux_max'.format(i) for i in passbands]
    # fe_set_df['passband_flux_maxes_var'] = \
    #     fe_set_df[passband_flux_maxes].var(axis=1)
    for passband_flux_max in passband_flux_maxes:
        fe_set_df[passband_flux_max + '_ratio_to_the_max'] = \
            fe_set_df[passband_flux_max] / fe_set_df['flux_max']
    passband_flux_mins = \
        ['band-{}_flux_min'.format(i) for i in passbands]
    fe_set_df['passband_flux_min_var'] = \
        fe_set_df[passband_flux_mins].var(axis=1)
    # for passband_flux_min in passband_flux_mins:
    #     fe_set_df[passband_flux_min + '_ratio_to_the_min'] = \
    #          fe_set_df[passband_flux_min] / fe_set_df['flux_min']
    passband_flux_means = \
        ['band-{}_flux_mean'.format(i) for i in passbands]
    fe_set_df['passband_flux_means_var'] = \
        fe_set_df[passband_flux_means].var(axis=1)
    passband_flux_counts = \
        ['band-{}_flux_count'.format(i) for i in passbands]
    fe_set_df['passband_flux_counts_var'] = \
        fe_set_df[passband_flux_counts].var(axis=1)
    passband_detected_means = \
        ['band-{}_detected_mean'.format(i) for i in passbands]
    fe_set_df['passband_detected_means_var'] = \
        fe_set_df[passband_detected_means].var(axis=1)
    # passband_flux_ratio_sq_sum = \
    #    ['band-{}_flux_ratio_sq_sum'.format(i) for i in passbands]
    # fe_set_df['passband_flux_ratio_sq_sum_var'] = \
    #    fe_set_df[passband_flux_ratio_sq_sum].var(axis=1)
    # passband_flux_ratio_sq_skew = \
    #    ['band-{}_flux_ratio_sq_skew'.format(i) for i in passbands]
    # fe_set_df['passband_flux_ratio_sq_skew_var'] = \
    #    fe_set_df[passband_flux_ratio_sq_skew].var(axis=1)
    # band の欠損率の var とかも良さそう

    # 最後にいらない features を drop するところ
    drop_cols = [
        'flux_ratio_sq_sum',
    ]
    drop_cols += passband_flux_counts
    drop_cols += passband_flux_maxes
    drop_cols += passband_flux_mins
    drop_cols += passband_flux_means
#    drop_cols += passband_flux_ratio_sq_sum
    fe_set_df.drop(drop_cols, axis=1, inplace=True)

    return fe_set_df


def feature_engineering(set_df, set_metadata_df, nthread,
                        logger, test_flg=False):
    logger.info('getting split dfs ...')
    if test_flg:
        set_dfs = load_test_set_dfs(nthread, logger)
        #set_dfs = split_dfs(set_df, nthread, logger, save_flg=True)
    else:
        set_dfs = split_dfs(set_df, nthread, logger)

####    logger.info('adding corrected flux...')
####    for i, _set_df in tqdm(enumerate(set_dfs)):
####        set_dfs[i] = add_corrected_flux(_set_df, set_metadata_df)
####        del _set_df
    gc.collect()

    logger.info('start fature engineering ...')
    logger.info('feature engineering ...')
    p = Pool(nthread)
    set_res_list = p.map(_for_set_df, set_dfs)
    p.close()
    p.join()
    set_res_df = pd.concat(set_res_list, axis=0)
    set_res_df.reset_index(inplace=True)
    gc.collect()

    # logger.info('cesium features ...')
    # p = Pool(nthread)
    # phase_res_list = p.map(get_phase_features, set_dfs)
    # p.close()
    # p.join()
    # phase_df = pd.concat(phase_res_list, axis=0).reset_index(drop=True)
    # gc.collect()

###    if test_flg:
###        _phase_df = pd.read_csv('/home/naoya.taguchi/src/train_set_full_features.csv')
###        phase_df = pd.read_csv('/home/naoya.taguchi/src/single_output_test_ts_features.csv')
###        phase_df.columns = _phase_df.columns[:-5]
###    else:
###        phase_df = pd.read_csv('/home/naoya.taguchi/src/train_set_full_features.csv')
###    phase_df = phase_df[[
###        'object_id',
###        '__max_slope___0_',
###        '__max_slope___1_',
###        '__max_slope___2_',
###        '__max_slope___3_',
###        '__max_slope___4_',
###        '__max_slope___5_',
###        '__median_absolute_deviation___0_',
###        '__median_absolute_deviation___1_',
###        '__median_absolute_deviation___2_',
###        '__median_absolute_deviation___3_',
###        '__median_absolute_deviation___4_',
###        '__median_absolute_deviation___5_',
###        '__freq_varrat___0_',
###        '__freq_varrat___1_',
###        '__freq_varrat___2_',
###        '__freq_varrat___3_',
###        '__freq_varrat___4_',
###        '__freq_varrat___5_',
###        ]]

    # phase_dfs = []
    # for df in tqdm(set_dfs):
    #    phase_dfs.append(get_phase_features(df))
    # phase_df = pd.concat(phase_dfs, axis=0).reset_index(drop=True)
    # phase_df.set_index('object_id', inplace=True)
    # phase_df.to_csv('./temp.csv', index=False)
    # phase_df = pd.read_csv('./temp.csv').reset_index(drop=True)
    # print(phase_df)
    # print(set_res_df)
    # fe_set_df = fe_set_df.merge(phase_df, on='object_id')
    # set_res_df = pd.concat([set_res_df, phase_df], axis=1)

#    logger.info('adding fft features ...')
#    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],
#           'kurtosis' : None,
#           'skewness' : None}
#    agg_df_ts = extract_features(
#            set_df, 
#            column_id='object_id', 
#            column_sort='mjd', 
#            column_kind='passband', 
#            column_value = 'flux', 
#            default_fc_parameters = fcp, 
#            n_jobs=nthread)
#    agg_df_ts.index.rename('object_id',inplace=True)

###    set_res_df = set_res_df.merge(phase_df, on='object_id', how='left')
    # set_res_df = set_res_df.merge(agg_df_ts, on='object_id', how='left')
    # del set_df, phase_df
    del set_df
    gc.collect()

    logger.info('post processing ...')
    res_df = set_metadata_df.merge(set_res_df, on='object_id', how='left')
#    res_df = res_df.merge(phase_df, on='object_id')
    res_df['internal'] = res_df.hostgal_photoz == 0.
    # res_df['astrodist'] = res_df.hostgal_photoz.apply(_get_astro_distance)
    # res_df['hostgal_photoz_square'] = np.power(res_df.hostgal_photoz, 2)
    # res_df.drop(['object_id', 'hostgal_specz', 'ra', 'decl',
    res_df.drop(['object_id', 'hostgal_specz', 'hostgal_photoz', 'ra', 'decl',
                 'gal_l', 'gal_b', 'ddf', 'mwebv'], axis=1, inplace=True)

    passbands = [0, 1, 2, 3, 4, 5]
    band_x_beyound_1stds = ['band-{}_beyond_1std'.format(i) for i in passbands]
    # band_x_flux_by_flux_ratio_sq_sums = ['band-{}_flux_by_flux_ratio_sq_sum'.format(i) for i in passbands]
    # band_x_flux_max_ratio_to_the_max = ['band-{}_flux_max_ratio_to_the_max'.format(i) for i in passbands]
    # band_x_flux_diff = ['band-{}_flux_diff'.format(i) for i in passbands]
    band_x_normed_mad = ['band-{}_normed_mad'.format(i) for i in passbands]
    # band_x_normed_std = ['band-{}_normed_std'.format(i) for i in passbands]
    # band_x_flux_var = ['band-{}_flux_var'.format(i) for i in passbands]
    # band_x_wmean = ['band-{}_wmean'.format(i) for i in passbands]
    # band_x_flux_by_flux_ratio_sq_skew = ['band-{}_flux_by_flux_ratio_sq_skew'.format(i) for i in passbands]
    # band_x_flux_skew = ['band-{}_flux_skew'.format(i) for i in passbands]
    # band_x_flux_ratio_sq_sum = ['band-{}_flux_ratio_sq_sum'.format(i) for i in passbands]
    others = []
    #res_df.drop(band_x_beyound_1stds + band_x_normed_mad + others, axis=1, inplace=True)

    del set_res_df
    gc.collect()
    return res_df
