import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import lightgbm

from logging import getLogger
import gc
from tqdm import tqdm
import argparse
import datetime
import pickle
import warnings
from matplotlib import pyplot as plt
import seaborn as sns

from tools.my_logging import logInit
from tools.feature_tools import feature_engineering
from tools.objective_function import weighted_multi_logloss, lgb_multi_weighted_logloss, wloss_objective, wloss_metric, softmax, calc_team_score, wloss_metric_for_zeropad
from tools.model_io import save_models, load_models
from tools.fold_resampling import get_fold_resampling_dict


np.random.seed(71)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)
plt.switch_backend('agg')


BASE_DIR = '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/'
#BASE_DIR = '/Users/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/'
FOLD_NUM = 5
SAMPLING_LOWER = 60
# SAMPLING_LOWER = 10
SAMPLING_LOWER_RATE = 2.


ORDERED_FEATURES = pd.read_csv('./importances/Booster_weight-multi-logloss-0.551369_2018-12-09-13-05-52_importance.csv').feature
SPLIT_SIZE = 100
N_SPLITS = 3
COMMON_TOPS = 10



def parse_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        usage='ex) python train.py --with_test',
        description='easy explanation',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-w', '--with_test',
                        help='flg to specify test type.',
                        action='store_true',
                        default=False)
    parser.add_argument('-n', '--nthread',
                        help='number of avalable threads.',
                        type=int,
                        required=True)
    parser.add_argument('-z', '--specz',
                        help='flg to use specz',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    return args


def get_params(args):
    PARAMS = {
        #        'objective': wloss_objective,
        'objective': 'multiclass',
#        'metric': ['multi_logloss', ],
        'metric': 'None',
        'num_class': 14,
        'nthread': args.nthread,
        'learning_rate': 0.4,
        #        'learning_rate': 0.02,
        #        'num_leaves': 32,
        'max_depth': 3,
        'subsample': .9,
        'colsample_bytree': .7,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.01,
        'min_child_weight': 200,
#        'n_estimators': 10000,
        'verbose': -1,
        'silent': -1,
        'random_state': 71,
        'seed': 71,
#        'early_stopping_rounds': 100,
        #        'min_data_in_leaf': 30,
        'max_bin': 20,
#        'min_data_in_leaf': 300,
#        'bagging_fraction': 0.1, 
#        'bagging_freq': 10, 
    }
    return PARAMS


# Display/plot feature importance
def display_importances(feature_importance_df_,
                        filename='importance_application'):
    # cols = feature_importance_df_[["feature",
    # "importance"]].groupby("feature").mean().sort_values(by="importance",
    # ascending=False).index
    csv_df = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").agg({'importance': ['mean', 'std']})
    csv_df.columns = pd.Index(
        [e[0] + "_" + e[1].upper()
            for e in csv_df.columns.tolist()])
    csv_df['importance_RAT'] = csv_df['importance_STD'] / \
        csv_df['importance_MEAN']
    csv_df.sort_values(
        by="importance_MEAN",
        ascending=False).to_csv(
        filename +
        '.csv')
#    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
#    plt.figure(figsize=(8, 10))
#    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
#    plt.title('LightGBM Features (avg over folds)')
#    plt.tight_layout()
#    plt.savefig(filename + '.png')


def save_importance(df, filename):
    df.set_index('feature', inplace=True)
    imp_mean = df.mean(axis=1)
    imp_std = df.std(axis=1)
    df['importance_mean'] = imp_mean
    df['importance_std'] = imp_std
    df['importance_cov'] = df['importance_std'] / df['importance_mean']
    df.sort_values(by="importance_cov", ascending=True).to_csv(filename[:-4] + '.csv')
    df.reset_index(inplace=True)
    plt.figure(figsize=(8, 30))
    sns.barplot(x="importance_mean", y="feature", data=df.sort_values(by="importance_mean", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(filename)


def plt_confusion_matrics():
    1 + 1


def split_features_based_on_a_metric(features, split_size=120, n_splits=3, common_tops=10):
    assert (split_size-common_tops)*n_splits+common_tops <= len(features), 'split_size * n_splits should less than features'
    splitted_feats = [[col for col in features[:common_tops]] +
            [features[common_tops+i+j*n_splits] for j in range(split_size-common_tops)]
            for i in range(n_splits)]
    return splitted_feats
#    feature_idx_list = [
#            [features.get_loc(feat) for feat in splitted_feat]
#            for splitted_feat in splitted_feats]
#    return feature_idx_list


def main(args, features):
    FEATURES_TO_USE = features
    logger = getLogger(__name__)
    logInit(logger, log_dir='./log/', log_filename='train.log')
    logger.info(
        '''
        start main, the args settings are ...
        --with_test : {}
        '''.format(args.with_test))

    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger.info('start training, the starting time is {}'.format(start_time))

    PARAMS = get_params(args)

    logger.info('loading train_df ...')
    train_df = pd.read_feather('./features/train/meta_features.ftr')

    with open('./lcfit/LCfit_features_train_20181129.pkl', 'rb') as fin:
        train_df = train_df.merge(pickle.load(fin), on='object_id', how='left')
    train_df.drop('object_id', axis=1, inplace=True)
    train_df = train_df[FEATURES_TO_USE + ['target']]

    logger.debug('the cols of train_df : {}'.
                format(train_df.drop('target', axis=1).columns.tolist()))

    for model_id, splitted_feat in enumerate(split_features_based_on_a_metric(
            features=ORDERED_FEATURES,
            split_size=SPLIT_SIZE, 
            n_splits=N_SPLITS, 
            common_tops=COMMON_TOPS,
            )):
        print(f'------- {model_id} -------')
        if model_id == 0:
            continue
        # label encoding しないと lgbm が認識してくれない
        # 若い class に 若い label がつくと良いんだけど...
        le = LabelEncoder()
        le.fit(train_df['target'].values)
        x_train = train_df.drop('target', axis=1).values
        y_train = le.transform(train_df.target)

        skf = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=71)
        folds = skf.split(x_train, y_train)

        best_scores = []
        team_scores = []
        zeropad_scores = []
        val_pred_score_zeropads = []
        trained_models = []
        best_iterations = []
        oof = []
        x_train = train_df.drop('target', axis=1)[splitted_feat].values
        y_train = le.transform(train_df['target'].values)
        logger.info('the shape of x_train : {}'.format(x_train.shape))
        train_columns = train_df.drop('target', axis=1)[splitted_feat].columns
        feature_importance_df = pd.DataFrame()
        feature_importance_df['feature'] = train_columns
        conf_y_true = []
        conf_y_pred = []
        i = 1

        for trn_idx, val_idx in tqdm(list(folds)):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

            fold_resampling_dict = \
                get_fold_resampling_dict(
                    y_trn,
                    logger,
                    SAMPLING_LOWER,
                    SAMPLING_LOWER_RATE)
            ros = RandomOverSampler(
                ratio=fold_resampling_dict,
                random_state=71)
#            x_trn, y_trn = ros.fit_sample(x_trn, y_trn)

            train_dataset = lightgbm.Dataset(x_trn, y_trn)
            valid_dataset = lightgbm.Dataset(x_val, y_val)
            booster = lightgbm.train(
                PARAMS.copy(), train_dataset,
                num_boost_round=2000,
                fobj=wloss_objective,
                feval=wloss_metric,
                valid_sets=[train_dataset, valid_dataset],
                verbose_eval=100,
                early_stopping_rounds=100,
            )
            logger.debug('valid info : {}'.format(booster.best_score))
            logger.info('best score : {}'.format(booster.best_score['valid_1']['wloss']))
            logger.info('best iteration : {}'.format(booster.best_iteration))
            best_scores.append(booster.best_score['valid_1']['wloss'])
            best_iterations.append(booster.best_iteration)
            trained_models.append(booster)
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = train_columns
            fold_importance_df["importance_{}".format(i)] = booster.feature_importance('gain')
            feature_importance_df = feature_importance_df.merge(fold_importance_df, on='feature', how='left')
            #feature_importance_df = pd.concat(
            #    [feature_importance_df, fold_importance_df], axis=0)
            val_pred_score = softmax(booster.predict(x_val, raw_score=False))
            val_pred_score_zeropad = booster.predict(x_val, raw_score=False)
            oof.append([val_pred_score_zeropad, y_val, val_idx])
            conf_y_true.append(y_val)
            conf_y_pred.append(np.argmax(val_pred_score, axis=1))
            i += 1

        mean_best_score = np.mean(best_scores)
        mean_team_score = np.mean(team_scores)
        mean_best_iteration = np.mean(best_iterations)
        mean_zeropads_score = np.mean(np.array(zeropad_scores, dtype=float))
        logger.info('mean valid score is {}'.format(mean_best_score))
        logger.info('mean team score is {}'.format(mean_team_score))
        logger.info('mean best iteration is {}'.format(mean_best_iteration))
        #logger.info('mean zeropad score is {}'.format(mean_zeropads_score))
        oof_path = './oof/{}_weight-multi-logloss-{:.6}_{}_model-{}.pkl'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, model_id)
        with open(oof_path, 'wb') as fout:
            pickle.dump(oof, fout)

        models_path = './trained_models/{}_weight-multi-logloss-{:.6}_{}_model-{}.pkl'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, model_id)
        logger.info('saving models to {} ...'.format(models_path))
        save_models(trained_models, models_path)

        imp_path = './importances/{}_weight-multi-logloss-{:.6}_{}_importance_model-{}.png'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, model_id)
        logger.info('saving importance to {} ...'.format(models_path))
        save_importance(feature_importance_df, imp_path)

        conf_path = './confusion_matrices/{}_weight-multi-logloss-{:.6}_{}_confusion_model-{}.png'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, model_id)
        logger.info('saving confusion matrix to {} ...'.format(models_path))
        conf_y_pred = np.concatenate(conf_y_pred)
        conf_y_true = np.concatenate(conf_y_true)
        cm = confusion_matrix(conf_y_true, conf_y_pred)
        classes = ['class_' + str(clnum)
                for clnum in [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]]
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df[cm_df.columns] = cm_df.values / cm_df.sum(axis=1).values.reshape(-1, 1)
        plt.figure(figsize=(14, 14))
        sns.heatmap(cm_df, annot=True, cmap=plt.cm.Blues)
        #plt.imshow(cm_df.values, interpolat='nearest', cmap=plt.cm.Blues)
        plt.title('score : {}'.format(mean_best_score))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(conf_path)

        if args.with_test == False:
            continue

        if args.with_test:
            logger.info('start linear interpolation training')
            interpolated_num_boost_round =\
                    int(mean_best_iteration * FOLD_NUM / (FOLD_NUM - 1))
            logger.info('the num boost round is {}'.format(interpolated_num_boost_round))

            fold_resampling_dict = \
                get_fold_resampling_dict(
                    y_train,
                    logger,
                    SAMPLING_LOWER,
                    SAMPLING_LOWER_RATE)
            ros = RandomOverSampler(
                ratio=fold_resampling_dict,
                random_state=71)
#            x_train, y_train = ros.fit_sample(x_train, y_train)

            train_dataset = lightgbm.Dataset(x_train, y_train)
            lin_booster = lightgbm.train(
                PARAMS.copy(), train_dataset,
                num_boost_round=interpolated_num_boost_round,
                fobj=wloss_objective,
                feval=wloss_metric,
                valid_sets=[train_dataset, ],
                verbose_eval=100,
                early_stopping_rounds=100
)
            logger.info('best score : {}'.format(lin_booster.best_score))
            models_path = './trained_models/{}_weight-multi-logloss-{:.6}_{}_linear_interpolated.pkl'\
                .format(lin_booster.__class__.__name__,
                        mean_best_score,
                        start_time, )
            logger.info('saving models to {} ...'.format(models_path))
            save_models(lin_booster, models_path)

            logger.info('loading test_df ...')
            test_df = pd.read_feather('./features/test/meta_features.ftr', nthreads=args.nthread)

            with open('./lcfit/LCfit_feature_test_v1_20181203.pkl', 'rb') as fin:
#                test_df = test_df[list(set(test_df.columns.tolist()) & set(FEATURES_TO_USE)) + ['object_id']].\
                test_df = test_df[list(set(test_df.columns.tolist()) & set(splitted_feat)) + ['object_id']].\
                        merge(pickle.load(fin), on='object_id', how='left')

            object_ids = test_df.object_id

            test_df.drop('object_id', axis=1, inplace=True)
            test_df = test_df[splitted_feat]
            #test_df = test_df[FEATURES_TO_USE]
            gc.collect()

            test_df.reset_index(drop=True).to_feather('./test_dfs/test_df_for_nn.fth')

            logger.info(f'test cols {test_df.columns.tolist()}')
            x_test = test_df.values
            logger.info(f'test size: {x_test.shape}')

            logger.info('predicting')
            test_reses = []
            for lgb in tqdm(trained_models):
                test_reses.append(
                    softmax(lgb.predict(x_test, raw_score=False)))
                # test_reses.append(lgb.predict_proba(x_test, raw_score=False))

            # prediction of linear interpolated
            lin_test_res = \
                    softmax(lin_booster.predict(x_test, raw_score=False))

            res = np.clip(np.mean(
                          [np.mean(test_reses, axis=0),
                           lin_test_res],
                          axis=0),
                  10**(-15), 1 - 10**(-15))
            preds_99 = np.ones((res.shape[0]))
            for i in range(res.shape[1]):
                preds_99 *= (1 - res[:, i])
            preds_99 = 0.14 * preds_99 / np.mean(preds_99)
            #res *= 8/9
            #preds_99 = 1/9

            # res = np.concatenate((res, preds_99), axis=1)
            # res = np.concatenate((res, np.zeros((res.shape[0], 1))), axis=1)
            logger.info('now creating the submission file ...')
            res_df = pd.DataFrame(res, columns=[
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
                #                'class_99',
            ])
            res_df['class_99'] = preds_99
            submission_file_name = './submissions/{}_weight-multi-logloss-{:.6}_{}_model-{}.csv'\
                .format(trained_models[0].__class__.__name__,
                        mean_best_score,
                        start_time, model_id)
            logger.info(
                'saving the test result to {}'.format(submission_file_name))
            pd.concat([object_ids, res_df], axis=1)\
                .to_csv(submission_file_name, index=False)

    logger.info('finish !')


if __name__ == '__main__':
    args = parse_args()
    FEATURES_TO_USE = [
    #        'hostgal_photoz',
            'hostgal_photoz_err',
            'distmod',
            'lumi_dist',
            'flux_min',
            'flux_max',
            'flux_mean',
            'flux_median',
            'flux_std',
            'flux_var',
            'flux_skew',
            'flux_count',
            'flux_kurtosis',
            'flux_err_min',
            'flux_err_max',
            'flux_err_mean',
            'flux_err_median',
            'flux_err_std',
            'flux_err_var',
            'flux_err_skew',
            'flux_err_kurtosis',
            'flux_ratio_to_flux_err_min',
            'flux_ratio_to_flux_err_max',
            'detected_mean',
            'flux_ratio_sq_skew',
            'flux_ratio_sq_mean',
            'flux_ratio_sq_kurtosis',
            'flux_by_flux_ratio_sq_sum',
            'flux_by_flux_ratio_sq_skew',
            'std_upper_mjd_get_max_min_diff',
            'std_upper_mjd_var',
            'std_upper_mjd_skew',
            'std_upper_flux_count',
            'std_upper_flux_min',
            'detected_mjd_get_max_min_diff',
            'detected_mjd_skew',
            'band-0_wmean',
            'band-0_normed_std',
            'band-0_normed_amp',
            'band-0_normed_mad',
            'band-0_beyond_1std',
            'band-0_flux_var',
            'band-0_flux_skew',
            'band-0_flux_kurtosis',
            'band-0_flux_quantile10',
            'band-0_flux_quantile25',
            'band-0_flux_quantile75',
            'band-0_flux_quantile90',
            'band-0_flux_quantile2575_range',
            'band-0_flux_quantile1090_range',
            'band-0_normed_flux_diff_mean',
            'band-0_detected_mean',
            'band-0_flux_ratio_sq_sum',
            'band-0_flux_ratio_sq_skew',
            'band-0_flux_by_flux_ratio_sq_sum',
            'band-0_flux_by_flux_ratio_sq_skew',
            'band-0_flux_get_max_min_diff',
            'band-0_std_upper_mjd_get_max_min_diff',
            'band-0_std_upper_mjd_var',
            'band-0_std_upper_mjd_skew',
            'band-0_std_upper_mjd_diff_mean',
            'band-0_std_upper_flux_count',
    #        'band-0_std_upper_flux_count_ratio',
            'band-0_std_upper_flux_diff_mean',
            'band-1_wmean',
            'band-1_normed_std',
            'band-1_normed_amp',
            'band-1_normed_mad',
            'band-1_beyond_1std',
            'band-1_flux_var',
            'band-1_flux_skew',
            'band-1_flux_kurtosis',
            'band-1_flux_quantile10',
            'band-1_flux_quantile25',
            'band-1_flux_quantile75',
            'band-1_flux_quantile90',
            'band-1_flux_quantile2575_range',
            'band-1_flux_quantile1090_range',
            'band-1_normed_flux_diff_mean',
            'band-1_detected_mean',
            'band-1_flux_ratio_sq_sum',
            'band-1_flux_ratio_sq_skew',
            'band-1_flux_by_flux_ratio_sq_sum',
            'band-1_flux_by_flux_ratio_sq_skew',
            'band-1_flux_get_max_min_diff',
            'band-1_std_upper_mjd_get_max_min_diff',
            'band-1_std_upper_mjd_var',
            'band-1_std_upper_mjd_skew',
            'band-1_std_upper_mjd_diff_mean',
            'band-1_std_upper_flux_count',
    #        'band-1_std_upper_flux_count_ratio',
            'band-1_std_upper_flux_diff_mean',
            'band-2_wmean',
            'band-2_normed_std',
            'band-2_normed_amp',
            'band-2_normed_mad',
            'band-2_beyond_1std',
            'band-2_flux_var',
            'band-2_flux_skew',
            'band-2_flux_kurtosis',
            'band-2_flux_quantile10',
            'band-2_flux_quantile25',
            'band-2_flux_quantile75',
            'band-2_flux_quantile90',
            'band-2_flux_quantile2575_range',
            'band-2_flux_quantile1090_range',
            'band-2_normed_flux_diff_mean',
            'band-2_detected_mean',
            'band-2_flux_ratio_sq_sum',
            'band-2_flux_ratio_sq_skew',
            'band-2_flux_by_flux_ratio_sq_sum',
            'band-2_flux_by_flux_ratio_sq_skew',
            'band-2_flux_get_max_min_diff',
            'band-2_std_upper_mjd_get_max_min_diff',
            'band-2_std_upper_mjd_var',
            'band-2_std_upper_mjd_skew',
            'band-2_std_upper_mjd_diff_mean',
            'band-2_std_upper_flux_count',
    #        'band-2_std_upper_flux_count_ratio',
            'band-2_std_upper_flux_diff_mean',
            'band-3_wmean',
            'band-3_normed_std',
            'band-3_normed_amp',
            'band-3_normed_mad',
            'band-3_beyond_1std',
            'band-3_flux_var',
            'band-3_flux_skew',
            'band-3_flux_kurtosis',
            'band-3_flux_quantile10',
            'band-3_flux_quantile25',
            'band-3_flux_quantile75',
            'band-3_flux_quantile90',
            'band-3_flux_quantile2575_range',
            'band-3_flux_quantile1090_range',
            'band-3_normed_flux_diff_mean',
            'band-3_detected_mean',
            'band-3_flux_ratio_sq_sum',
            'band-3_flux_ratio_sq_skew',
            'band-3_flux_by_flux_ratio_sq_sum',
            'band-3_flux_by_flux_ratio_sq_skew',
            'band-3_flux_get_max_min_diff',
            'band-3_std_upper_mjd_get_max_min_diff',
            'band-3_std_upper_mjd_var',
            'band-3_std_upper_mjd_skew',
            'band-3_std_upper_mjd_diff_mean',
            'band-3_std_upper_flux_count',
    #        'band-3_std_upper_flux_count_ratio',
            'band-3_std_upper_flux_diff_mean',
            'band-4_wmean',
            'band-4_normed_std',
            'band-4_normed_amp',
            'band-4_normed_mad',
            'band-4_beyond_1std',
            'band-4_flux_var',
            'band-4_flux_skew',
            'band-4_flux_kurtosis',
            'band-4_flux_quantile10',
            'band-4_flux_quantile25',
            'band-4_flux_quantile75',
            'band-4_flux_quantile90',
            'band-4_flux_quantile2575_range',
            'band-4_flux_quantile1090_range',
            'band-4_normed_flux_diff_mean',
            'band-4_detected_mean',
            'band-4_flux_ratio_sq_sum',
            'band-4_flux_ratio_sq_skew',
            'band-4_flux_by_flux_ratio_sq_sum',
            'band-4_flux_by_flux_ratio_sq_skew',
            'band-4_flux_get_max_min_diff',
            'band-4_std_upper_mjd_get_max_min_diff',
            'band-4_std_upper_mjd_var',
            'band-4_std_upper_mjd_skew',
            'band-4_std_upper_mjd_diff_mean',
            'band-4_std_upper_flux_count',
    #        'band-4_std_upper_flux_count_ratio',
            'band-4_std_upper_flux_diff_mean',
            'band-5_wmean',
            'band-5_normed_std',
            'band-5_normed_amp',
            'band-5_normed_mad',
            'band-5_beyond_1std',
            'band-5_flux_var',
            'band-5_flux_skew',
            'band-5_flux_kurtosis',
            'band-5_flux_quantile10',
            'band-5_flux_quantile25',
            'band-5_flux_quantile75',
            'band-5_flux_quantile90',
            'band-5_flux_quantile2575_range',
            'band-5_flux_quantile1090_range',
            'band-5_normed_flux_diff_mean',
            'band-5_detected_mean',
            'band-5_flux_ratio_sq_sum',
            'band-5_flux_ratio_sq_skew',
            'band-5_flux_by_flux_ratio_sq_sum',
            'band-5_flux_by_flux_ratio_sq_skew',
            'band-5_flux_get_max_min_diff',
            'band-5_std_upper_mjd_get_max_min_diff',
            'band-5_std_upper_mjd_var',
            'band-5_std_upper_mjd_skew',
            'band-5_std_upper_mjd_diff_mean',
            'band-5_std_upper_flux_count',
    #        'band-5_std_upper_flux_count_ratio',
            'band-5_std_upper_flux_diff_mean',
            '0_minus_1_wmean',
            '0_minus_1_std',
            '0_minus_1_amp',
            '1_minus_2_wmean',
            '1_minus_2_std',
            '1_minus_2_amp',
            '2_minus_3_wmean',
            '2_minus_3_std',
            '2_minus_3_amp',
            '3_minus_4_wmean',
            '3_minus_4_std',
            '3_minus_4_amp',
            '4_minus_5_wmean',
            '4_minus_5_std',
            '4_minus_5_amp',
            '5_minus_0_wmean',
            '5_minus_0_std',
            '5_minus_0_amp',
            'flux_diff',
            'flux_dif2',
            'flux_w_mean',
            'flux_dif3',
            'std_upper_rat',
            'band-0_flux_max_ratio_to_the_max',
            'band-1_flux_max_ratio_to_the_max',
            'band-2_flux_max_ratio_to_the_max',
            'band-3_flux_max_ratio_to_the_max',
            'band-4_flux_max_ratio_to_the_max',
            'band-5_flux_max_ratio_to_the_max',
            'passband_flux_min_var',
            'passband_flux_means_var',
            'passband_flux_counts_var',
            'passband_detected_means_var',
            'band_flux_diff_max',
            'band_flux_diff_min',
            'band_flux_diff_diff',
            'band_flux_diff_diff_rat',
            'band_flux_max_min_rat',
            '0__length',
            '0__longest_strike_above_mean',
            '0__longest_strike_below_mean',
            '0__mean_abs_change',
            '0__mean_change',
            '1__length',
            '1__longest_strike_above_mean',
            '1__longest_strike_below_mean',
            '1__mean_abs_change',
            '1__mean_change',
            '2__length',
            '2__longest_strike_above_mean',
            '2__longest_strike_below_mean',
            '2__mean_abs_change',
            '2__mean_change',
            '3__length',
            '3__longest_strike_above_mean',
            '3__longest_strike_below_mean',
            '3__mean_abs_change',
            '3__mean_change',
            '4__length',
            '4__longest_strike_above_mean',
            '4__longest_strike_below_mean',
            '4__mean_abs_change',
            '4__mean_change',
            '5__length',
            '5__longest_strike_above_mean',
            '5__longest_strike_below_mean',
            '5__mean_abs_change',
            '5__mean_change',
            'internal',
            'c90_z_z1',
            'c90_y_z1',
            'c52_y_z1',
            'c67_g_z2',
            'c67_i_z2',
            'c67_y_z2',
            'c52_r_z3',
            'c42_i_z4',
            'c42_z_z4',
    ###        'band-0_detected_mjd_get_max_min_diff', # これ系は cv ↑ lb ↓
    #        'band-0_detected_mjd_var',
    #        'band-0_detected_mjd_skew',
    #        'band-0_detected_mjd_diff_mean',
    ###        'band-1_detected_mjd_get_max_min_diff',
    #        'band-1_detected_mjd_var',
    #        'band-1_detected_mjd_skew',
    #        'band-1_detected_mjd_diff_mean',
    ###        'band-2_detected_mjd_get_max_min_diff',
    #        'band-2_detected_mjd_var',
    #        'band-2_detected_mjd_skew',
    #        'band-2_detected_mjd_diff_mean',
    ###        'band-3_detected_mjd_get_max_min_diff',
    #        'band-3_detected_mjd_var',
    #        'band-3_detected_mjd_skew',
    #        'band-3_detected_mjd_diff_mean',
    ###        'band-4_detected_mjd_get_max_min_diff',
    #        'band-4_detected_mjd_var',
    #        'band-4_detected_mjd_skew',
    #        'band-4_detected_mjd_diff_mean',
    ###        'band-5_detected_mjd_get_max_min_diff',
    #        'band-5_detected_mjd_var',
    #        'band-5_detected_mjd_skew',
    #        'band-5_detected_mjd_diff_mean'
    #        '0_minus_1_dmgmmd',
    #        '1_minus_2_dmgmmd',
    #        '2_minus_3_dmgmmd',
    #        '3_minus_4_dmgmmd',
    #        '4_minus_5_dmgmmd',
    #        '5_minus_0_dmgmmd',
    #        'std_lower_mjd_get_max_min_diff',
    #        'std_lower_mjd_var',
    #        'std_lower_mjd_skew',
    #        'std_lower_flux_count',
    #        'std_lower_flux_max'
    #        '0_minus_1_skew',
    #        '1_minus_2_skew',
    #        '2_minus_3_skew',
    #        '3_minus_4_skew',
    #        '4_minus_5_skew',
    #        '5_minus_0_skew',
    #        '0_minus_1_kurt',
    #        '1_minus_2_kurt',
    #        '2_minus_3_kurt',
    #        '3_minus_4_kurt',
    #        '4_minus_5_kurt',
    #        '5_minus_0_kurt',
            '0_minus_1_q2575_rng',
            '1_minus_2_q2575_rng',
            '2_minus_3_q2575_rng',
            '3_minus_4_q2575_rng',
            '4_minus_5_q2575_rng',
            '5_minus_0_q2575_rng',
            'band-0_std_upper_flux_quantile10',
            'band-1_std_upper_flux_quantile10',
            'band-2_std_upper_flux_quantile10',
            'band-3_std_upper_flux_quantile10',
            'band-4_std_upper_flux_quantile10',
            'band-5_std_upper_flux_quantile10',
    #        'band-0_std_upper_flux_quantile25',
    #        'band-1_std_upper_flux_quantile25',
    #        'band-2_std_upper_flux_quantile25',
    #        'band-3_std_upper_flux_quantile25',
    #        'band-4_std_upper_flux_quantile25',
    #        'band-5_std_upper_flux_quantile25',
    #        'band-0_std_upper_flux_quantile75',
    #        'band-1_std_upper_flux_quantile75',
    #        'band-2_std_upper_flux_quantile75',
    #        'band-3_std_upper_flux_quantile75',
    #        'band-4_std_upper_flux_quantile75',
    #        'band-5_std_upper_flux_quantile75',
            'band-0_std_upper_flux_quantile90',
            'band-1_std_upper_flux_quantile90',
            'band-2_std_upper_flux_quantile90',
            'band-3_std_upper_flux_quantile90',
            'band-4_std_upper_flux_quantile90',
            'band-5_std_upper_flux_quantile90',
    #        'band-0_std_upper_flux_quantile2575_range',
    #        'band-1_std_upper_flux_quantile2575_range',
    #        'band-2_std_upper_flux_quantile2575_range',
    #        'band-3_std_upper_flux_quantile2575_range',
    #        'band-4_std_upper_flux_quantile2575_range',
    #        'band-5_std_upper_flux_quantile2575_range',
    #        'band-0_std_upper_flux_quantile1090_range',
    #        'band-1_std_upper_flux_quantile1090_range',
    #        'band-2_std_upper_flux_quantile1090_range',
    #        'band-3_std_upper_flux_quantile1090_range',
    #        'band-4_std_upper_flux_quantile1090_range',
    #        'band-5_std_upper_flux_quantile1090_range',
    #        'band-0_flux_max',
    #        'band-1_flux_max',
    #        'band-2_flux_max',
    #        'band-3_flux_max',
    #        'band-4_flux_max',
    #        'band-5_flux_max',
            '0_minus_1_max',
            '1_minus_2_max',
            '2_minus_3_max',
            '3_minus_4_max',
            '4_minus_5_max',
            '5_minus_0_max',
    #        'abs_magnitude_min',
    #        'abs_magnitude_max',
######            'abs_magnitude_mean',
######            'abs_magnitude_median',
######            'abs_magnitude_std',
######            'abs_magnitude_var',
######            'abs_magnitude_skew',
    #        'abs_magnitude_kurtosis',
    #####        'luminosity_max', 
    #####        'peak-14-14_flux_mean',
    #####        'peak-30-30_flux_mean',
    #####        'peak-90-90_flux_mean',
    #####        'peak-14-14_flux_kurtosis',
    #####        'peak-30-30_flux_kurtosis',
    #####        'peak-90-90_flux_kurtosis',
    #####        'peak_kurt_14to30',
    #        'peak_kurt_14to90',
    #####        'peak_kurt_30to90',
    #####        'peak-14-14_flux_skew',
    #        'peak-30-30_flux_skew',
    #        'peak-90-90_flux_skew',
    #####        'peak_skew_14to30',
    #        'peak_skew_14to90',
    #        'peak_skew_30to90',
###############################            'peak-0-14_flux_diff_var',
###############################            'peak-0-30_flux_diff_var',
    #####        'peak-0-90_flux_diff_var',
    #        'peak-14-0_flux_diff_var',
    #        'peak-30-0_flux_diff_var',
    #        'peak-14-14_flux_get_max_min_diff',
    #        'peak-30-30_flux_get_max_min_diff',
    #        'peak-90-90_flux_get_max_min_diff',
    #        'peak-30-30_luminosity_kurtosis',
    #        'peak-14-14_detected_mean',
    #        'peak-0-30_abs_magnitude_diff_var',
    #        'peak-0-90_abs_magnitude_diff_var',
    #        'peak-14-14_abs_magnitude_skew',
    #####        'peak-30-30_abs_magnitude_skew',
    #####        'peak-90-90_abs_magnitude_skew',
    #        'peak-14-14_abs_magnitude_kurtosis',
    #####        'peak-30-30_abs_magnitude_kurtosis',
    #####        'peak-90-90_abs_magnitude_kurtosis',
    #        'peak-14-14_flux_ratio_sq_sum',
    #        'peak-30-30_flux_ratio_sq_sum',
    #        'peak-90-90_flux_ratio_sq_sum',
    #####        'ratsq-peak-14-14_flux_ratio_sq_skew',
    #        'peak-30-30_flux_ratio_sq_skew',
    #####        'ratsq-peak-90-90_flux_ratio_sq_skew',
    #        'peak-14-14_flux_ratio_sq_kurtosis',
    #        'peak-30-30_flux_ratio_sq_kurtosis',
    #        'peak-90-90_flux_ratio_sq_kurtosis',
    #        'peak-0-14_flux_ratio_sq_skew',
    #        'peak-14-14_flux_ratio_sq_mean',
    #        'peak-30-30_flux_ratio_sq_mean',
    #        'peak-90-90_flux_ratio_sq_mean',
    #        'peak-14-14_corrected_flux_by_flux_ratio_sq_skew',
    #        'peak-30-30_corrected_flux_by_flux_ratio_sq_skew',
    #        'band-0_flux_ratio_sq_get_max_min_diff',
    #        'band-1_flux_ratio_sq_get_max_min_diff',
    #        'band-2_flux_ratio_sq_get_max_min_diff',
    #        'band-3_flux_ratio_sq_get_max_min_diff',
    #        'band-4_flux_ratio_sq_get_max_min_diff',
    #        'band-5_flux_ratio_sq_get_max_min_diff',
            '0_minus_1_ratsqmax',
            '1_minus_2_ratsqmax',
            '2_minus_3_ratsqmax',
            '3_minus_4_ratsqmax',
            '4_minus_5_ratsqmax',
            '5_minus_0_ratsqmax',
    #        '0_minus_1_ratsqmax_log',
    #        '1_minus_2_ratsqmax_log',
    #        '2_minus_3_ratsqmax_log',
    #        '3_minus_4_ratsqmax_log',
    #        '4_minus_5_ratsqmax_log',
    #        '5_minus_0_ratsqmax_log',
    #        'band-0_flux_ratio_sq_max_ratio',
    #        'band-1_flux_ratio_sq_max_ratio',
    #        'band-2_flux_ratio_sq_max_ratio',
    #        'band-3_flux_ratio_sq_max_ratio',
    #        'band-4_flux_ratio_sq_max_ratio',
    #        'band-5_flux_ratio_sq_max_ratio',
    #        'flux_ratio_sq_max',
    #        'ddf'
            'my_skew',
            'my_kurt',
            'mjd_diff_af_det1',
#            'mjd_diff_bf_det1',
            'mjd_diff_ab_sum',
#            'band-0_my_skew',
#            'band-1_my_skew',
#            'band-2_my_skew',
#            'band-3_my_skew',
#            'band-4_my_skew',
#            'band-5_my_skew',
#            'band-0_my_kurt',
#            'band-1_my_kurt',
#            'band-2_my_kurt',
#            'band-3_my_kurt',
#            'band-4_my_kurt',
#            'band-5_my_kurt',
#            'hostgal_photoz',
#            'det_my_skew',
#            'det_my_kurt',
    ]


    if args.specz:
        FEATURES_TO_USE += [
            'hostgal_specz',
            'z_corrected_flux_diff',
            'z_corrected_flux_dif2',
            'z_corrected_flux_w_mean',
            'z_corrected_flux_dif3',
            'z_corrected_flux_min',
            'z_corrected_flux_max',
            'z_corrected_flux_mean',
            'z_corrected_flux_median',
            'z_corrected_flux_std',
            'z_corrected_flux_var',
            'z_corrected_flux_skew',
            'z_corrected_flux_ratio_sq_sum',
            'z_corrected_flux_ratio_sq_skew',
            'z_corrected_flux_by_flux_ratio_sq_sum',
            'z_corrected_flux_by_flux_ratio_sq_skew',
#            'band-0_z_corrected_flux_min',
#            'band-1_z_corrected_flux_min',
#            'band-2_z_corrected_flux_min',
#            'band-3_z_corrected_flux_min',
#            'band-4_z_corrected_flux_min',
#            'band-5_z_corrected_flux_min',
            '0_minus_1_zcorrmax_diff',
            '1_minus_2_zcorrmax_diff',
            '2_minus_3_zcorrmax_diff',
            '3_minus_4_zcorrmax_diff',
            '4_minus_5_zcorrmax_diff',
            '5_minus_0_zcorrmax_diff',
            ]
    else:
#        FEATURES_TO_USE = FEATURES_TO_USE
        FEATURES_TO_USE += [
            'corrected_flux_diff',
            'corrected_flux_dif2',
            'corrected_flux_w_mean',
            'corrected_flux_dif3',
            'corrected_flux_min',
            'corrected_flux_max',
            'corrected_flux_mean',
            'corrected_flux_median',
            'corrected_flux_std',
            'corrected_flux_var',
            'corrected_flux_skew',
            'corrected_flux_ratio_sq_sum',
            'corrected_flux_ratio_sq_skew',
            'corrected_flux_by_flux_ratio_sq_sum',
            'corrected_flux_by_flux_ratio_sq_skew',
        ]
    
    main(args, FEATURES_TO_USE)
