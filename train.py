import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import lightgbm

from logging import getLogger
from tqdm import tqdm
import argparse
import datetime
import pickle
import warnings
from matplotlib import pyplot as plt
import seaborn as sns

from tools.my_logging import logInit
from tools.feature_tools import feature_engineering
from tools.objective_function import weighted_multi_logloss, lgb_multi_weighted_logloss, wloss_objective, wloss_metric, softmax, calc_team_score
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

    args = parser.parse_args()
    return args


def get_params(args):
    PARAMS = {
        #        'objective': wloss_objective,
        'objective': 'multiclass',
#        'metric': ['multi_logloss', ],
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


def main(args):
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

    logger.info('loading training_set.csv')
    training_set_df = pd.read_csv(
        BASE_DIR + 'training_set.csv')
    logger.info('loading training_set_metadata.csv')
    training_set_metadata_df = pd.read_csv(
        BASE_DIR + 'training_set_metadata.csv')
    # training_set_metadata_df =
    # training_set_metadata_df[training_set_metadata_df.ddf == 1]

    logger.info('start feagture engineering')
    train_df = feature_engineering(
        training_set_df,
        training_set_metadata_df,
        nthread=args.nthread,
        logger=logger)

    # label encoding しないと lgbm が認識してくれない
    # 若い class に 若い label がつくと良いんだけど...
    le = LabelEncoder()
    le.fit(train_df['target'].values)
    x_train = train_df.drop('target', axis=1).values
    y_train = le.transform(train_df.target)
    train_set = lightgbm.Dataset(
        data=train_df.drop('target', axis=1).values,
        label=le.transform(train_df['target'].values),
    )

    skf = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=71)
#    folds = skf.split(
#        train_df.drop('target', axis=1), le.transform(train_df.target))
    folds = skf.split(x_train, y_train)

    logger.info('the shape of x_train : {}'.format(x_train.shape))
    # logger.info('the shape of train_df : {}'.format(train_df.shape))
    logger.debug('the cols of train_df : {}'.
                format(train_df.drop('target', axis=1).columns.tolist()))
#    categotical_features = ['passband_maxes_argmaxes', ]
#    categorical_features_idx = np.argwhere(train_df.drop('target', axis=1).columns == 'passband_maxes_argmaxes')[0]
#    logger.debug('categorical features are : {}'.format(categotical_features))
#    logger.debug('categorical features indexes are : {}'.format(categotical_features))
#    PARAMS['categorical_feature'] = categorical_features_idx


    if False:  # args.with_test:
        cv_hist = lightgbm.cv(
            params=PARAMS,
            folds=folds,
            train_set=train_set,
            nfold=FOLD_NUM,
            verbose_eval=100,
            feval=lgb_multi_weighted_logloss,
        )
        logger.info('best_scores : {}'.format(
            np.min(cv_hist['multi_logloss-mean'])))
        logger.debug(cv_hist)

    elif False:
        best_scores = []
        trained_models = []
        x_train = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values
        train_columns = train_df.drop('target', axis=1).columns
        feature_importance_df = pd.DataFrame()
        i = 1

        for trn_idx, val_idx in tqdm(list(folds)):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]
            lgb = lightgbm.LGBMClassifier(**PARAMS)
            lgb.fit(x_trn, y_trn,
                    eval_set=[(x_trn, y_trn), (x_val, y_val)],
                    verbose=100,
                    eval_metric=lgb_multi_weighted_logloss,
                    #                    eval_metric=weighted_multi_logloss,
                    #                    eval_metric='multi_logloss',
                    )
#            logger.info('best_itr : {}'.format(lgb.best_iteration_))
            logger.info('best_scores : {}'.format(lgb.best_score_))
            best_scores.append(lgb.best_score_['valid_1']['wloss'])
            trained_models.append(lgb)
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = train_columns
            fold_importance_df["importance"] = lgb.feature_importances_
            fold_importance_df["fold"] = i
            feature_importance_df = pd.concat(
                [feature_importance_df, fold_importance_df], axis=0)
            i += 1
    else:
        best_scores = []
        team_scores = []
        trained_models = []
        best_iterations = []
        x_train = train_df.drop('target', axis=1).values
        y_train = le.transform(train_df['target'].values)
        train_columns = train_df.drop('target', axis=1).columns
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
            x_trn, y_trn = ros.fit_sample(x_trn, y_trn)

            train_dataset = lightgbm.Dataset(x_trn, y_trn)
            valid_dataset = lightgbm.Dataset(x_val, y_val)
            booster = lightgbm.train(
                PARAMS.copy(), train_dataset,
                num_boost_round=2000,
                fobj=wloss_objective,
                feval=wloss_metric,
                valid_sets=[train_dataset, valid_dataset],
                verbose_eval=100,
                early_stopping_rounds=100
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
            team_score = calc_team_score(y_val, val_pred_score)
            logger.info('team score : {}'.format(team_score))
            team_scores.append(team_score)
            conf_y_true.append(np.argmax(val_pred_score, axis=1))
            conf_y_pred.append(y_val)
            i += 1

        mean_best_score = np.mean(best_scores)
        mean_team_score = np.mean(team_scores)
        mean_best_iteration = np.mean(best_iterations)
        logger.info('mean valid score is {}'.format(mean_best_score))
        logger.info('mean team score is {}'.format(mean_team_score))
        logger.info('mean best iteration is {}'.format(mean_best_iteration))
        models_path = './trained_models/{}_weight-multi-logloss-{:.6}_{}.pkl'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, )
        logger.info('saving models to {} ...'.format(models_path))
        save_models(trained_models, models_path)

        imp_path = './importances/{}_weight-multi-logloss-{:.6}_{}_importance.png'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, )
        logger.info('saving importance to {} ...'.format(models_path))
        save_importance(feature_importance_df, imp_path)

        conf_path = './confusion_matrices/{}_weight-multi-logloss-{:.6}_{}_confusion.png'\
            .format(trained_models[0].__class__.__name__,
                    mean_best_score,
                    start_time, )
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
            x_train, y_train = ros.fit_sample(x_train, y_train)

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

            #            logger.info('loading test_set.csv')
            #            test_set_df = pd.read_feather(
            #                BASE_DIR + 'test_set.fth', nthreads=args.nthread)
            logger.info('loading test_set_metadata.csv')
            test_set_metadata_df = pd.read_csv(
                BASE_DIR + 'test_set_metadata.csv')
            object_ids = test_set_metadata_df.object_id

            logger.info('feature engineering for test set...')
            test_df = feature_engineering(
                None,
                test_set_metadata_df,
                nthread=args.nthread,
                test_flg=True,
                logger=logger)
#            test_df = feature_engineering(
#                test_set_df,
#                test_set_metadata_df,
#                nthread=args.nthread,
#                test_flg=True,
#                logger=logger)
            test_df.reset_index().to_feather('./test_dfs/test_df_for_nn.fth')
            x_test = test_df.values

            logger.info('predicting')
            test_reses = []
            for lgb in tqdm(trained_models):
                test_reses.append(
                    softmax(lgb.predict(x_test, raw_score=False)))
                # test_reses.append(lgb.predict_proba(x_test, raw_score=False))

#            res = np.clip(np.mean(test_reses, axis=0),
#                          10**(-15), 1 - 10**(-15))

            # prediction of linear interpolated
            lin_test_res = \
                    softmax(lin_booster.predict(x_test, raw_score=False))

#            test_reses.append(lin_test_res)
#            temp_filename = './temp/{}_weight-multi-logloss-{:.6}_{}_res.csv'\
#                .format(trained_models[0].__class__.__name__,
#                        mean_best_score,
#                        start_time,)
#            with open(temp_filename, 'wb') as fout:
#                pickle.dump(test_reses + [lin_test_res], fout)

            res = np.clip(np.mean(
                          [np.mean(test_reses, axis=0),
                           lin_test_res],
                          axis=0),
                  10**(-15), 1 - 10**(-15))
            preds_99 = np.ones((res.shape[0]))
            #for i in range(res.shape[1]):
            #    preds_99 *= (1 - res[:, i])
            #preds_99 = 0.14 * preds_99 / np.mean(preds_99)
            res *= 8/9
            preds_99 = 1/9

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
            submission_file_name = './submissions/{}_weight-multi-logloss-{:.6}_{}.csv'\
                .format(trained_models[0].__class__.__name__,
                        mean_best_score,
                        start_time,)
            logger.info(
                'saving the test result to {}'.format(submission_file_name))
            pd.concat([object_ids, res_df], axis=1)\
                .to_csv(submission_file_name, index=False)

    logger.info('finish !')


if __name__ == '__main__':
    args = parse_args()
    main(args)
