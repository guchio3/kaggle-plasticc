import datetime
import argparse
import gc
from logging import getLogger


from my_logging import logInit
from plasticc_features import featureCreatorPreprocess, featureCreatorSet
from plasticc_features import fe_set_df_base, fe_set_df_detected, fe_set_df_std_upper_and_lower, fe_set_df_passband, fe_set_df_passband_std_upper, featureCreatorTsfresh, featureCreatorMeta, fe_meta, fe_set_df_passband_detected, fe_set_df_peak_around, fe_set_df_ratsq_peak_around

LOAD_DIR = '/home/naoya.taguchi/.kaggle/competitions/PLAsTiCC-2018/'
SAVE_DIR_BASE = '../features/'


def parse_args():
    parser = argparse.ArgumentParser(
        prog='train.py',
        usage='ex) python train.py --with_test',
        description='easy explanation',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-t', '--train',
                        help='flg to specify test type.',
                        action='store_true',
                        default=False)
    parser.add_argument('-n', '--nthread',
                        help='number of avalable threads.',
                        type=int,
                        required=True)

    args = parser.parse_args()
    return args


def main(args):
    logger = getLogger(__name__)
    logInit(logger, log_dir='../log/', log_filename='feature_engineering.log')
    logger.info(
        '''
        start main, the args settings are ...
        --train : {}
        '''.format(args.train))

    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger.info('start training, the starting time is {}'.format(start_time))

    if args.train:
        SAVE_DIR = SAVE_DIR_BASE + 'train/'
    else:
        SAVE_DIR = SAVE_DIR_BASE + 'test/'

    # preprocess
    logger.info('preprocessing set dfs ...')
    prep_feat_creator = featureCreatorPreprocess(
            load_dir=LOAD_DIR,
            save_dir=None,
            src_df_dict=None,
            logger=logger,
            nthread=args.nthread,
            train=args.train)
    prep_feat_creator.run()


    ### feature engineerings using preprocessed_src_df_dict
    preprocessed_src_df_dict = prep_feat_creator.src_df_dict

    # basic aggregations
    logger.info('creating basic features ...')
    base_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_base,
            set_res_df_name='set_base_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    base_feat_creator.run().save()
    del base_feat_creator
    gc.collect()

    # detected aggregations
    logger.info('creating detected features ...')
    detected_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_detected,
            set_res_df_name='set_detected_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    detected_feat_creator.run().save()
    del detected_feat_creator
    gc.collect()

    # std upper aggregation
    logger.info('creating std upper features ...')
    std_upper_and_lower_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_std_upper_and_lower,
            set_res_df_name='set_std_upper_and_lower_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    std_upper_and_lower_feat_creator.run().save()
    del std_upper_and_lower_feat_creator
    gc.collect()

    # passband aggregation
    logger.info('creating passband features ...')
    passband_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_passband,
            set_res_df_name='set_passband_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    passband_feat_creator.run().save()
    del passband_feat_creator
    gc.collect()

    # passband std upper aggregation
    logger.info('creating passband std upper features ...')
    passband_std_upper_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_passband_std_upper,
            set_res_df_name='set_passband_std_upper_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    passband_std_upper_feat_creator.run().save()
    del passband_std_upper_feat_creator
    gc.collect()

    # passband detected aggregation
    logger.info('creating passband detected features ...')
    passband_detected_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_passband_detected,
            set_res_df_name='set_passband_detected_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    passband_detected_feat_creator.run().save()
    del passband_detected_feat_creator
    gc.collect()

    # peak around
    logger.info('creating ratsq peak around features ...')
    ratsq_peak_around_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_ratsq_peak_around,
            set_res_df_name='set_ratsq_peak_around_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
#    ratsq_peak_around_feat_creator.run().save()

    del ratsq_peak_around_feat_creator
    gc.collect()

    # peak around
    logger.info('creating peak around features ...')
    peak_around_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_peak_around,
            set_res_df_name='set_peak_around_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
#    peak_around_feat_creator.run().save()

    del peak_around_feat_creator
    gc.collect()

    del preprocessed_src_df_dict
    gc.collect()


    ### ts fresh features
    logger.info('creating tsfresh features ...')
    tsfresh_feat_creator = featureCreatorTsfresh(
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=None,
            logger=logger,
            nthread=args.nthread,
            train=args.train)
#    tsfresh_feat_creator.run().save()
    del tsfresh_feat_creator
    gc.collect()


    ### feature engineerings using created features
    logger.info('feature engineering on aggregated df ...')
    meta_feat_creator = featureCreatorMeta(
            fe_set_df=fe_meta,
            set_res_df_name='meta_features',
            load_dir=SAVE_DIR,
            #load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=None,
            logger=logger,
            nthread=args.nthread,
            train=args.train)
    meta_feat_creator.run().save()
    del meta_feat_creator
    gc.collect()




if __name__ == '__main__':
    args = parse_args()
    main(args)
