import datetime
import argparse
import gc
from logging import getLogger


from my_logging import logInit
from plasticc_features import featureCreatorPreprocess, featureCreatorSet
from plasticc_features import fe_set_df_base, fe_set_df_detected

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
    detected_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_detected,
            set_res_df_name='set_base_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    detected_feat_creator.run().save()
    del detected_feat_creator
    gc.collect()

    # std upper aggregation
    std_upper_feat_creator = featureCreatorSet(
            fe_set_df=fe_set_df_detected,
            set_res_df_name='set_base_features',
            load_dir=LOAD_DIR,
            save_dir=SAVE_DIR,
            src_df_dict=preprocessed_src_df_dict,
            logger=logger,
            nthread=args.nthread)
    detected_feat_creator.run().save()
    del detected_feat_creator
    gc.collect()




    ### feature engineerings using created features


if __name__ == '__main__':
    args = parse_args()
    main(args)
