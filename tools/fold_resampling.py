#
# the tool for resamping each target for column
#

#    ros = RandomOverSampler(
#        ratio={
#            0: max(151, SAMPLING_LOWER),
#            1: max(495, SAMPLING_LOWER),
#            2: max(924, SAMPLING_LOWER),
#            3: max(1193, SAMPLING_LOWER),
#            4: max(183, SAMPLING_LOWER),
#            5: max(30, SAMPLING_LOWER),
#            6: max(484, SAMPLING_LOWER),
#            7: max(102, SAMPLING_LOWER),
#            8: max(981, SAMPLING_LOWER),
#            9: max(208, SAMPLING_LOWER),
#            10: max(370, SAMPLING_LOWER),
#            11: max(2313, SAMPLING_LOWER),
#            12: max(239, SAMPLING_LOWER),
#            13: max(175, SAMPLING_LOWER),
#        }, random_state=71)
#    x_train, y_train = ros.fit_sample(x_train, y_train)


def get_fold_resampling_dict(y_sample, logger,
                             sampling_lower, sampling_lower_rate):
    fold_resampling_dict = {}
    targets = [i for i in range(14)]
    for target in targets:
        fold_resampling_dict[target] = y_sample[y_sample == target].shape[0]
    logger.debug('fold_samples_num : {}'.format(fold_resampling_dict))
    for target in fold_resampling_dict.keys():
        fold_resampling_dict[target] = \
            int(max(fold_resampling_dict[target], sampling_lower))
#        if sampling_lower > fold_resampling_dict[target]:
#            fold_resampling_dict[target] = \
#                int(max(fold_resampling_dict[target], sampling_lower))
#                int(fold_resampling_dict[target] * sampling_lower_rate)
    logger.info('resampled fold_samples_num : {}'.format(fold_resampling_dict))
    return fold_resampling_dict
