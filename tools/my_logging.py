import os

from logging import Formatter, StreamHandler, FileHandler, DEBUG


def logInit(logger, log_dir, log_filename):
    log_fmt = Formatter('%(asctime)s %(name)s \
            %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_dir + log_filename, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    return logger
