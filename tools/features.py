import re
import time
import os
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import pickle

import pandas as pd


def tomap(args):
    return getattr(args[0], args[1])(*args[2:])

def toapply(cls, mtd_name, *args, **kwargs):
    return getattr(cls, mtd_name)(*args, **kwargs)

class MulHelper(object):
    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


class featureCreator(metaclass=ABCMeta):
    """
    feture 毎に load, save 等する設計だと数万 feature とかを 
    concat する際に重くなる。
    よって feature 群毎に扱う設計にする。連続で同じ data を扱う場合は
    init で src_df_dict を使って使い回す。それ以外のデータは load で 
    dataframe レベルで呼び出し、保存は dataframe レベルで行う。

    """
    def __init__(self, load_dir=None, save_dir=None, 
            src_df_dict=None, logger=None, nthread=1):
        if load_dir:
            self.load_dir = load_dir if load_dir[-1] == '/' else load_dir + '/'
        elif not src_df_dict:
            raise 'pleaes set load_dir or src_df_dict at least.'
        if save_dir:
            self.save_dir = save_dir if save_dir[-1] == '/' else save_dir + '/'
        self.src_df_dict = src_df_dict
        self.logger = logger
        self.nthread = nthread

        self.name = self.__class__.__name__
        if src_df_dict:
            self.src_df_dict = src_df_dict
        else:
            self.src_df_dict = {}
        self.df_dict = {}

    def _log_print(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    @contextmanager
    def _timer(self):
        t0 = time.time()
        start_str = f'[{self.name}] start'
        self._log_print(start_str)
        try:
            yield
        finally:
            end_str = f'[{self.name}] done in {time.time() - t0:.0f} s'
            self._log_print(end_str)

    @abstractmethod
    def _create_features(self):
        '''
        create features, and hold the result df as self.df.

        '''
        raise NotImplementedError

    def _load_dfs_from_paths(self, path_dict):
        '''
        path_dict は df_name: path_name の dict

        '''
        self._log_print('now loading features ...')
        self._log_print(f'the path dict is {path_dict}')
        for df_name in tqdm(path_dict):
            path = path_dict[df_name]
            ext_name = path.split('.')[-1]
            if ext_name == 'csv':
                _df = pd.read_csv(path)
            elif ext_name == 'ftr' or ext_name == 'fth':
                _df = pd.read_feather(path, nthreads=self.nthread)
            elif ext_name == 'pkl':
                with open(path, 'rb') as fin:
                    _df = pickle.load(fin)
            else:
                self._log_print(f'the extension {ext_name} is not supported yet.')
                raise NotImplementedError
            self.src_df_dict[df_name] = _df

    @abstractmethod
    def _load(self):
        raise NotImplementedError
        #loaded_features = []
        #for col in tqdm(load_cols):
        #    load_filename = self.load_dir + str(col) + '.ftr'
        #    self._log_print(f'loading {col} from {load_filename}')
        #    loaded_features.append(pd.read_feather(load_filename, nthreads=self.nthread))

    def run(self):
        with self._timer():
            self._load()
        with self._timer():
            self._create_features()
        return self

    def save(self):
        if len(self.df_dict) > 0:
            for key in self.df_dict:
                save_filename =  self.save_dir + key + '.ftr'
                self.df_dict[key].to_feather(save_filename)
            #for col in tqdm(self.df.columns):
            #    save_filename = self.save_dir + str(col) + '.ftr'
            #    if os.path.isfile(save_filename):
            #        self._log_print(f'saving {col} to {save_filename}')
            #        self.df[col].to_feather(save_filename)
        else:
            self._log_print('The creator does not have any dfs to save.')
            self._log_print('Try creating features using run() at first.')
