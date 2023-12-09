# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import datetime
import importlib
import pickle
import random

import numpy as np
import torch

from data.dataloader import TagDataLoader


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = ['tag_aware_recommender', 'review_aware_recommender']
    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type: str, model_name: str, tagset_num:int):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        tagset_num (int): number of tagset (1 or 2)
        model_name (str): model name
    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == 'review_aware':
            return getattr(importlib.import_module('trainer'), 'ReviewTrainer')
        else:
            if tagset_num == 1:
                return getattr(importlib.import_module('trainer'), 'OneTagTrainer')
            else:
                return getattr(importlib.import_module('trainer'), 'TwoTagTrainer')


def get_dataloader(model_type):
    """Return a dataloader class.

    """

    if model_type == 'tag_aware':
        return TagDataLoader
    else:
        return ReviewDataLoader


def get_batchify(model_type, model_name, train_type, procedure, tagset_num):
    tagset_num_prefix = "OneTag" if tagset_num == 1 else "TwoTag"
    try:
        return getattr(importlib.import_module('data'), model_name + 'Batchify')
    except AttributeError:
        if model_type == 'review_aware':
            return getattr(importlib.import_module('data'), 'ReviewBatchify')
        else:
            if procedure == 'test':
                return getattr(importlib.import_module('data'), f"{tagset_num_prefix}TestBatchify")
            else:
                if train_type == 'bpr':
                    return getattr(importlib.import_module('data'), f"{tagset_num_prefix}NegSamplingBatchify")
                else:
                    return getattr(importlib.import_module('data'), f"{tagset_num_prefix}Batchify")


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']: '


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def set_seed(seed):
    r"""
    set seed for random sampling.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ids2tokens(ids, word2idx, idx2word):  # tranform ids to token_seq(sentence)
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens


def save_pickle(data: any, file_path: str):
    r"""Save data in pickle format.

    Args:
        data (obj): data to be saved
        file_path (str): path of the file to save
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path: str):
    r"""Load data from pickle file.

    Args:
        file_path (str): path of the file to load
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data