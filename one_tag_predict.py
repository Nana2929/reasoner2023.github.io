# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
import torch
import argparse

from config import Config
from utils import now_time, set_seed, get_model, get_trainer, get_dataloader, get_batchify

links = {
    "yelp23": "",
    "gest": "",
}


parser = argparse.ArgumentParser(description='Tag Prediction')

parser.add_argument('--model', '-m', type=str, default='SULM',
                    help='base model name')
parser.add_argument('--dataset_name', '-d', type=str, default='yelp23',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='SULM.yaml',
                    help='config files')
args, _ = parser.parse_known_args()

config_file_list = args.config.strip().split(' ') if args.config else None
config = Config(config_file_list=config_file_list).final_config_dict
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    print('{:40} {}'.format(param, config[param]))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

user_data_path = ...# user_id path


if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])

# Set the random seed manually for reproducibility.
set_seed(config['seed'])
if torch.cuda.is_available():
    if not config['cuda']:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
# device = torch.device('cpu')

device = torch.device('cuda' if config['cuda'] else 'cpu')
if config['cuda']:
    torch.cuda.set_device(config['gpu_id'])

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
root = links.get(args.dataset_name, None)
if root is None:
    raise NotImplementedError('Dataset {} is not implemented.'.format(args.dataset_name))


corpus = get_dataloader(config['model_type'])(dataset_root = root, config=config)
tag_num = corpus.tag_num
user_num = corpus.user_num
item_num = corpus.item_num
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
print(now_time() + '{}: user_num:{} | item_num:{} | tag_num:{}'.format(config['dataset'], user_num, item_num, tag_num))
print(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))

train_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'train')(corpus.trainset, config,
                                                                                                tag_num, shuffle=True)
val_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'valid')(corpus.validset, config,
                                                                                              tag_num)
test_data = get_batchify(config['model_type'], config['model'], config['train_type'], 'test')(corpus.testset, config,
                                                                                              tag_num)

# Bulid the user-item & user-tag & item-tag interaction matrix based on trainset
if config['model'] == 'EFM' or config['model'] == 'AMF':
    X_a, Y_a = corpus.build_inter_matrix(model_name=config['model'])
    config['X_a'] = X_a.to(device)
    config['Y_a'] = Y_a.to(device)
if config['model'] == 'DERM_H':
    config['user_aspect_list'], config['item_aspect_list'] = corpus.build_history_interaction()

###############################################################################
# Update Config
###############################################################################

config['user_num'] = user_num
config['item_num'] = item_num
config['tag_num'] = tag_num
config['max_rating'] = corpus.max_rating
config['min_rating'] = corpus.min_rating
config['device'] = device

###############################################################################
# Build the model
###############################################################################

model = get_model(config['model'])(config).to(device)
trainer = get_trainer(config['model_type'], config['model'])(config, model, train_data, val_data)
###############################################################################
# Loop over epochs
###############################################################################

model_path, best_epoch = trainer.train_loop()

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
print(now_time() + 'Load the best model' + model_path)

# Run on test data.
rmse, mae, score_dict = trainer.evaluate(model, test_data)
pos_aspect_p, pos_aspect_r, pos_aspect_f1, pos_aspect_ndcg = score_dict['pos']
neg_aspect_p, neg_aspect_r, neg_aspect_f1, neg_aspect_ndcg = score_dict['neg']
print('=' * 89)
# Results
print('Best model in epoch {}'.format(best_epoch))
print('Best results: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mae))
print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mae))
print('Best test: positive_sentiment_tag   @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], pos_aspect_p, pos_aspect_r, pos_aspect_f1, pos_aspect_ndcg))
print('Best test: negative_sentiment_tag    @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], neg_aspect_p, neg_aspect_r, neg_aspect_f1, neg_aspect_ndcg))

