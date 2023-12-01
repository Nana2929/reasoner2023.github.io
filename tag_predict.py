# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse
import logging
import os
from pathlib import Path

import torch
from config import Config
from utils import get_batchify, get_dataloader, get_model, get_trainer, now_time, set_seed

links = {
    "yelp23": "",
    "gest": "../enrich_rec_dataset/data/gest",
}


parser = argparse.ArgumentParser(description='Tag Prediction')

parser.add_argument('--model', '-m', type=str, default='SULM',
                    help='base model name')
parser.add_argument('--dataset_name', '-d', type=str, default='gest',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='properties/SULM.yaml',
                    help='config files')
args, _ = parser.parse_known_args()

config_file_list = args.config.strip().split(' ') if args.config else None
config = Config(config_file_list=config_file_list).final_config_dict

###############################################################################
# Logger / Record
###############################################################################
main_log_dir = Path(config['main_log_dir'])
main_log_dir.mkdir(parents=True, exist_ok=True)


# {args.dataset_name}-{args.model}-rating_train_type={config['rating_train_type']}-tagset_num={config['tagset_num']}
exp_log_dir = main_log_dir/ '{}-{}-rating_train_type={}-tagset_num={}-{}'.format(args.dataset_name, args.model, config['rating_train_type'], config['tagset_num'], now_time())
exp_log_dir.mkdir(parents=True, exist_ok=True)
logger_file = exp_log_dir / 'run.log'
os.system('cp {} {}'.format(args.config, exp_log_dir))
fh = logging.FileHandler(logger_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    logger.info('{:40} {}'.format(param, config[param]))
logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])

# Set the random seed manually for reproducibility.
set_seed(config['seed'])
if torch.cuda.is_available():
    if not config['cuda']:
        logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
# device = torch.device('cpu')

device = torch.device('cuda' if config['cuda'] else 'cpu')
if config['cuda']:
    torch.cuda.set_device(config['gpu_id'])

###############################################################################
# Load data
###############################################################################

logger.info(now_time() + 'Loading data')
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
config['rating_num'] = corpus.rating_num
config['dataset'] = args.dataset_name
logger.info(now_time() + '{}: user_num:{} | item_num:{} | tag_num:{}'.format(config['dataset'], user_num, item_num, tag_num))
logger.info(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))
train_data = get_batchify(model_type = config['model_type'],
                          model_name = config['model'],
                          train_type = config['train_type'],
                          procedure = 'train',
                        tagset_num= config['tagset_num']
                          )(corpus.trainset, config, tag_num, shuffle=True)
val_data = get_batchify(model_type = config['model_type'],
                          model_name = config['model'],
                          train_type = config['train_type'],
                          procedure = 'valid',
                        tagset_num= config['tagset_num']
                        )(corpus.validset, config, tag_num)
test_data = get_batchify( model_type = config['model_type'],
                          model_name = config['model'],
                          train_type = config['train_type'],
                          procedure = 'test',
                        tagset_num= config['tagset_num'])(corpus.testset, config, tag_num)

# Bulid the user-item & user-tag & item-tag interaction matrix based on trainset
if config['model'] == 'EFM' or config['model'] == 'AMF':
    X_p, Y_p = corpus.build_inter_matrix(model_name=config['model'])
    X_n, Y_n = corpus.build_inter_matrix(model_name=config['model'])
    config['X_p'] = X_p.to(device)
    config['Y_p'] = Y_p.to(device)
    config['X_n'] = X_n.to(device)
    config['Y_n'] = Y_n.to(device)


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
config['exp_log_dir'] = exp_log_dir

###############################################################################
# Build the model
###############################################################################
tagset_prefix = 'ONE_TAG_' if config['tagset_num'] == 1 else 'TWO_TAG_'
model_name_variant = tagset_prefix + config['model']
model = get_model(model_name_variant)(config).to(device)
trainer = get_trainer(model_type = config['model_type'],
                      model_name = config['model'],
                      tagset_num=config['tagset_num'])(config, model, train_data, val_data)
logger.info(f"Config: {config}")
###############################################################################
# Loop over epochs
###############################################################################

model_path, best_epoch = trainer.train_loop()

###############################################################################
# For prediction
###############################################################################

#============================
# model_path = "/home/P76114511/projects/reasoner2023.github.io/checkpoints/sulm/SULM-gest-Nov-29-2023_15-59-22.pt"
# best_epoch = 0
# =========================

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
logger.info(now_time() + 'Load the best model' + model_path)

# Run on test data.
rmse, mae, score_dict = trainer.evaluate(model, test_data)
pos_aspect_p, pos_aspect_r, pos_aspect_f1, pos_aspect_ndcg = score_dict['pos']
neg_aspect_p, neg_aspect_r, neg_aspect_f1, neg_aspect_ndcg = score_dict['neg']
logger.info('=' * 89)
# Results
logger.info('Best model in epoch {}'.format(best_epoch))
logger.info('Best results: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mae))
logger.info('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(rmse, mae))
logger.info('Best test: positive_sentiment_tag   @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], pos_aspect_p, pos_aspect_r, pos_aspect_f1, pos_aspect_ndcg))
logger.info('Best test: negative_sentiment_tag    @{} precision{:7.4f} | recall {:7.4f} | f1 {:7.5f} | ndcg {:7.5f}'
      .format(config['top_k'], neg_aspect_p, neg_aspect_r, neg_aspect_f1, neg_aspect_ndcg))

