# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

'''
@File    :   two_tag_mter.py
@Time    :   2023/12/05 22:29:29
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Revision: Ching-Wen Yang
MTER uses NegSamplingBatchify().
To avoid confusion of negative sentiments with negative sampling,
for sentiment polairites, we use good/bad;
for positive/negtive sampling, we use pos/neg.
'''

r"""
MTER
################################################
Reference:
    Nan Wang et al. "Explainable Recommendation via Multi-Task Learning in Opinionated Text Data." in SIGIR 2018.
"""
import torch
import torch.nn as nn

from model.loss import BPRLoss


class TWO_TAG_MTER(nn.Module):
    r"""MTER is a multi-task learning solution for explainable recommendation. Two companion tasks of user preference
    modeling for recommendation and opinionated content modeling for explanation are integrated via a joint tensor factorization.

    We only focus to the part of user-item-aspect since this task does not involve the processing of review text.

    Method: Matrix multiplication implementation for faster running.

    """

    def __init__(self, config):
        super(TWO_TAG_MTER, self).__init__()

        self.device =config['device']
        self.candidate_num=config['candidate_num']
        self.user_embeddings = nn.Embedding(config['user_num'], config['u_emb_size'])
        self.item_embeddings = nn.Embedding(config['item_num'], config['i_emb_size'])
        self.core_tensor = nn.Parameter(torch.Tensor(config['u_emb_size'], config['i_emb_size'], config['t_emb_size']),
                                        requires_grad=True)
        self.good_tag_embeddings = nn.Embedding(config['tag_num'], config['t_emb_size'])
        self.bad_tag_embeddings = nn.Embedding(config['tag_num'], config['t_emb_size'])
        self.tag_num = config['tag_num']

        self.bpr_loss = BPRLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=0.01)

    def forward(self):
        return

    def predict_rating(self, user, item):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        rating = torch.mul(u_emb, i_emb).sum(dim=1)
        return rating

    def predict_tag_score(self, u_emb, i_emb, t_emb):
        i_emb = i_emb.unsqueeze(-1)  # (B,E,1)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(1)  # (B,1,E,1)
        core_tensor = self.core_tensor.unsqueeze(0)  # (1,E,E,E)

        t_score = torch.matmul(core_tensor, t_emb).squeeze(-1)  #
        t_score = torch.matmul(t_score, i_emb).squeeze(-1)
        t_score = torch.mul(t_score, u_emb).sum(-1)
        return t_score  # (B)

    def calculate_rating_mseloss(self, user, item, rating_label):
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_good_aspect_loss(self, user, item, pos_tag, neg_tag):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        pos_emb = self.good_tag_embeddings(pos_tag)
        neg_emb = self.good_tag_embeddings(neg_tag)
        pos_score = self.predict_tag_score(u_emb, i_emb, pos_emb)
        neg_score = self.predict_tag_score(u_emb, i_emb, neg_emb)
        good_aspect_loss = self.bpr_loss(pos_score, neg_score)
        return good_aspect_loss


    def calculate_bad_aspect_loss(self, user, item, pos_tag, neg_tag):
        u_emb = self.user_embeddings(user)
        i_emb = self.item_embeddings(item)
        pos_emb = self.bad_tag_embeddings(pos_tag)
        neg_emb = self.bad_tag_embeddings(neg_tag)
        pos_score = self.predict_tag_score(u_emb, i_emb, pos_emb)
        neg_score = self.predict_tag_score(u_emb, i_emb, neg_emb)
        bad_aspect_loss = self.bpr_loss(pos_score, neg_score)
        return bad_aspect_loss

    def calculate_non_negative_reg(self):
        u_reg = torch.sum((torch.abs(self.user_embeddings.weight) - self.user_embeddings.weight))
        i_reg = torch.sum((torch.abs(self.item_embeddings.weight) - self.item_embeddings.weight))
        good_aspect_reg = torch.sum((torch.abs(self.good_tag_embeddings.weight) - self.good_tag_embeddings.weight))
        bad_aspect_reg = torch.sum((torch.abs(self.bad_tag_embeddings.weight) - self.bad_tag_embeddings.weight))
        core_tensor_reg = torch.sum((torch.abs(self.core_tensor) - self.core_tensor))
        non_negative_reg = u_reg + i_reg + good_aspect_reg + bad_aspect_reg + core_tensor_reg
        return non_negative_reg

    def rank_good_tags(self, user, item, tag):
        score = torch.tensor([]).to(self.device)
        for i in range(user.size()[0]):
            u_emb = self.user_embeddings(user[i]).repeat(tag.size()[1], 1)  # (1,E) -> (C,E)
            i_emb = self.item_embeddings(item[i]).repeat(tag.size()[1], 1)  # (1,E) -> (C,E)
            t_emb = self.good_tag_embeddings(tag[i])  # (C,E)
            s = self.predict_tag_score(u_emb, i_emb, t_emb).unsqueeze(0)  # (C)->(1,C)
            score = torch.cat((score, s), dim=0)  # (B,C)
        return score

    def rank_bad_tags(self, user, item, tag):
        score = torch.tensor([]).to(self.device)
        for i in range(user.size()[0]):
            u_emb = self.user_embeddings(user[i]).repeat(tag.size()[1], 1)  # (1,E) -> (C,E)
            i_emb = self.item_embeddings(item[i]).repeat(tag.size()[1], 1)  # (1,E) -> (C,E)
            t_emb = self.bad_tag_embeddings(tag[i])  # (C,E)
            s = self.predict_tag_score(u_emb, i_emb, t_emb).unsqueeze(0)  # (C)
            score = torch.cat((score, s), dim=0)
        return score

