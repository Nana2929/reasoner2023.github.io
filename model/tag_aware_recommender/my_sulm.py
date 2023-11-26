# -*- coding: utf-8 -*-
# @Time   : 2023/02/11
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
SULM
################################################
Reference:
    Konstantin Bauman et al. "Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews." in SIGIR 2016.
"""

import torch
import torch.nn as nn


class SULM(nn.Module):
    def __init__(self, config):
        super(SULM, self).__init__()

        self.candidate_num = config['candidate_num']
        self.user_tag_embeddings = nn.Parameter(
            torch.Tensor(config['user_num'], config['tag_num'], config['embedding_size']), requires_grad=True)
        self.item_tag_embeddings = nn.Parameter(
            torch.Tensor(config['item_num'], config['tag_num'], config['embedding_size']), requires_grad=True)
        self.tag_num = config['tag_num']

        # tag bias: random initialization (average init in orig code)
        self.user_aspect_bias = nn.Parameter(torch.Tensor(config['user_num'], config['tag_num']), requires_grad=True)
        self.item_aspect_bias = nn.Parameter(torch.Tensor(config['item_num'], config['tag_num']), requires_grad=True)
        self.global_aspect_bias = nn.Parameter(torch.Tensor(1, config['tag_num']), requires_grad=True)

        # coefficients
        self.user_coeff = nn.Parameter(torch.Tensor(config['user_num'], config['tag_num']), requires_grad=True)
        self.item_coeff = nn.Parameter(torch.Tensor(config['item_num'], config['tag_num']), requires_grad=True)
        self.global_coeff = nn.Parameter(torch.Tensor(1, config['tag_num']), requires_grad=True)

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.device=config['device']

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.user_tag_embeddings, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_tag_embeddings, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.user_aspect_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_aspect_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.global_aspect_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.user_coeff, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_coeff, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.global_coeff, mean=1.0 / self.tag_num, std=0.01)

    def forward(self):
        return

    def predict_rating(self, user, item):
        uif_score = (self.get_all_tag_score(user, item, 0) + self.get_all_tag_score(user, item, 1) +
                     self.get_all_tag_score(user, item, 2)) / 3
        uif_coeff = self.user_coeff[user] + self.item_coeff[item] + self.global_coeff
        rating = torch.mul(uif_score, uif_coeff).sum(dim=1)
        return rating  # (B)

    # Calculate the aspect sentiments predictions based on user and item profile
    def get_all_tag_score(self, user, item, tag_type):
        u_emb = self.user_tag_embeddings[user]  # (B,T,E)
        i_emb = self.item_tag_embeddings[item]  # (B,T,E)
        user_bias = self.user_aspect_bias
        item_bias = self.item_aspect_bias
        global_bias = self.global_aspect_bias

        score = torch.mul(u_emb, i_emb).sum(dim=2) + user_bias[user] + item_bias[item] + global_bias  # (B,T)
        score = self.sigmoid(score)
        return score  # (B, T)


    def predict_aspect_score(self, user, item, tag):
        score = torch.tensor([]).to(self.device)
        for i in range(user.size()[0]):
            u_emb = self.user_tag_embeddings[user[i]][tag[i]]  # (C, E)
            i_emb = self.item_tag_embeddings[item[i]][tag[i]]  # (C, E)
            u_bias = self.user_aspect_bias[user[i]][tag[i]]  # (C)
            i_bias = self.item_aspect_bias[item[i]][tag[i]]  # (C)
            g_bias = self.global_aspect_bias[0][tag[i]]  # (C)
            s = torch.mul(u_emb, i_emb).sum(dim=1) + u_bias + i_bias + g_bias  # (C)
            s = self.sigmoid(s).unsqueeze(0)  # (C)->(1,C)
            score = torch.cat((score, s), dim=0)  # (B,C)
        return score  # (B, C)


    def predict_specific_tag_score(self, user, item, tag, tag_type):
        user_indices = torch.stack([user, tag], dim=1)
        item_indices = torch.stack([item, tag], dim=1)

        u_emb = self.user_tag_embeddings[user_indices[:, 0], user_indices[:, 1]]  # (B, E)
        i_emb = self.item_tag_embeddings[item_indices[:, 0], item_indices[:, 1]]

        user_bias = self.user_aspect_bias[user_indices[:, 0], user_indices[:, 1]]  # (B,)
        item_bias = self.item_aspect_bias[item_indices[:, 0], item_indices[:, 1]]  # (B,)
        global_bias = self.global_aspect_bias[0][tag]  # (B,)
        # same as https://github.com/kobauman/SULM/blob/91832a00aa006533db5b551c60e7ce3c692c62b8/sulm.py#L378
        score = torch.mul(u_emb, i_emb).sum(dim=1) + user_bias + item_bias + global_bias
        score = self.sigmoid(score)
        return score

    def calculate_rating_loss(self, user, item, rating_label):
        """
            In order to keep the rating prediction task consistent with other base model settings,
        discard the 0/1 setting in the paper

        """
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_aspect_loss(self, user, item, aspect_tag, aspect_label):
        # BCEloss
        aspect_score = self.predict_specific_tag_score(user, item, aspect_tag, tag_type=0)
        aspect_loss = self.bce_loss(aspect_score, aspect_label)
        return aspect_loss

    def calculate_l2_loss(self):
        l2_loss1 = self.user_coeff.norm(2) + self.item_coeff.norm(2) + self.global_coeff.norm(2)
        l2_loss2 = self.user_tag_embeddings.norm(2) + self.item_tag_embeddings.norm(2) + \
                   self.user_aspect_bias.norm(2) + self.item_aspect_bias.norm(2) + self.global_aspect_bias.norm(2)
        return l2_loss1 + l2_loss2

    # Ranking score for Recommendations
    def build_prediction(self, user, item):
        final_score = self.predict_rating(user, item)
        return final_score
