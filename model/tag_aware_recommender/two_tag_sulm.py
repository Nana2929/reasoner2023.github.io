'''
@File    :   two_tag_sulm.py
@Time    :   2023/12/01 00:40:46
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   None
'''

"""
# -*- coding: utf-8 -*-
# @Time   : 2023/02/11
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn
SULM
################################################
Reference:
    Konstantin Bauman et al. "Aspect Based Recommendations: Recommending Items with the Most Valuable Aspects Based on User Reviews." in SIGIR 2016.
"""

import torch
import torch.nn as nn


class TWO_TAG_SULM(nn.Module):
    def __init__(self, config):
        super(TWO_TAG_SULM, self).__init__()

        self.candidate_num = config['candidate_num']
        self.user_tag_embeddings = nn.Parameter(
            torch.Tensor(config['user_num'], config['tag_num'], config['embedding_size']), requires_grad=True)
        self.item_tag_embeddings = nn.Parameter(
            torch.Tensor(config['item_num'], config['tag_num'], config['embedding_size']), requires_grad=True)
        self.tag_num = config['tag_num']

        # tag bias: random initialization (average init in orig code)
        # =========== positive aspect =============
        self.user_pos_bias = nn.Parameter(torch.Tensor(config['user_num'], config['tag_num']), requires_grad=True)
        self.item_pos_bias = nn.Parameter(torch.Tensor(config['item_num'], config['tag_num']), requires_grad=True)
        self.global_pos_bias = nn.Parameter(torch.Tensor(1, config['tag_num']), requires_grad=True)
        # =========== negative aspect =============
        self.user_neg_bias = nn.Parameter(torch.Tensor(config['user_num'], config['tag_num']), requires_grad=True)
        self.item_neg_bias = nn.Parameter(torch.Tensor(config['item_num'], config['tag_num']), requires_grad=True)
        self.global_neg_bias = nn.Parameter(torch.Tensor(1, config['tag_num']), requires_grad=True)

        # coefficients
        self.user_coeff = nn.Parameter(torch.Tensor(config['user_num'], config['tag_num']), requires_grad=True)
        self.item_coeff = nn.Parameter(torch.Tensor(config['item_num'], config['tag_num']), requires_grad=True)
        self.global_coeff = nn.Parameter(torch.Tensor(1, config['tag_num']), requires_grad=True)

        # classifier
        self.rating_classifier = nn.Linear(config['embedding_size']*2, config['rating_num'])

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        # cannot do ce loss of 3 classes, because the aspect scores are 0-1 matrix and has 1 value only
        self.sigmoid = nn.Sigmoid()
        self.device=config['device']
        # https://pytorch.org/docs/stable/generated/torch.bucketize.html


        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.user_tag_embeddings, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_tag_embeddings, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.user_pos_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_pos_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.global_pos_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.user_neg_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_neg_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.global_neg_bias, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.user_coeff, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_coeff, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.global_coeff, mean=1.0 / self.tag_num, std=0.01)

    def forward(self):
        return

    def predict_rating(self, user, item):
        uif_score = (self.get_all_tag_score(user, item, 0)
            + self.get_all_tag_score(user, item, 1)) / 2 # (B, T)
        uif_coeff = self.user_coeff[user] + self.item_coeff[item] + self.global_coeff
        rating = torch.mul(uif_score, uif_coeff).sum(dim=1)
        return rating  # (B)

    def predict_discretized_rating(self, user, item):
        """ concatenate the user and item embeddings and feed into the classifier
        """
        u_emb = self.user_tag_embeddings[user].sum(dim=1)  # (B, E)
        i_emb = self.item_tag_embeddings[item].sum(dim=1)  # (B, E)
        ui_cat_emb = torch.cat((u_emb, i_emb), dim=1)  # (B, 2E)
        rating = self.rating_classifier(ui_cat_emb)  # (B, 5)
        return rating  # (B, 5)


    # Calculate the aspect sentiments predictions based on user and item profile
    # tag_type: 0 for positive, 1 for negative
    def get_all_tag_score(self, user, item, tag_type):
        u_emb = self.user_tag_embeddings[user]  # (B,T,E)
        i_emb = self.item_tag_embeddings[item]  # (B,T,E)
        if tag_type == 0:
            user_bias = self.user_pos_bias
            item_bias = self.item_pos_bias
            global_bias = self.global_pos_bias
        else:
            user_bias = self.user_neg_bias
            item_bias = self.item_neg_bias
            global_bias = self.global_neg_bias

        score = torch.mul(u_emb, i_emb).sum(dim=2) + user_bias[user] + item_bias[item] + global_bias  # (B,T)
        score = self.sigmoid(score)
        return score  # (B, T)


    def predict_pos_score(self, user, item, tag):
        score = torch.tensor([]).to(self.device)
        for i in range(user.size()[0]):
            u_emb = self.user_tag_embeddings[user[i]][tag[i]]  # (C, E)
            i_emb = self.item_tag_embeddings[item[i]][tag[i]]  # (C, E)
            u_bias = self.user_pos_bias[user[i]][tag[i]]  # (C)
            i_bias = self.item_pos_bias[item[i]][tag[i]]  # (C)
            g_bias = self.global_pos_bias[0][tag[i]]  # (C)
            s = torch.mul(u_emb, i_emb).sum(dim=1) + u_bias + i_bias + g_bias  # (C)
            s = self.sigmoid(s).unsqueeze(0)  # (C)->(1,C)
            score = torch.cat((score, s), dim=0)  # (B,C)
        return score  # (B, C)

    def predict_neg_score(self, user, item, tag):
        score = torch.tensor([]).to(self.device)
        for i in range(user.size()[0]):
            u_emb = self.user_tag_embeddings[user[i]][tag[i]]  # (C, E)
            i_emb = self.item_tag_embeddings[item[i]][tag[i]]  # (C, E)
            u_bias = self.user_neg_bias[user[i]][tag[i]]  # (C)
            i_bias = self.item_neg_bias[item[i]][tag[i]]  # (C)
            g_bias = self.global_neg_bias[0][tag[i]]  # (C)
            s = torch.mul(u_emb, i_emb).sum(dim=1) + u_bias + i_bias + g_bias  # (C)
            s = self.sigmoid(s).unsqueeze(0)  # (C)->(1,C)
            score = torch.cat((score, s), dim=0)  # (B,C)
        return score  # (B, C)



    def predict_specific_tag_score(self, user, item, tag, tag_type):
        user_indices = torch.stack([user, tag], dim=1)
        item_indices = torch.stack([item, tag], dim=1)

        u_emb = self.user_tag_embeddings[user_indices[:, 0], user_indices[:, 1]]  # (B, E)
        i_emb = self.item_tag_embeddings[item_indices[:, 0], item_indices[:, 1]]
        if tag_type == 0:

            user_bias = self.user_pos_bias[user_indices[:, 0], user_indices[:, 1]]  # (B,)
            item_bias = self.item_pos_bias[item_indices[:, 0], item_indices[:, 1]]  # (B,)
            global_bias = self.global_pos_bias[0][tag]  # (B,)
            # same as https://github.com/kobauman/SULM/blob/91832a00aa006533db5b551c60e7ce3c692c62b8/sulm.py#L378
        else:
            user_bias = self.user_neg_bias[user_indices[:, 0], user_indices[:, 1]]  # (B,)
            item_bias = self.item_neg_bias[item_indices[:, 0], item_indices[:, 1]]  # (B,)
            global_bias = self.global_neg_bias[0][tag]  # (B,)

        score = torch.mul(u_emb, i_emb).sum(dim=1) + user_bias + item_bias + global_bias # (B,), fit only for binary classification
        score = self.sigmoid(score)
        return score

    def calculate_rating_mseloss(self, user, item, rating_label):
        """
            In order to keep the rating prediction task consistent with other base model settings,
        discard the 0/1 setting in the paper

        """
        predicted_rating = self.predict_rating(user, item)
        rating_loss = self.mse_loss(predicted_rating, rating_label)
        return rating_loss

    def calculate_rating_celoss(self, user, item, rating_label):
        # classify into 3 classes and use cross entropy loss to guide the rating prediction task
        # 0 - 0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        # 1, 2, 3, 4, 5 star
        predicted_discretized_rating = self.predict_discretized_rating(user, item)
        rating_loss = self.ce_loss(predicted_discretized_rating, rating_label)
        return rating_loss


    def calculate_pos_loss(self, user, item, aspect_tag, pos_aspect_label):
        # BCEloss
        pos_aspect_score = self.predict_specific_tag_score(user, item, aspect_tag, tag_type=0)
        pos_aspect_loss = self.bce_loss(pos_aspect_score, pos_aspect_label)
        return pos_aspect_loss

    def calculate_neg_loss(self, user, item, aspect_tag, neg_aspect_label):
        # BCEloss
        neg_aspect_score = self.predict_specific_tag_score(user, item, aspect_tag, tag_type=0)
        neg_aspect_loss = self.bce_loss(neg_aspect_score, neg_aspect_label)
        return neg_aspect_loss

    def calculate_l2_loss(self):
        l2_loss1 = self.user_coeff.norm(2) + self.item_coeff.norm(2) + self.global_coeff.norm(2)
        l2_loss2 = self.user_tag_embeddings.norm(2) + self.item_tag_embeddings.norm(2) + \
                   self.user_pos_bias.norm(2) + self.item_pos_bias.norm(2) + self.global_pos_bias.norm(2) + \
                    self.user_neg_bias.norm(2) + self.item_neg_bias.norm(2) + self.global_neg_bias.norm(2) 

        return l2_loss1 + l2_loss2

    # Ranking score for Recommendations
    def build_prediction(self, user, item):
        final_score = self.predict_rating(user, item)
        return final_score
