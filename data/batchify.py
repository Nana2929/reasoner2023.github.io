# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import copy
import math
import random

import numpy as np
import torch

senti_to_score = {
    "positive": 1,
    "negative": 0,
    "neutral": 0.3,
    "unmentioned": 0.3,
}


class Batchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        (user, item, rating, aspect, aspect_label) = ([], [], [], [], [])
        neg_sample_num = config["neg_sample_num"]

        for x in data:
            #  positive tag
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])

            # aspects with positive sentiments
            pos_aspect_list = eval(x["pos_aspect_tag"])
            # aspects with negative sentiments
            neg_aspect_list = eval(x["neg_aspect_tag"])

            if len(pos_aspect_list) != 0:
                a_pos_aspect_tag = random.choice(pos_aspect_list)
                aspect.append(a_pos_aspect_tag)
                aspect_label.append(senti_to_score["positive"])

            else:
                neg_sample_num += 1

            if len(neg_aspect_list) != 0:
                a_neg_aspect_tag = random.choice(neg_aspect_list)
                aspect.append(a_neg_aspect_tag)
                aspect_label.append(senti_to_score["negative"])
            else:
                neg_sample_num += 1

            # add "neutral or not mentioned" tags as negative samples (ns)
            # if no aspect_pos_tag or aspect_neg_tag, this vector can be all negative samples.
            for _ in range(neg_sample_num):
                user.append(x["user_id"])
                item.append(x["item_id"])
                rating.append(x["rating"])
                ns_aspect = np.random.randint(tag_num)

                while ns_aspect in pos_aspect_list or ns_aspect in neg_aspect_list:
                    ns_aspect = np.random.randint(tag_num)
                aspect.append(ns_aspect)
                aspect_label.append(senti_to_score["unmentioned"])

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)

        self.aspect = torch.tensor(aspect, dtype=torch.int64)
        self.aspect_label = torch.tensor(aspect_label, dtype=torch.float)

        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.tag_num = tag_num

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        aspect = self.aspect[index]
        aspect_label = self.aspect_label[index]

        return (user, item, rating, aspect, aspect_label)


class TagTestBatchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        self.tag_num = tag_num
        (
            user,
            item,
            rating,
            pos_aspect,
            neg_aspect,
            candi_aspect,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        for x in data:
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])
            pos_aspect.append(eval(x["pos_aspect_tag"]))
            neg_aspect.append(eval(x["neg_aspect_tag"]))
            candi_aspect.append(
                self.get_candidate_tags(
                    pos_tag_list=eval(x["pos_aspect_tag"]),
                    neg_tag_list=eval(x["neg_aspect_tag"]),
                    candidate_num=config["candidate_num"],
                )
            )

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.candi_aspect_tag = torch.tensor(candi_aspect, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

        # tag acquisition, used to calculate sorting indicators, only used in the test phase
        self.positive_aspect_tag = pos_aspect
        self.negative_aspect_tag = neg_aspect

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        candi_aspect_tag = self.candi_aspect_tag[index]
        return user, item, candi_aspect_tag

    def get_candidate_tags(
        self, pos_tag_list: list, neg_tag_list: list, candidate_num: int
    ):
        """
        pos_tag_list: list of positive tags, e.g. [1, 2, 3, 3, 4, 5], an aspect tag should repeat if it has multiple
        neg_tag_list: list of negative tags
        candidate_num: number of candidate tags to be sampled
        """
        # ns: negative sampling
        # neg: negative sentiments

        #  What if for an aspect (category), there are both positive and negative sentiments?
        # see https://github.com/users/Nana2929/projects/4?pane=issue&itemId=45486697
        # solution: vote by sentiment number (dicussed w/ DNan on 2023.11.24)
        cleaned_pos_tag_list, cleaned_neg_tag_list = [], []
        all_tags = set(pos_tag_list + neg_tag_list)
        for tag in all_tags:
            if pos_tag_list.count(tag) > neg_tag_list.count(tag):
                cleaned_pos_tag_list.append(tag)
            elif pos_tag_list.count(tag) < neg_tag_list.count(tag):
                cleaned_neg_tag_list.append(tag)
            else:
                pass

        ns_tag_num = max(candidate_num - len(pos_tag_list) - len(neg_tag_list), 0)
        # copy pos_tag_list and neg_tag_list to candi_tag_list
        candi_tag_list = copy.deepcopy(pos_tag_list) + copy.deepcopy(neg_tag_list)
        for _ in range(ns_tag_num):
            ns_tag = np.random.randint(self.tag_num)
            while ns_tag in pos_tag_list or ns_tag in neg_tag_list:
                ns_tag = np.random.randint(self.tag_num)
            candi_tag_list.append(ns_tag)
        random.shuffle(candi_tag_list)
        return candi_tag_list
