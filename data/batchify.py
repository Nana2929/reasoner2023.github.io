# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import copy
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Final

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
senti_to_score = {
    "positive": 1,
    "negative": 0,
    "neutral": 0.3,
    "unmentioned": 0.3,
}

def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper


class OneTagBatchify:


    def __init__(self, data, config, tag_num, shuffle=False):
        self.data: list[dict] = data
        self.tag_num: Final[int] = tag_num
        self.neg_sample_num:Final[int] = config["neg_sample_num"]
        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.rating_train_type = config["rating_train_type"] # `mse` or `ce`

        self.batchify_parallel(max_workers=config["num_workers"])




    def batchify(self, start:int, offset:int):
        logging.info(f"[Processing index range] start_index: {start}, offset: {offset}")
        data = self.data[start:offset]
        (user, item, rating, aspect, aspect_label) = ([], [], [], [], [])
        neg_sample_num = self.neg_sample_num

        for x in data:
            #  positive tag
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])

            # aspects with positive sentiments
            pos_aspect_list = x["pos_aspect_tag"]
            # aspects with negative sentiments
            neg_aspect_list = x["neg_aspect_tag"]

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

            # !!add "neutral or not mentioned" tags as negative samples (ns)!!
            # if no aspect_pos_tag or aspect_neg_tag, this vector can be all negative samples.
            for _ in range(neg_sample_num):
                user.append(x["user_id"])
                item.append(x["item_id"])
                rating.append(x["rating"])
                # https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice

                pos_aspect_set = set(pos_aspect_list)
                neg_aspect_set = set(neg_aspect_list)
                ns_aspect = np.random.randint(self.tag_num)
                while (ns_aspect in pos_aspect_set) or (ns_aspect in neg_aspect_set):
                    ns_aspect = np.random.randint(self.tag_num)
                aspect.append(ns_aspect)
                aspect_label.append(senti_to_score["unmentioned"])
        return user, item, rating, aspect, aspect_label

    def batchify_parallel(self, max_workers=int):

        workload = int(math.ceil(len(self.data) / max_workers))
        total_user, total_item, total_rating, total_aspect, total_aspect_label = [], [], [], [], []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use list comprehension to create a list of futures
            future_to_index = {
                executor.submit(self.batchify, i*workload, min((i+1)*workload, len(self.data)), ): i
                for i in range(max_workers)
            }
            futures = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                futures[index] = future
            for i in range(len(futures)):
                # to ensure order
                future = futures[i]
                try:
                    user, item, rating, aspect, aspect_label = future.result()
                    total_user.extend(user)
                    total_item.extend(item)
                    total_rating.extend(rating)
                    total_aspect.extend(aspect)
                    total_aspect_label.extend(aspect_label)

                except Exception as e:
                    logging.info(f"Caught Exception in Parallelization: {e.__class__}")
        # print("total_user", len(total_user))
        # print("total_item", len(total_item))
        # print("total_rating", len(total_rating))
        # print("total_aspect", len(total_aspect))
        # print("total_aspect_label", len(total_aspect_label))
        self.user = torch.tensor(total_user, dtype=torch.int64)
        self.item = torch.tensor(total_item, dtype=torch.int64)
        rating_dtype = torch.float if self.rating_train_type == "mse" else torch.int64
        self.rating = torch.tensor(total_rating, dtype=rating_dtype)
        self.aspect = torch.tensor(total_aspect, dtype=torch.int64)
        self.aspect_label = torch.tensor(total_aspect_label, dtype=torch.float)

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


class TwoTagBatchify:

    def __init__(self, data, config, tag_num, shuffle=False):
        self.data: list[dict] = data
        self.tag_num: Final[int] = tag_num
        self.neg_sample_num:Final[int] = config["neg_sample_num"]
        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.rating_train_type = config["rating_train_type"] # `mse` or `ce`

        self.batchify_parallel(max_workers=config["num_workers"])

    def batchify(self, start:int, offset:int):
        logging.info(f"[Processing index range] start_index: {start}, offset: {offset}")
        data = self.data[start:offset]
        (user, item, rating, pos_aspect, pos_aspect_label, neg_aspect, neg_aspect_label) = ([], [], [], [], [], [], [])
        neg_sample_num = self.neg_sample_num

        for x in data:
            #  positive tag
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])

            # aspects with positive sentiments
            pos_aspect_list = x["pos_aspect_tag"]
            # aspects with negative sentiments
            neg_aspect_list = x["neg_aspect_tag"]

            if len(pos_aspect_list) != 0:
                a_pos_aspect_tag = random.choice(pos_aspect_list)
                pos_aspect.append(a_pos_aspect_tag)
                pos_aspect_label.append(1)

            else:
                neg_sample_num += 1

            if len(neg_aspect_list) != 0:
                a_neg_aspect_tag = random.choice(neg_aspect_list)
                neg_aspect.append(a_neg_aspect_tag)
                neg_aspect_label.append(1)
            else:
                neg_sample_num += 1

            # !!add "neutral or not mentioned" tags as negative samples (ns)!!
            # if no aspect_pos_tag or aspect_neg_tag, this vector can be all negative samples.
            for _ in range(neg_sample_num):
                user.append(x["user_id"])
                item.append(x["item_id"])
                rating.append(x["rating"])
                # https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice

                pos_aspect_set = set(pos_aspect_list)
                neg_aspect_set = set(neg_aspect_list)
                ns_pos = np.random.randint(self.tag_num)
                ns_neg = np.random.randint(self.tag_num)
                while (ns_pos in pos_aspect_set):
                    ns_pos = np.random.randint(self.tag_num)
                while (ns_neg in neg_aspect_set):
                    ns_neg = np.random.randint(self.tag_num)
                pos_aspect.append(ns_pos)
                pos_aspect_label.append(0)
                neg_aspect.append(ns_neg)
                neg_aspect_label.append(0)
        return user, item, rating, pos_aspect, pos_aspect_label, neg_aspect, neg_aspect_label

    def batchify_parallel(self, max_workers=int):

        workload = int(math.ceil(len(self.data) / max_workers))
        (total_user, total_item, total_rating, total_pos_aspect,
         total_pos_aspect_label, total_neg_aspect, total_neg_aspect_label) = ([], [], [], [], [], [], [])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use list comprehension to create a list of futures
            future_to_index = {
                executor.submit(self.batchify, i*workload, min((i+1)*workload, len(self.data)), ): i
                for i in range(max_workers)
            }
            futures = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                futures[index] = future
            for i in range(len(futures)):
                # to ensure order
                future = futures[i]
                try:
                    user, item, rating, pos_aspect, pos_aspect_label, neg_aspect, neg_aspect_label  = future.result()
                    total_user.extend(user)
                    total_item.extend(item)
                    total_rating.extend(rating)
                    total_pos_aspect.extend(pos_aspect)
                    total_pos_aspect_label.extend(pos_aspect_label)
                    total_neg_aspect.extend(neg_aspect)
                    total_neg_aspect_label.extend(neg_aspect_label)

                except Exception as e:
                    logging.info(f"Caught Exception in Parallelization: {e.__class__}")

        self.user = torch.tensor(total_user, dtype=torch.int64)
        self.item = torch.tensor(total_item, dtype=torch.int64)
        rating_dtype = torch.float if self.rating_train_type == "mse" else torch.int64
        self.rating = torch.tensor(total_rating, dtype=rating_dtype)
        self.pos_aspect = torch.tensor(total_pos_aspect, dtype=torch.int64)
        self.pos_aspect_label = torch.tensor(total_pos_aspect_label, dtype=torch.float)
        self.neg_aspect = torch.tensor(total_neg_aspect, dtype=torch.int64)
        self.neg_aspect_label = torch.tensor(total_neg_aspect_label, dtype=torch.float)

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
        pos_aspect = self.pos_aspect[index]
        pos_aspect_label = self.pos_aspect_label[index]
        neg_aspect = self.neg_aspect[index]
        neg_aspect_label = self.neg_aspect_label[index]

        return (user, item, rating, pos_aspect, pos_aspect_label, neg_aspect, neg_aspect_label)



class OneTagTestBatchify:
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
            []
        )
        for x in data:
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])
            pos_aspect.append(x["pos_aspect_tag"])
            neg_aspect.append(x["neg_aspect_tag"])
            candi_aspect.append(
                self.get_candidate_tags(
                    pos_tag_list=x["pos_aspect_tag"],
                    neg_tag_list=x["neg_aspect_tag"],
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




class TwoTagTestBatchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        self.tag_num = tag_num
        (
            user,
            item,
            rating,
            pos_aspect,
            neg_aspect,
            candi_pos_aspect,
            candi_neg_aspect
        ) = (
            [],
            [],
            [],
            [],
            [],
            [], []
        )
        for x in data:
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])
            pos_aspect.append(x["pos_aspect_tag"])
            neg_aspect.append(x["neg_aspect_tag"])
            candi_pos_aspect.append(
                self.get_candidate_tags(
                    pos_tag_list=x["pos_aspect_tag"],
                    neg_tag_list=x["neg_aspect_tag"],
                    candidate_num=config["candidate_num"],
                )
            )
            candi_neg_aspect.append(
                self.get_candidate_tags(
                    pos_tag_list=x["neg_aspect_tag"],
                    neg_tag_list=x["pos_aspect_tag"],
                    candidate_num=config["candidate_num"],
                )
            )

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.candi_pos_aspect_tag = torch.tensor(candi_pos_aspect, dtype=torch.int64)
        self.candi_neg_aspect_tag = torch.tensor(candi_neg_aspect, dtype=torch.int64)

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
        a_candi_pos_aspect_tag = self.candi_pos_aspect_tag[index]
        a_candi_neg_aspect_tag = self.candi_neg_aspect_tag[index]
        return user, item, a_candi_pos_aspect_tag, a_candi_neg_aspect_tag

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
