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
            good_aspect_list = x["pos_aspect_tag"]
            # aspects with negative sentiments
            bad_aspect_list = x["neg_aspect_tag"]

            if len(good_aspect_list) != 0:
                a_good_aspect_tag = random.choice(good_aspect_list)
                aspect.append(a_good_aspect_tag)
                aspect_label.append(senti_to_score["positive"])

            else:
                neg_sample_num += 1

            if len(bad_aspect_list) != 0:
                a_bad_aspect_tag = random.choice(bad_aspect_list)
                aspect.append(a_bad_aspect_tag)
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

                good_aspect_set = set(good_aspect_list)
                bad_aspect_set = set(bad_aspect_list)
                ns_aspect = np.random.randint(self.tag_num)
                while (ns_aspect in good_aspect_set) or (ns_aspect in bad_aspect_set):
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
        if self.rating_train_type == "ce":
            total_rating = [r-1 for r in total_rating] # 1,2,3,4,5 -> 0,1,2,3,4
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
        (user, item, rating, good_aspect, good_aspect_label, bad_aspect, bad_aspect_label) = ([], [], [], [], [], [], [])
        neg_sample_num = self.neg_sample_num

        for x in data:
            #  positive tag
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])

            # aspects with positive sentiments
            good_aspect_list = x["pos_aspect_tag"]
            # aspects with negative sentiments
            bad_aspect_list = x["neg_aspect_tag"]

            if len(good_aspect_list) != 0:
                a_good_aspect_tag = random.choice(good_aspect_list)
                good_aspect.append(a_good_aspect_tag)
                good_aspect_label.append(1)

            else:
                neg_sample_num += 1

            if len(bad_aspect_list) != 0:
                a_bad_aspect_tag = random.choice(bad_aspect_list)
                bad_aspect.append(a_bad_aspect_tag)
                bad_aspect_label.append(1)
            else:
                neg_sample_num += 1

            # !!add "neutral or not mentioned" tags as negative samples (ns)!!
            # if no aspect_pos_tag or aspect_neg_tag, this vector can be all negative samples.
            for _ in range(neg_sample_num):
                user.append(x["user_id"])
                item.append(x["item_id"])
                rating.append(x["rating"])
                # https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice

                good_aspect_set = set(good_aspect_list)
                bad_aspect_set = set(bad_aspect_list)
                ns_pos = np.random.randint(self.tag_num)
                ns_neg = np.random.randint(self.tag_num)
                while (ns_pos in good_aspect_set):
                    ns_pos = np.random.randint(self.tag_num)
                while (ns_neg in bad_aspect_set):
                    ns_neg = np.random.randint(self.tag_num)
                good_aspect.append(ns_pos)
                good_aspect_label.append(0)
                bad_aspect.append(ns_neg)
                bad_aspect_label.append(0)
        return user, item, rating, good_aspect, good_aspect_label, bad_aspect, bad_aspect_label

    def batchify_parallel(self, max_workers=int):

        workload = int(math.ceil(len(self.data) / max_workers))
        (total_user, total_item, total_rating, total_good_aspect,
         total_good_aspect_label, total_bad_aspect, total_bad_aspect_label) = ([], [], [], [], [], [], [])
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
                    user, item, rating, good_aspect, good_aspect_label, bad_aspect, bad_aspect_label  = future.result()
                    total_user.extend(user)
                    total_item.extend(item)
                    total_rating.extend(rating)
                    total_good_aspect.extend(good_aspect)
                    total_good_aspect_label.extend(good_aspect_label)
                    total_bad_aspect.extend(bad_aspect)
                    total_bad_aspect_label.extend(bad_aspect_label)

                except Exception as e:
                    logging.info(f"Caught Exception in Parallelization: {e.__class__}")

        self.user = torch.tensor(total_user, dtype=torch.int64)
        self.item = torch.tensor(total_item, dtype=torch.int64)
        rating_dtype = torch.float if self.rating_train_type == "mse" else torch.int64
        if self.rating_train_type == "ce":
            total_rating = [r-1 for r in total_rating] # 1,2,3,4,5 -> 0,1,2,3,4
        self.rating = torch.tensor(total_rating, dtype=rating_dtype)
        self.good_aspect = torch.tensor(total_good_aspect, dtype=torch.int64)
        self.good_aspect_label = torch.tensor(total_good_aspect_label, dtype=torch.float)
        self.bad_aspect = torch.tensor(total_bad_aspect, dtype=torch.int64)
        self.bad_aspect_label = torch.tensor(total_bad_aspect_label, dtype=torch.float)

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
        good_aspect = self.good_aspect[index]
        good_aspect_label = self.good_aspect_label[index]
        bad_aspect = self.bad_aspect[index]
        bad_aspect_label = self.bad_aspect_label[index]

        return (user, item, rating, good_aspect, good_aspect_label, bad_aspect, bad_aspect_label)



class OneTagTestBatchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        self.tag_num = tag_num
        (
            user,
            item,
            rating,
            good_aspect,
            bad_aspect,
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
            good_aspect.append(x["pos_aspect_tag"])
            bad_aspect.append(x["neg_aspect_tag"])
            candi_aspect.append(
                self.get_candidate_tags(
                    good_tag_list=x["pos_aspect_tag"],
                    bad_tag_list=x["neg_aspect_tag"],
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
        self.positive_aspect_tag = good_aspect
        self.negative_aspect_tag = bad_aspect

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
        self, good_tag_list: list, bad_tag_list: list, candidate_num: int
    ):
        """
        good_tag_list: list of positive sentiment tags, e.g. [1, 2, 3, 3, 4, 5], an aspect tag should repeat if it has multiple
        bad_tag_list: list of negative sentiment tags
        candidate_num: number of candidate tags to be sampled
        """
        # ns: negative sampling
        # neg: negative sentiments

        #  What if for an aspect (category), there are both positive and negative sentiments?
        # see https://github.com/users/Nana2929/projects/4?pane=issue&itemId=45486697
        # solution: vote by sentiment number (dicussed w/ DNan on 2023.11.24)
        cleaned_good_tag_list, cleaned_bad_tag_list = [], []
        all_tags = set(good_tag_list + bad_tag_list)
        for tag in all_tags:
            if good_tag_list.count(tag) > bad_tag_list.count(tag):
                cleaned_good_tag_list.append(tag)
            elif good_tag_list.count(tag) < bad_tag_list.count(tag):
                cleaned_bad_tag_list.append(tag)
            else:
                pass

        ns_tag_num = max(candidate_num - len(good_tag_list) - len(bad_tag_list), 0)
        # copy good_tag_list and bad_tag_list to candi_tag_list
        candi_tag_list = copy.deepcopy(good_tag_list) + copy.deepcopy(bad_tag_list)
        for _ in range(ns_tag_num):
            ns_tag = np.random.randint(self.tag_num)
            while ns_tag in good_tag_list or ns_tag in bad_tag_list:
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
            good_aspect,
            bad_aspect,
            candi_good_aspect,
            candi_bad_aspect
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
            good_aspect.append(x["pos_aspect_tag"])
            bad_aspect.append(x["neg_aspect_tag"])
            candi_good_aspect.append(
                self.get_candidate_tags(
                    good_tag_list=x["pos_aspect_tag"],
                    bad_tag_list=x["neg_aspect_tag"],
                    candidate_num=config["candidate_num"],
                )
            )
            candi_bad_aspect.append(
                self.get_candidate_tags(
                    good_tag_list=x["neg_aspect_tag"],
                    bad_tag_list=x["pos_aspect_tag"],
                    candidate_num=config["candidate_num"],
                )
            )

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.candi_good_aspect_tag = torch.tensor(candi_good_aspect, dtype=torch.int64)
        self.candi_bad_aspect_tag = torch.tensor(candi_bad_aspect, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

        # tag acquisition, used to calculate sorting indicators, only used in the test phase
        self.positive_aspect_tag = good_aspect
        self.negative_aspect_tag = bad_aspect

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
        a_candi_good_aspect_tag = self.candi_good_aspect_tag[index]
        a_candi_bad_aspect_tag = self.candi_bad_aspect_tag[index]
        return user, item, a_candi_good_aspect_tag, a_candi_bad_aspect_tag

    def get_candidate_tags(
        self, good_tag_list: list, bad_tag_list: list, candidate_num: int
    ):
        """
        good_tag_list: list of positive sentiment tags, e.g. [1, 2, 3, 3, 4, 5], an aspect tag should repeat if it has multiple
        bad_tag_list: list of negative sentiment tags
        candidate_num: number of candidate tags to be sampled
        """
        # ns: negative sampling
        # neg: negative sentiments

        #  What if for an aspect (category), there are both positive and negative sentiments?
        # see https://github.com/users/Nana2929/projects/4?pane=issue&itemId=45486697
        # solution: vote by sentiment number (dicussed w/ DNan on 2023.11.24)
        cleaned_good_tag_list, cleaned_bad_tag_list = [], []
        all_tags = set(good_tag_list + bad_tag_list)
        for tag in all_tags:
            if good_tag_list.count(tag) > bad_tag_list.count(tag):
                cleaned_good_tag_list.append(tag)
            elif good_tag_list.count(tag) < bad_tag_list.count(tag):
                cleaned_bad_tag_list.append(tag)
            else:
                pass

        ns_tag_num = max(candidate_num - len(good_tag_list) - len(bad_tag_list), 0)
        # copy good_tag_list and bad_tag_list to candi_tag_list
        candi_tag_list = copy.deepcopy(good_tag_list) + copy.deepcopy(bad_tag_list)
        for _ in range(ns_tag_num):
            ns_tag = np.random.randint(self.tag_num)
            while ns_tag in good_tag_list or ns_tag in bad_tag_list:
                ns_tag = np.random.randint(self.tag_num)
            candi_tag_list.append(ns_tag)
        random.shuffle(candi_tag_list)
        return candi_tag_list


class TwoTagNegSamplingBatchify:
    r"""
    The function of negative sampling is provided for the label prediction task, and it is only used in the
    training phase.
    This is first revised due to MTER model.
    Has to be parallelized.
    """

    def __init__(self, data, config, tag_num, shuffle=False):

        # batchify parallel
        self.data = data
        self.neg_sample_num: Final[int] = config["neg_sample_num"]
        self.shuffle: Final[bool] = shuffle
        self.batch_size: Final[int] = config["batch_size"]
        self.tag_num: Final[int] = tag_num

        self.batchify_parallel(max_workers=config["num_workers"])

        self.sample_num: Final[int] = len(self.user)
        self.index_list: list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0


    def batchify(self, start:int, offset:int):
        logging.info(f"[Processing index range] start_index: {start}, offset: {offset}")
        (
            user,
            item,
            rating,
            good_aspect_pos,
            good_aspect_neg,
            bad_aspect_pos,
            bad_aspect_neg,
        ) = ([], [], [], [], [], [], [])
        data = self.data[start:offset]
        for x in data:
            good_aspect_list = x["pos_aspect_tag"]
            bad_aspect_list = x["neg_aspect_tag"]

            a_good_tag = random.choice(good_aspect_list)
            a_bad_tag = random.choice(bad_aspect_list)

            for _ in range(self.neg_sample_num):
                user.append(x["user_id"])
                item.append(x["video_id"])
                rating.append(x["rating"])
                good_aspect_pos.append(a_good_tag)
                bad_aspect_pos.append(a_bad_tag)

                neg_good = np.random.randint(self.tag_num)
                neg_bad = np.random.randint(self.tag_num)
                while neg_good in good_aspect_list:
                    neg_good = np.random.randint(self.tag_num)
                good_aspect_neg.append(neg_good)
                while neg_bad in bad_aspect_list:
                    neg_bad = np.random.randint(self.tag_num)
                bad_aspect_neg.append(neg_bad)
        return (user, item, rating, good_aspect_pos, good_aspect_neg, bad_aspect_pos, bad_aspect_neg)

    def batchify_parallel(self, max_workers=int):
        workload = int(math.ceil(len(self.data) / max_workers))

        (
        total_user,
        total_item,
        total_rating,
        total_good_aspect_pos,
        total_good_aspect_neg,
        total_bad_aspect_pos,
        total_bad_aspect_neg,
        ) = ([], [], [], [], [], [], [])
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
                    (user, item, rating, good_aspect_pos,
                    good_aspect_neg, bad_aspect_pos, bad_aspect_neg) = future.result()
                    total_user.extend(user)
                    total_item.extend(item)
                    total_rating.extend(rating)
                    total_good_aspect_pos.extend(good_aspect_pos)
                    total_good_aspect_neg.extend(good_aspect_neg)
                    total_bad_aspect_pos.extend(bad_aspect_pos)
                    total_bad_aspect_neg.extend(bad_aspect_neg)
                except Exception as e:
                    logging.info(f"Caught Exception in Parallelization: {e.__class__}")
        self.user = torch.tensor(total_user, dtype=torch.int64)
        self.item = torch.tensor(total_item, dtype=torch.int64)
        self.rating = torch.tensor(total_rating, dtype=torch.float)
        self.good_aspect_pos = torch.tensor(total_good_aspect_pos, dtype=torch.int64)
        self.good_aspect_neg = torch.tensor(total_good_aspect_neg, dtype=torch.int64)
        self.bad_aspect_pos = torch.tensor(total_bad_aspect_pos, dtype=torch.int64)
        self.bad_aspect_neg = torch.tensor(total_bad_aspect_neg, dtype=torch.int64)



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
        good_aspect_pos = self.good_aspect_pos[index]
        good_aspect_neg = self.good_aspect_neg[index]
        bad_aspect_pos = self.bad_aspect_pos[index]
        bad_aspect_neg = self.bad_aspect_neg[index]
        return (
            user,
            item,
            rating,
            good_aspect_pos,
            good_aspect_neg,
            bad_aspect_pos,
            bad_aspect_neg,
        )

    def neg_tag_sampling(self):
        return

