# -*- coding: utf-8 -*-
# @Time   : 2023/02/06
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import torch
import math
import random
import numpy as np
import copy

senti_to_label = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "unmentioned": 2,
}


class Batchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        (
            user,
            item,
            rating,
            aspect,
            aspect_label
        ) = ([], [], [], [], [])
        neg_sample_num = config["neg_sample_num"]

        for x in data:

            #  positive tag
            user.append(x["user_id"])
            item.append(x["item_id"])
            rating.append(x["rating"])

            # aspects with positive sentiments
            pos_aspect_list = eval(x["aspect_pos_tag"])
            # aspects with negative sentiments
            neg_aspect_list = eval(x["aspect_neg_tag"])

            if len(pos_aspect_list) != 0:
                a_pos_aspect_tag = random.choice(pos_aspect_list)
                aspect.append(a_pos_aspect_tag)
                aspect_label.append(senti_to_label["positive"])

            else:
                neg_sample_num +=1

            if len(neg_aspect_list) != 0:

                a_neg_aspect_tag = random.choice(neg_aspect_list)
                aspect.append(a_neg_aspect_tag)
                aspect_label.append(senti_to_label["negative"])
            else:
                neg_sample_num +=1

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
                aspect_label.append(senti_to_label["unmentioned"])


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

        return (
            user,
            item,
            rating,
            aspect,
            aspect_label
        )


class TagTestBatchify:
    def __init__(self, data, config, tag_num, shuffle=False):
        self.tag_num = tag_num
        (
            user,
            item,
            rating,
            pos_aspect,
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
            pos_aspect.append(eval(x["aspect_tag"]))
            candi_aspect.append(
                self.get_candidate_tags(eval(x["aspect_tag"]), config["candidate_num"])
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

        # Positive_tag acquisition, used to calculate sorting indicators, only used in the test phase
        # TODO: trainer evaluate_...() function needs to be revised.
        self.positive_aspect_tag = pos_aspect

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

    def get_candidate_tags(self, pos_tag_list: list, neg_tag_list: list, candidate_num: int):
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



class NegSamplingBatchify:
    r"""
    The function of negative sampling is provided for the label prediction task, and it is only used in the
    training phase.
    """

    def __init__(self, data, config, tag_num, shuffle=False):
        (
            user,
            item,
            rating,
            aspect_pos,
            aspect_neg,
            video_pos,
            video_neg,
            interest_pos,
            interest_neg,
        ) = ([], [], [], [], [], [], [], [], [])

        for x in data:
            pos_aspect_list = eval(x["aspect_tag"])
            pos_video_list = eval(x["video_tag"])
            pos_interest_list = eval(x["interest_tag"])

            a_pos_aspect_tag = random.choice(pos_aspect_list)
            a_pos_video_tag = random.choice(pos_video_list)
            a_pos_interest_tag = random.choice(pos_interest_list)

            for _ in range(config["neg_sample_num"]):
                user.append(x["user_id"])
                item.append(x["item_id"])
                rating.append(x["rating"])
                aspect_pos.append(a_pos_aspect_tag)
                video_pos.append(a_pos_video_tag)
                interest_pos.append(a_pos_interest_tag)

                neg_ra = np.random.randint(tag_num)
                neg_vi = np.random.randint(tag_num)
                neg_in = np.random.randint(tag_num)
                while neg_ra in pos_aspect_list:
                    neg_ra = np.random.randint(tag_num)
                aspect_neg.append(neg_ra)
                while neg_vi in pos_video_list:
                    neg_vi = np.random.randint(tag_num)
                video_neg.append(neg_vi)
                while neg_in in pos_interest_list:
                    neg_in = np.random.randint(tag_num)
                interest_neg.append(neg_in)

        self.user = torch.tensor(user, dtype=torch.int64)
        self.item = torch.tensor(item, dtype=torch.int64)
        self.rating = torch.tensor(rating, dtype=torch.float)
        self.aspect_pos = torch.tensor(aspect_pos, dtype=torch.int64)
        self.aspect_neg = torch.tensor(aspect_neg, dtype=torch.int64)
        self.video_pos = torch.tensor(video_pos, dtype=torch.int64)
        self.video_neg = torch.tensor(video_neg, dtype=torch.int64)
        self.interest_pos = torch.tensor(interest_pos, dtype=torch.int64)
        self.interest_neg = torch.tensor(interest_neg, dtype=torch.int64)

        self.shuffle = shuffle
        self.batch_size = config["batch_size"]
        self.sample_num = len(user)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

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
        aspect_pos = self.aspect_pos[index]
        aspect_neg = self.aspect_neg[index]
        video_pos = self.aspect_pos[index]
        video_neg = self.aspect_neg[index]
        interest_pos = self.aspect_pos[index]
        interest_neg = self.aspect_neg[index]

        return (
            user,
            item,
            rating,
            aspect_pos,
            aspect_neg,
            video_pos,
            video_neg,
            interest_pos,
            interest_neg,
        )

    def neg_tag_sampling(self):
        return


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


class ReviewBatchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos = word2idx["<bos>"]
        eos = word2idx["<eos>"]
        pad = word2idx["<pad>"]
        u, i, r, t = [], [], [], []
        for x in data:
            u.append(x["user_id"])
            i.append(x["item_id"])
            r.append(x["rating"])
            t.append(sentence_format(x["review"], seq_len, pad, bos, eos))

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        if self.shuffle:
            random.shuffle(self.index_list)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

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
        seq = self.seq[index]  # (batch_size, seq_len)
        return user, item, rating, seq
