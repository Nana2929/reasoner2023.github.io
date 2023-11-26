# -*- coding: utf-8 -*-
# @Time   : 2023/02/07
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os

import torch
import torch.optim as optim

from metrics.metrics import (bleu_score, evaluate_ndcg,
                             evaluate_precision_recall_f1, mean_absolute_error,
                             root_mean_square_error, rouge_score)
from utils import get_local_time, ids2tokens, now_time


class Trainer(object):
    def __init__(self, config, model, train_data, val_data, rating_train_type="mse"):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.train_type = rating_train_type

        self.model_name = config["model"]
        self.dataset = config["dataset"]
        self.epochs = config["epochs"]

        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.learner = config["learner"]
        self.learning_rate = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.95
        )  # gamma: lr_decay
        self.rating_weight = config["rating_weight"]
        self.aspect_weight = config["aspect_weight"]
        self.l2_weight = config["l2_weight"]
        self.top_k = config["top_k"]
        self.max_rating = config["max_rating"]
        self.min_rating = config["min_rating"]

        self.endure_times = config["endure_times"]
        self.checkpoint = config["checkpoint"]

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def train_mse(self, data):  # train mse+bce+l2
        self.model.train()
        loss_sum = 0.0
        Rating_loss = 0.0
        Tag_loss = 0.0
        L2_loss = 0.0
        total_sample = 0
        while True:
            user, item, rating, aspect_tag, aspect_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            aspect_tag = aspect_tag.to(self.device)

            aspect_label = aspect_label.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = (
                self.model.calculate_rating_mseloss(user, item, rating)
                * self.rating_weight
            )
            aspect_loss = (
                self.model.calculate_aspect_loss(user, item, aspect_tag, aspect_label)
                * self.aspect_weight
            )
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + aspect_loss + l2_loss
            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (aspect_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return (
            Rating_loss / total_sample,
            Tag_loss / total_sample,
            L2_loss / total_sample,
            loss_sum / total_sample,
        )

    def train_ce(self, data):  # train ce+bce+l2
        self.model.train()
        loss_sum = 0.0
        Rating_loss = 0.0
        Tag_loss = 0.0
        L2_loss = 0.0
        total_sample = 0
        while True:
            user, item, rating, aspect_tag, aspect_label = data.next_batch()
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            aspect_tag = aspect_tag.to(self.device)

            aspect_label = aspect_label.to(self.device)

            self.optimizer.zero_grad()
            rating_loss = (
                self.model.calculate_rating_celoss(user, item, rating)
                * self.rating_weight
            )
            aspect_loss = (
                self.model.calculate_aspect_loss(user, item, aspect_tag, aspect_label)
                * self.aspect_weight
            )
            l2_loss = self.model.calculate_l2_loss() * self.l2_weight
            loss = rating_loss + aspect_loss + l2_loss
            loss.backward()
            self.optimizer.step()
            Rating_loss += self.batch_size * rating_loss.item()
            Tag_loss += self.batch_size * (aspect_loss.item())
            L2_loss += self.batch_size * l2_loss.item()
            loss_sum += self.batch_size * loss.item()
            total_sample += self.batch_size

            if data.step == data.total_step:
                break
        return (
            Rating_loss / total_sample,
            Tag_loss / total_sample,
            L2_loss / total_sample,
            loss_sum / total_sample,
        )

    def valid(self, data):  # valid
        self.model.eval()
        loss_sum = 0.0
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, aspect_tag, aspect_label = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                aspect_tag = aspect_tag.to(self.device)
                aspect_label = aspect_label.to(self.device)

                mse_rating_loss = (
                    self.model.calculate_rating_mseloss(user, item, rating)
                    * self.rating_weight
                )
                ce_rating_loss = (
                    self.model.calculate_rating_celoss(user, item, rating)
                    * self.rating_weight
                )
                aspect_loss = (
                    self.model.calculate_reason_loss(
                        user, item, aspect_tag, aspect_label
                    )
                    * self.aspect_weight
                )

                l2_loss = self.model.calculate_l2_loss() * self.l2_weight
                if self.train_type == "mse":
                    loss = mse_rating_loss + aspect_loss + l2_loss
                else:
                    loss = ce_rating_loss + aspect_loss + l2_loss

                loss_sum += self.batch_size * loss.item()
                total_sample += self.batch_size

                if data.step == data.total_step:
                    break

        return loss_sum / total_sample

    def evaluate(self, model, data):  # test
        model.eval()
        rating_predict = []
        pos_aspect_predict = []  # positive sentiment tags
        neg_aspect_predict = []  # negative sentiment tags
        with torch.no_grad():
            while True:
                user, item, candi_aspect_tag = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)

                aspect_candidate_tag = candi_aspect_tag.to(self.device)

                rating_p = model.predict_rating(user, item)  # (batch_size,)
                rating_predict.extend(rating_p.tolist())

                aspect_p = model.predict_aspect_score(
                    user, item, aspect_candidate_tag
                )  # (batch_size,candidate_num)

                _, aspect_p_topk = torch.topk(
                    aspect_p, dim=-1, k=self.top_k, largest=True, sorted=True
                )  # values & index
                _, aspect_p_bottomk = torch.topk(
                    aspect_p, dim=-1, k=self.top_k, largest=False, sorted=True
                )
                pos_aspect_predict.extend(
                    aspect_candidate_tag.gather(1, aspect_p_topk).tolist()
                )
                neg_aspect_predict.extend(
                    aspect_candidate_tag.gather(1, aspect_p_bottomk).tolist()
                )

                if data.step == data.total_step:
                    break
        # rating
        # TRAINER needs to be revised
        rating_zip = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(rating_zip, self.max_rating, self.min_rating)
        MAE = mean_absolute_error(rating_zip, self.max_rating, self.min_rating)

        # aspect_tag scores
        score_dict = {"pos": None, "neg": None}
        # =================== positive tags =======================
        aspect_pr, aspect_r, aspect_f1 = evaluate_precision_recall_f1(
            self.top_k, data.positive_aspect_tag, pos_aspect_predict
        )
        aspect_ndcg = evaluate_ndcg(
            self.top_k, data.positive_aspect_tag, pos_aspect_predict
        )
        score_dict["pos"] = (aspect_pr, aspect_r, aspect_f1, aspect_ndcg)
        # =================== negative tags =======================
        aspect_pr, aspect_r, aspect_f1 = evaluate_precision_recall_f1(
            self.top_k, data.negative_aspect_tag, neg_aspect_predict
        )
        aspect_ndcg = evaluate_ndcg(
            self.top_k, data.negative_aspect_tag, neg_aspect_predict
        )
        score_dict["neg"] = (aspect_pr, aspect_r, aspect_f1, aspect_ndcg)

        return RMSE, MAE, score_dict

    def train_loop(self):  # mse or ce
        best_val_loss = float("inf")
        best_epoch = 0
        endure_count = 0
        model_path = ""
        train = self.train_mse if self.train_type == "mse" else self.train_ce

        for epoch in range(1, self.epochs + 1):
            print(now_time() + "epoch {}".format(epoch))
            train_r_loss, train_t_loss, train_l_loss, train_sum_loss = train(
                self.train_data
            )
            print(
                now_time()
                + "rating loss {:4.4f} | tag loss {:4.4f} | l2 loss {:4.4f} |total loss {:4.4f} on train".format(
                    train_r_loss, train_t_loss, train_l_loss, train_sum_loss
                )
            )
            val_loss = self.valid(self.val_data)
            print(now_time() + "total loss {:4.4f} on validation".format(val_loss))

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val_loss:
                saved_model_file = "{}-{}-{}.pt".format(
                    self.model_name, self.dataset, get_local_time()
                )
                model_path = os.path.join(self.checkpoint, saved_model_file)
                with open(model_path, "wb") as f:
                    torch.save(self.model, f)
                print(now_time() + "Save the best model" + model_path)
                best_val_loss = val_loss
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + "Endured {} time(s)".format(endure_count))
                if endure_count == self.endure_times:
                    print(
                        now_time()
                        + "Cannot endure it anymore | Exiting from early stop"
                    )
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                print(
                    now_time()
                    + "Learning rate set to {:2.8f}".format(
                        self.scheduler.get_last_lr()[0]
                    )
                )

        return model_path, best_epoch
