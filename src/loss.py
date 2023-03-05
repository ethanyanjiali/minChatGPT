import torch
from torch import nn
import torch.nn.functional as F
import math


class CrossEntropyLoss(nn.Module):

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        y_hat: (B, T, vocab_size)
        y: (B, T)
        """
        # Convert y_hat to (B*T, vocab_size), y to (B*T)
        return F.cross_entropy(y_hat.view(-1, y_hat.size(-1)),
                               y.view(-1),
                               ignore_index=-1)


class KPairwiseLoss(nn.Module):

    def forward(self, scores: torch.Tensor):
        """
        scores: shape of (B, C) where C is number of completions ranked in order
        """
        # Consider scores as [[0.8, 0.7, 0.6]]
        B, C = scores.size()
        # scores = [[[0.8], [0.7], [0.6]]]
        scores = scores[:, :, None]    # (B, C, 1)
        # subtrahend = [[[0.8, 0.8, 0.8],
        #                [0.7, 0.7, 0.7],
        #                [0.6, 0.6, 0.6]]]
        subtrahend = scores.tile((1, C))    # (B, C, C)
        # minuend = [[[0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6]]]
        minuend = subtrahend.transpose(2, 1)    # (B, C, C)
        # diff = [[[0,                 0,                 0],
        #          [log(sigmoid(0.1)), 0,                 0],
        #          [log(sigmoid(0.2)), log(sigmoid(0.1)), 0]]]
        log_odds = torch.tril(torch.log(torch.sigmoid(minuend - subtrahend)),
                              -1)    # (B, C, C)
        total_comparision = math.comb(C, 2)
        expectation = torch.sum(log_odds, dim=(1, 2)) / total_comparision
        loss = -(1 / total_comparision) * expectation.mean()
        return loss
