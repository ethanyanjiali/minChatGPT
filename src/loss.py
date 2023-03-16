import torch
from torch import nn
import torch.nn.functional as F
import math


class PolicyLoss(nn.Module):
    """
    Proximal Policy Optimization Algorithms
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, eps=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, new_actor_log_probs: torch.Tensor,
                old_actor_log_probs: torch.Tensor, advantage: torch.Tensor, action_mask: torch.Tensor):
        # reverse the log to get π_new(a_t|s_t) / π_old(a_t|s_t)
        ratio = (new_actor_log_probs -
                 old_actor_log_probs).exp()  # (B, num_actions)
        surrogate_objectives = torch.min(
            ratio * advantage,
            ratio.clamp(1 - self.eps, 1 + self.eps) *
            advantage)  # (B, num_actions)
        # minimize the negative loss -> maximize the objective
        loss = -surrogate_objectives  # (B, num_actions)
        return loss.mean()


class ValueLoss(nn.Module):

    def __init__(self, eps=0.4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, values: torch.Tensor, reward: torch.Tensor, old_values: torch.Tensor, action_mask: torch.Tensor):
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L69-L75
        # https://github.com/openai/baselines/issues/91
        values_clipped = old_values + (values - old_values).clamp(-self.eps, self.eps)
        surrogate_values = torch.max(torch.square(values - reward), torch.square(values_clipped - reward))
        return surrogate_values.mean()  # (B, 1)


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
        # print(scores.shape)
        B, C = scores.size()
        # scores = [[[0.8], [0.7], [0.6]]]
        scores = scores[:, :, None]  # (B, C, 1)
        # subtrahend = [[[0.8, 0.8, 0.8],
        #                [0.7, 0.7, 0.7],
        #                [0.6, 0.6, 0.6]]]
        subtrahend = scores.tile((1, C))  # (B, C, C)
        # minuend = [[[0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6],
        #             [0.8, 0.7, 0.6]]]
        minuend = subtrahend.transpose(2, 1)  # (B, C, C)
        # diff = [[[0,                 0,                 0],
        #          [log(sigmoid(0.1)), 0,                 0],
        #          [log(sigmoid(0.2)), log(sigmoid(0.1)), 0]]]
        log_odds = torch.tril(torch.log(torch.sigmoid(minuend - subtrahend)),
                              -1)  # (B, C, C)
        total_comparision = math.comb(C, 2)
        expectation = torch.sum(log_odds, dim=(1, 2)) / total_comparision
        loss = -(1 / total_comparision) * expectation.mean()
        return loss
