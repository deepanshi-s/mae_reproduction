import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

def separate_weight_decay_weights(model: nn.Module):
    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            weight_decay_params.append(param)
        else:
            non_weight_decay_params.append(param)
    return weight_decay_params, non_weight_decay_params


class lrScheduler(LRScheduler):
    """Linear Warmp up plus cosine decay learning rate scheduler

    Args:
        optimizer: optimizer
    """

    def __init__(self, optimizer, peak_warmup_itr, min_lr, total_itr):
        self.peak_warmup_itr = peak_warmup_itr
        self.min_lr = min_lr
        self.total_itr = total_itr
        super().__init__(optimizer, -1)

    def get_lr(self) -> float:
        if self.last_epoch < self.peak_warmup_itr:
            return [
                base_lr * self.last_epoch / self.peak_warmup_itr
                for base_lr in self.base_lrs
            ]

        decay_val = math.cos(
            math.pi
            * (self.last_epoch - self.peak_warmup_itr)
            / (self.total_itr - self.peak_warmup_itr)
        )
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + decay_val)
            for base_lr in self.base_lrs
        ]
