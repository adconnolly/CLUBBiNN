"""Module defining an early stopper for the training process."""

import numpy as np
import torch


class EarlyStopper:
    """Class to trigger early stopping if loss threshold is reached in training."""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, val_loss):
        """
        Check whether to trigger early stopping.

        Based on loss not decreasing below best yet for a certain number of steps set
        by the patience attribute.
        """
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
