"""
From
https://raw.githubusercontent.com/sayakpaul/Barlow-Twins-TF/main/lr_scheduler.py

References:
	* https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2
"""

# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
from keras.optimizers.schedules import LearningRateSchedule
from math import pi, cos

class WarmupCosineDecay(LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(self, learning_rate_base:float, warmup_learning_rate:float, warmup_steps:int, total_steps:int):
        super().__init__()

        assert total_steps >= warmup_steps > 0, "Total_steps must be larger or equal to warmup_steps."
        assert learning_rate_base >= warmup_learning_rate, "Learning_rate_base must be larger or equal to warmup_learning_rate."
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        # self.pi = tf.constant(np.pi)
        self.slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps

    def __call__(self, step):
        learning_rate = self.learning_rate_base * (1+cos(pi*(step-self.warmup_steps)/(self.total_steps-self.warmup_steps)))
        warmup_rate = self.slope * step + self.warmup_learning_rate

        if step > self.total_steps:
            return 0
        elif step < self.warmup_steps:
            return warmup_rate
        else:
            return learning_rate


__all__ = ['WarmupCosineDecay']