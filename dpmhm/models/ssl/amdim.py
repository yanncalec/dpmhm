""" Augmented Multiscale DIM (AMDIM).

References
----------
Bachman, P., Hjelm, R.D., Buchwalter, W., 2019. Learning Representations by Maximizing Mutual Information Across Views. https://doi.org/10.48550/arXiv.1906.00910


Code
----
https://github.com/Philip-Bachman/amdim-public.
"""

# import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dataclasses import dataclass, field

# from tensorflow.keras.losses import cosine_similarity

# import numpy as np


@dataclass
class Config:
    pass

class AMDIM(models.Model):
    pass