"""Some loss functions.
"""

# import sys
import tensorflow as tf
from tensorflow import keras, linalg
# from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras import losses
# from tensorflow.keras.applications import resnet

# import numpy as np


##### NT_Xent #####
"""
Used in: SimCLR, CPC...

References
----------
1. Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A Simple Framework for Contrastive Learning of Visual Representations, in: International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 1597–1607.
2. Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pp. 1857–1865, 2016.
3. Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
4. Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733–3742, 2018.
"""

# Notes on the implementation
"""
Iteration over a tensor is not allowed in graph mode:
[losses.cosine_similarity(x, Y) for x in X], X has shape `(batch, dim)`

use instead:
losses.cosine_similarity(tf.expand_dims(X,1), Y)

Test:
V1 = tf.stack([losses.cosine_similarity(x, Y) for x in X])
V2 = losses.cosine_similarity(tf.expand_dims(X,1), Y)
V1-V2 is all zero
"""

def NT_Xent(X, Y, tau:float):
    """Normalized temperature-scaled cross entropy loss.

    Args
    ----
    X, Y: 2d tensors with the first dimension being the batch.

    This is eq. (1) of the original paper of SimCLR [1].
    """
    # Note: cosine_similarity(x,x)==-1
    B = tf.cast(tf.shape(X)[0], tf.float32)  # batch size
    # Original version:
    S = tf.reduce_sum(-losses.cosine_similarity(X, Y)) / tau
    Nx = NT_Xent_norm_vec(X, Y) / tau
    return -(S - tf.reduce_logsumexp(Nx)) / B
    # # Modified version:
    # S = tf.reduce_sum(losses.cosine_similarity(X, Y)) / tau
    # Nx = - NT_Xent_norm_vec(X, Y) / tau
    # return (S - tf.reduce_logsumexp(Nx)) / B


def NT_Xent_sym(X, Y, tau:float):
    """Normalized temperature-scaled Cross entropy loss, symmetrized version.
    """
    B = tf.cast(tf.shape(X)[0], tf.float32)  # batch size
    S = tf.reduce_sum(losses.cosine_similarity(X, Y)) / tau
    Nx = - NT_Xent_norm_vec(X, Y) / tau
    Ny = - NT_Xent_norm_vec(Y, X) / tau
    return (S - tf.reduce_logsumexp(Nx+Ny)) / B


def NT_Xent_norm_vec(X, Y, full:bool=False):
    """Compute the normalization vector of the NT-Ext loss.
    """
    if full:
        Nx = tf.reduce_sum(-losses.cosine_similarity(tf.expand_dims(X,1), Y), axis=-1)
    else:  # original version
        Sx = -losses.cosine_similarity(X[:,None,:], Y[None,:,:])  # ~(batch x, batch y)
        # Sx = -losses.cosine_similarity(tf.expand_dims(X,1), Y)  # same same
        # Nx = tf.reduce_sum(tf.linalg.set_diag(Sx,tf.zeros(tf.shape(X)[1])), axis=-1)
        Sx -= linalg.diag(linalg.diag_part(Sx))
        Nx = tf.reduce_sum(Sx, axis=-1)
    return Nx

