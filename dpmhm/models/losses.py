"""Some loss functions.
"""

import keras
from keras import losses, ops

# import tensorflow as tf
# from tensorflow import linalg
# from tensorflow.keras import models, layers, regularizers, callbacks, losses
# from tensorflow.keras.applications import resnet


"""NT-Xent

Used in e.g. SimCLR, CPC.

References
----------
1. Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A Simple Framework for Contrastive Learning of Visual Representations, in: International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 1597–1607.
2. Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pp. 1857–1865, 2016.
3. Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
4. Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733–3742, 2018.

Notes
-----
Iteration over a tensor is not allowed in graph mode:
[losses.cosine_similarity(x, Y) for x in X], X has shape `(batch, dim)`

use instead:
losses.cosine_similarity(tf.expand_dims(X,1), Y)

Test:
V1 = tf.stack([losses.cosine_similarity(x, Y) for x in X])
V2 = losses.cosine_similarity(tf.expand_dims(X,1), Y)
V1-V2 is all zero
"""

def NT_Xent(zi, zj, tau:float=0.5, axis:int=-1) -> float:
    """Normalized Temperature-scaled Cross Entropy Loss.

    We modify the original definition by excluding also the term of index `j` from the denominator. This is closer to the initial aim of NT-Xent to pull together positive pairs (between `i` and `j`) while pushing apart negative pairs (not including `j`).

    Parameters
    ----------
    zi
        Anchor samples
    zj
        Augmented samples
    tau, optional
        Temperature, by default 0.5. A small temperature implies a sharper distribution in the feature space.
    axis, optional
        Axis of feature, by default -1
    """
    # Cosine similarity
    # between anchor - anchor
    Sii = -losses.cosine_similarity(zi, zi, axis=axis) / tau
    # # or equivalently
    # zi = ops.normalize(zi, axis=axis)
    # Sii = ops.matmul(zi, ops.transpose(zi)) / tau
    #
    # between anchor - augmented
    Sij = -losses.cosine_similarity(zi, zj, axis=axis) / tau
    # # or equivalently
    # zj = ops.normalize(zj, axis=axis)
    # Sij = ops.matmul(zi, ops.transpose(zj)) / tau

    P = ops.diag(Sij)
    N = Sii - ops.diag(Sii) + Sij - ops.diag(Sij)

    return ops.sum(ops.logsumexp(N, axis=axis) - P)
