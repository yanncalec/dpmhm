"""Spectrogram Augmentation.

https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py
"""
# import librosa
# import librosa.display
import tensorflow as tf
# from tensorflow_addons.image import sparse_image_warp
import numpy as np
# import matplotlib.pyplot as plt
from skimage import transform, filters
import random

"""
import matplotlib.patches as patches

Y, hh, ww = dpmhm.datasets.spec_augment.random_crop(X, (64,64), area_ratio=(0.01,1.), aspect_ratio=(1/2,2), channel_axis=0)
print(hh, ww)

plt.figure(); ax=plt.gca()
ax.imshow(X[0])
rect = patches.Rectangle((ww[0], hh[0]), ww[1], hh[1], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.figure()
plt.imshow(Y[0])
"""


def random_crop(X:np.ndarray, output_shape:tuple, *, area_ratio:tuple=(0.01, 1.), aspect_ratio:tuple=(3/4, 4/3), channel_axis=None, max_attempts=1000, seed:int=None, **kwargs):
    """Randomly crop an image to small patch.

    See also:
    https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box

    Arguments
    ---------
    aspect_ratio: tuple
    """
    # # sanity check
    # assert (3 >= X.ndim >= 2) and (X.ndim == len(output_shape))
    # assert (X.ndim != 3) or (X.shape[0] == output_shape[0])  # channel-first
    # # assert X.ndim == len(output_shape) == 3 and X.shape[0] == output_shape[0]
    random.seed(seed)

    x_area = X.shape[-1]*X.shape[-2]
    valid_crop = False
    r = output_shape[0]/output_shape[1]
    # output_aspect_ratio = (min(r, 1/r), max(r, 1/r))
    sr_min, sr_max = min(aspect_ratio)*min(r, 1/r), max(aspect_ratio)*max(r, 1/r)

    # while not valid_crop:
    for _ in range(max_attempts):
        hp = random.randint(0, X.shape[-2]-output_shape[1])
        wp = random.randint(0, X.shape[-1]-output_shape[0])

        if area_ratio is None:
            dh, dw = output_shape
            valid_crop = True
            valid_crop = (sr_min<=(dh/dw)<=sr_max) and (hp+dh<X.shape[-2]) and (wp+dw<X.shape[-1])
        else:
            ar_min, ar_max = min(area_ratio), max(area_ratio)
            dh = random.randint(0, X.shape[-2]-hp)
            dw = random.randint(0, X.shape[-1]-wp)
            valid_crop = (ar_min<=(dh*dw/x_area)<=ar_max) and (sr_min<=(dh/dw)<=sr_max) # and (hp+dh<X.shape[-2]) and (wp+dw<X.shape[-1])

        if valid_crop:
            # print('OK')
            sl = [slice(None)]*X.ndim
            sl[-1] = slice(wp, wp+dw)
            sl[-2] = slice(hp, hp+dh)
            # patch = X[:,hp:(hp+dh), wp:(wp+dw)]
            patch = X[tuple(sl)]
            # return transform.resize(patch, output_shape, **kwargs), (hp, dh), (wp, dw)  # output_shape must have the same dimension as X
            return transform.resize_local_mean(patch, output_shape, channel_axis=channel_axis), (hp, dh), (wp, dw)
            # break
    else:
        return X


# def blur(X:np.ndarray):
# 	Y = filters.gaussian(X, sigma=2, channel_axis=0)

def fade(X:np.ndarray, ratio:tuple=(1, 0.5), axis=-1):
    """Fade.
    """
    f = np.linspace(*ratio, X.shape[axis])
    rs = [1]*X.ndim; rs[axis]=-1
    return X*f.reshape(rs)


__all__ = ['random_crop', 'fade']
