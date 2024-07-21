"""Methods for spectrogram & waveform augmentation.

See:
- https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py
- https://github.com/facebookresearch/WavAugment
"""
# import librosa
# import librosa.display
# import tensorflow as tf
# from tensorflow_addons.image import sparse_image_warp
# import matplotlib.pyplot as plt

import numpy as np
from skimage import transform #, filters
import random

"""
import matplotlib.patches as patches

Y, hh, ww = dpmhm.datasets.augment.random_crop(X, (64,64), area_ratio=(0.01,1.), aspect_ratio=(1/2,2), channel_axis=0)
print(hh, ww)

plt.figure(); ax=plt.gca()
ax.imshow(X[0])
rect = patches.Rectangle((ww[0], hh[0]), ww[1], hh[1], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.figure()
plt.imshow(Y[0])
"""

def randomly(p:float):
    def wrapper(func):
        def _func(X, *args, **kwargs):
            if random.random() <= p:
                return func(X, *args, **kwargs)
            else:
                return X
        return _func
    return wrapper


# def random_flip(X:np.ndarray, p:float, axis:int=-1) -> np.ndarray:
#     """Randomly flip an array along the given axis.
#     """
#     if random.random() < p:
#         return np.flip(X, axis=axis)
#     else:
#         return X


    """Randomly crop an image.

    See also:
    https://www.tensorflow.org/api_docs/python/tf/image/sample_distorted_bounding_box
    """
def random_crop(X:np.ndarray, output_shape:tuple, *, prob:float=0., area_ratio:tuple=(0.1, 1.), aspect_ratio:tuple=(3/4, 4/3), channel_axis:int=None, max_attempts:int=100, seed:int=None, **kwargs):
    """Randomly crop an image.

    Parameters
    ----------
    X
        Input image in channel first format.
    output_shape
        Desired output shape.
    prob
        probability to randomly crop a fixed patch of output shape from `X`, with all other parameters being ignored.
    area_ratio, optional
        range of the area ratio of crop to the full image, by default (0.01, 1.)
    aspect_ratio, optional
        range of the aspect ratio of the shape of the crop, by default (3/4, 4/3)
    max_attempts, optional
        maximum number of attemps, by default 1000
    """
    # # sanity check
    # assert (3 >= X.ndim >= 2) and (X.ndim == len(output_shape))
    # assert (X.ndim != 3) or (X.shape[0] == output_shape[0])  # channel-first
    # # assert X.ndim == len(output_shape) == 3 and X.shape[0] == output_shape[0]
    random.seed(seed)

    if random.random() <= prob:
        dh, dw = output_shape
        # assert all([d1>=d2 for d1, d2 in zip(X.shape, output_shape)])
        assert X.shape[-1] >= dh and X.shape[-2] >= dw
        hp = random.randint(0, X.shape[-2]-dw)
        wp = random.randint(0, X.shape[-1]-dh)
        sl = [slice(None)] * X.ndim
        sl[-1] = slice(wp, wp+dw)
        sl[-2] = slice(hp, hp+dh)
        return X[tuple(sl)], (hp, dh), (wp, dw)
    else:
        x_area = X.shape[-1]*X.shape[-2]
        valid_crop = False
        r = output_shape[0]/output_shape[1]
        # output_aspect_ratio = (min(r, 1/r), max(r, 1/r))
        sr_min, sr_max = min(aspect_ratio)*min(r, 1/r), max(aspect_ratio)*max(r, 1/r)
        ar_min, ar_max = min(area_ratio), max(area_ratio)

        for _ in range(max_attempts):
            hp = random.randint(0, X.shape[-2]-1)
            wp = random.randint(0, X.shape[-1]-1)
            # hp = random.randint(0, X.shape[-2]-output_shape[1])
            # wp = random.randint(0, X.shape[-1]-output_shape[0])
            dh = random.randint(1, X.shape[-2]-hp)
            dw = random.randint(1, X.shape[-1]-wp)

            cr_min, cr_max = min(dh/dw, dw/dh), max(dh/dw, dw/dh)
            valid_crop = (ar_min<=(dh*dw/x_area)<=ar_max) and (sr_min<=cr_min) and (cr_max<=sr_max)

            if valid_crop:
                # print('OK')
                sl = [slice(None)] * X.ndim
                sl[-1] = slice(wp, wp+dw)
                sl[-2] = slice(hp, hp+dh)
                # patch = X[:,hp:(hp+dh), wp:(wp+dw)]
                patch, ht, wt = X[tuple(sl)], (hp, dh), (wp, dw)
                break
        else:
            patch, ht, wt = X, None, None

        # return transform.resize(patch, output_shape, **kwargs), ht, wt  # output_shape must have the same dimension as X
        return transform.resize_local_mean(patch, output_shape, channel_axis=channel_axis), ht, wt


def fade(X:np.ndarray, ratio:float=0.5, axis:int=-1) -> np.ndarray:
    f = np.linspace(1., ratio, X.shape[axis])
    rs = [1] * X.ndim; rs[axis]=-1
    return X * f.reshape(rs)


# def blur(X:np.ndarray):
# 	Y = filters.gaussian(X, sigma=2, channel_axis=0)

__all__ = ['randomly', 'random_crop', 'fade']


