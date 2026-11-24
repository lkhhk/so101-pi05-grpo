"""Lossless transport of the RGB taps used by PI05's fixed bilinear resize."""
import numpy as np


CODEC = "pi05-bilinear-taps-480x640-v1"
SOURCE = (480, 640)
TARGET = (224, 224)


def _taps(source, resized):
    # Exact integer form of floor((j + 0.5) * source / resized - 0.5).
    numerator = (2*np.arange(resized, dtype=np.int64)+1)*source-resized
    lower, remainder = np.divmod(numerator, 2*resized)
    # This v1 geometry has no coordinates near an integer; CPU/CUDA float
    # rounding cannot select a different neighbor for the fixed dimensions.
    distance = np.minimum(remainder, 2*resized-remainder)/(2*resized)
    if distance.min() < .01:
        raise RuntimeError("Unsafe interpolation geometry for sparse RGB")
    return np.unique(np.clip(np.concatenate((lower,lower+1)),0,source-1))


_ROWS, _COLS = _taps(480, 168), _taps(640, 224)


def pack(image):
    if image.shape != (*SOURCE, 3) or image.dtype != np.uint8:
        raise ValueError("Sparse RGB needs an H×W×3 uint8 480×640 frame")
    return image[np.ix_(_ROWS, _COLS)]


def unpack(packed):
    if packed.shape != (len(_ROWS),len(_COLS),3) or packed.dtype != np.uint8:
        raise ValueError("Invalid sparse RGB tap image")
    out = np.zeros((*SOURCE,3),np.uint8)
    out[np.ix_(_ROWS,_COLS)] = packed
    return out
