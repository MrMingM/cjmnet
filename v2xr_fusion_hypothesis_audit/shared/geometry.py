"""Operations on common-grid candidates and wrapped errors."""
import numpy as np


def wrap_yaw(x):
    return (x+np.pi/2) % np.pi - np.pi/2


def cosine(a, b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def error(prediction, target):
    value = np.asarray(prediction, dtype=float)-target
    value[6] = wrap_yaw(value[6])
    return value
