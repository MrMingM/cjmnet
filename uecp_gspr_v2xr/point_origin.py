"""NumPy index rules shared by real simulator-origin lookup and local synthetic checks."""
import numpy as np


def resolve_neighbors(labels, distance, index):
    labels = np.asarray(labels)
    if not len(labels):
        return np.full(len(distance), -1., np.float32)
    safe = np.minimum(index, len(labels)-1)
    found = np.isfinite(distance[:, 0])
    result = np.full(len(distance), -1., np.float32)
    result[found] = labels[safe[found, 0]]
    conflict = np.isfinite(distance[:, 1]) & (labels[safe[:, 0]] != labels[safe[:, 1]])
    result[conflict] = -1
    return result


def aggregate_labels(labels_for_valid_slots, coords, valid, height=100, width=352):
    xy = np.repeat(coords[:, 2:4], valid.shape[1], 0)[valid.flatten()]//2
    flat = xy[:, 0]*width+xy[:, 1]
    if len(flat) != len(labels_for_valid_slots) or (len(flat) and (flat.min() < 0 or flat.max() >= height*width)):
        raise ValueError('Point-slot/region coordinate identity differs')
    known = labels_for_valid_slots >= 0
    counts = np.bincount(flat[known], minlength=height*width)
    sums = np.bincount(flat[known], weights=labels_for_valid_slots[known], minlength=height*width)
    return counts.reshape(height, width), sums.reshape(height, width)
