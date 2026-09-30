"""Immutable detection-set cache and OPV2V metric-aligned supervision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gspr_evidence.stage3_trace import polygon_ious


WEATHERS = ('clean', 'fog', 'rain', 'snow')
IOU_LEVELS = (.3, .5, .7)


def make_labels(corners, gt_corners, scores, threshold=.7):
    """One-to-one greedy labels using rotated BEV IoU, not nuScenes distance.

    Sorting by current refined confidence follows the paper's score-dependent
    greedy assignment. A candidate matched to an already claimed GT is FP.
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 8, 3)
    gt_corners = np.asarray(gt_corners, dtype=np.float32).reshape(-1, 8, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(corners) != len(scores):
        raise ValueError('candidate scores and boxes have different lengths')
    labels = np.zeros(len(scores), dtype=np.float32)
    if not len(scores) or not len(gt_corners):
        return labels
    ious = polygon_ious(corners, gt_corners)
    used = np.zeros(len(gt_corners), dtype=bool)
    for index in np.argsort(-scores, kind='stable'):
        values = ious[index].copy()
        values[used] = -1.
        target = int(np.argmax(values))
        if float(values[target]) >= threshold:
            labels[index] = 1.
            used[target] = True
    return labels


def _as_array(file, name, shape_tail):
    result = np.asarray(file[name], dtype=np.float32)
    if result.shape[1:] != shape_tail:
        raise ValueError(f'cache {name}: unexpected shape {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError(f'cache {name} contains non-finite values')
    return result


def read_frame(path):
    with np.load(path, allow_pickle=False) as packed:
        boxes = _as_array(packed, 'boxes', (7,))
        scores = np.asarray(packed['scores'], dtype=np.float32).reshape(-1)
        corners = _as_array(packed, 'corners', (8, 3))
        gt = _as_array(packed, 'gt', (8, 3))
        original_corners = _as_array(packed, 'original_corners', (8, 3))
        original_scores = np.asarray(
            packed['original_scores'], dtype=np.float32).reshape(-1)
        ids = np.asarray(packed['candidate_ids'], dtype=np.int64).reshape(-1)
        nms = float(packed['nms_threshold'])
        sample_index = int(packed['sample_index'])
    if (len(boxes) != len(scores) or len(boxes) != len(corners)
            or len(boxes) != len(ids) or len(boxes) > 256
            or len(ids) != len(np.unique(ids))):
        raise ValueError(f'invalid top256 frame cache: {path}')
    if len(original_corners) != len(original_scores):
        raise ValueError(f'invalid original predictions: {path}')
    if (not np.isfinite(scores).all()
            or not np.isfinite(original_scores).all()
            or not 0 < nms < 1):
        raise ValueError(f'invalid scores/NMS threshold: {path}')
    return dict(boxes=boxes, scores=scores, corners=corners, gt=gt,
                original_corners=original_corners,
                original_scores=original_scores,
                candidate_ids=ids, nms_threshold=nms,
                sample_index=sample_index, original_budget=len(original_scores))


class FrameCache(Dataset):
    def __init__(self, root, split, weathers=WEATHERS):
        if split not in ('train', 'validate'):
            raise ValueError('split must be train or validate')
        root = Path(root)
        metadata = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if metadata.get('status') != 'complete':
            raise RuntimeError('Only a complete detector extraction can be used')
        self.metadata = metadata
        self.frames = []
        for weather in weathers:
            if weather not in WEATHERS:
                raise ValueError(f'unsupported weather: {weather}')
            folder = root / split / weather
            entries = metadata['conditions'][split][weather]
            for entry in entries:
                path = folder / entry['file']
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.frames.append((weather, path))
        if not self.frames:
            raise ValueError('extraction has no frames')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        weather, path = self.frames[index]
        return weather, read_frame(path)


def collate_sets(records):
    """Pad detection sets; valid proposals only participate in BCE/attention."""
    batch = len(records)
    limit = max(1, max(len(frame['scores']) for _, frame in records))
    boxes = torch.zeros(batch, limit, 7, dtype=torch.float32)
    scores = torch.zeros(batch, limit, dtype=torch.float32)
    mask = torch.zeros(batch, limit, dtype=torch.bool)
    for index, (_, frame) in enumerate(records):
        count = len(frame['scores'])
        boxes[index, :count] = torch.from_numpy(frame['boxes'])
        scores[index, :count] = torch.from_numpy(frame['scores'])
        mask[index, :count] = True
    return boxes, scores, mask, records
