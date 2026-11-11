"""Validate warned scalar intersections without changing the official geometry."""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import math
import warnings

import numpy as np
import shapely


_MESSAGE = 'invalid value encountered in intersection'


@contextmanager
def checked_intersections():
    original = shapely.intersection
    counts = Counter()
    active = False

    @wraps(original)
    def checked(left, right, *args, **kwargs):
        nonlocal active
        before = counts['intersection_warnings']
        active = True
        try:
            result = original(left, right, *args, **kwargs)
        finally:
            active = False
        if counts['intersection_warnings'] != before:
            if (result is None or not result.is_valid or not math.isfinite(result.area)
                    or not np.isfinite(shapely.get_coordinates(result)).all()):
                raise FloatingPointError('Shapely warned and returned invalid intersection geometry')
            counts['validated'] += 1
            if result.is_empty:
                counts['empty_intersections'] += 1
            elif result.area == 0:
                counts['zero_area_nonempty'] += 1
            else:
                counts['positive_area'] += 1
        return result

    with warnings.catch_warnings():
        original_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if (active and category is RuntimeWarning and str(message) == _MESSAGE
                    and filename.replace('\\', '/').endswith('/shapely/set_operations.py')):
                counts['intersection_warnings'] += 1
                return
            original_show(message, category, filename, lineno, file, line)

        warnings.filterwarnings('always', message=f'^{_MESSAGE}$', category=RuntimeWarning,
                                module=r'^shapely\.set_operations$')
        warnings.showwarning = show
        shapely.intersection = checked
        try:
            yield counts
        finally:
            shapely.intersection = original


def checked_evaluation(evaluate):
    @wraps(evaluate)
    def wrapped(model, criterion, config, out, *args, **kwargs):
        with checked_intersections() as counts:
            result = evaluate(model, criterion, config, out, *args, **kwargs)
        if counts['intersection_warnings']:
            print(f'GEOMETRY WARNINGS VERIFIED {out}: {dict(counts)}', flush=True)
        return result
    return wrapped
