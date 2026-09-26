"""Deterministic frame sampling without GT, training dependencies, or torch."""
import bisect
import random


def blocked_frames(scene_ends, pilot_scenes, pilot_indices, exclusion_radius):
    ends = [int(v) for v in scene_ends]
    if not ends or any(b <= a for a, b in zip([0, *ends[:-1]], ends)):
        raise ValueError('Invalid cumulative scene lengths')
    pilot_scenes = {int(v) for v in pilot_scenes}
    if any(v < 0 or v >= len(ends) for v in pilot_scenes):
        raise ValueError('Pilot scene does not exist')
    if type(exclusion_radius) is not int or exclusion_radius < 0:
        raise ValueError('Exclusion radius must be a nonnegative integer')
    blocked = set()
    for index in pilot_indices:
        index = int(index)
        scene = bisect.bisect_right(ends, index)
        if scene not in pilot_scenes or index < 0 or scene >= len(ends):
            raise ValueError('Pilot frame does not belong to a pilot scene')
        start = ends[scene-1] if scene else 0
        blocked.update(range(max(start, index-exclusion_radius),
                             min(ends[scene]-1, index+exclusion_radius)+1))
    return blocked


def holdout_frames(scene_ends, pilot_scenes, pilot_indices, seed,
                   max_scenes, frames_per_scene, exclusion_radius):
    """Prefer unused scenes; otherwise use unused frames with an index gap."""
    ends = [int(v) for v in scene_ends]
    pilot_scenes = {int(v) for v in pilot_scenes}
    blocked = blocked_frames(ends, pilot_scenes, pilot_indices, exclusion_radius)
    available = {}
    for scene in range(len(ends)):
        start = ends[scene-1] if scene else 0
        candidates = [index for index in range(start, ends[scene]) if index not in blocked]
        if candidates:
            available[scene] = candidates
    if not available:
        raise ValueError('No pilot-frame-excluded validation frames remain')
    rng = random.Random(int(seed))
    unseen = [scene for scene in available if scene not in pilot_scenes]
    seen = [scene for scene in available if scene in pilot_scenes]
    rng.shuffle(unseen)
    rng.shuffle(seen)
    scenes = sorted((unseen+seen)[:max_scenes])
    indices = []
    for scene in scenes:
        candidates = available[scene]
        indices.extend(rng.sample(candidates,min(frames_per_scene,len(candidates))))
    indices.sort()
    if set(indices) & blocked:
        raise AssertionError('Sample includes a pilot or neighboring frame')
    if any(bisect.bisect_right(ends, i) not in scenes for i in indices):
        raise AssertionError('Sample escaped selected scenes')
    return scenes, indices
