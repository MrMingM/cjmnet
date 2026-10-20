"""Independent scalar implementation of the inspected OpenCOOD pose convention."""
from __future__ import annotations

import math


def pose_matrix(pose):
    if len(pose) != 6 or not all(math.isfinite(float(v)) for v in pose):
        raise ValueError("Pose must be finite [x,y,z,roll,yaw,pitch], degrees")
    x, y, z, roll, yaw, pitch = map(float, pose)
    r, a, p = map(math.radians, (roll, yaw, pitch))
    cr, sr, cy, sy, cp, sp = math.cos(r), math.sin(r), math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    # Rz(yaw) @ Ry(-pitch) @ Rx(-roll), matching transformation_utils.py.
    return [[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr, x],
            [sy*cp, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr, y],
            [sp, -cp*sr, cp*cr, z], [0., 0., 0., 1.]]


def transform(point, matrix):
    return [sum(matrix[i][j]*point[j] for j in range(3)) + matrix[i][3] for i in range(3)]


def inverse(matrix):
    rotation = [[matrix[j][i] for j in range(3)] for i in range(3)]
    return [rotation[i] + [-sum(rotation[i][j]*matrix[j][3] for j in range(3))]
            for i in range(3)] + [[0., 0., 0., 1.]]


def object_matrix(obj):
    # Deliberately inherit OpenCOOD's location + center convention.
    extent = list(map(float, obj["extent"]))
    if len(extent) != 3 or min(extent) <= 0 or not all(map(math.isfinite, extent)):
        raise ValueError("GT extent must contain three positive half sizes")
    return pose_matrix([float(a)+float(b) for a, b in zip(obj["location"], obj["center"])]
                       + list(obj["angle"]))


def box_corners(obj):
    matrix = object_matrix(obj)
    e = obj["extent"]
    return [transform([sx*e[0], sy*e[1], sz*e[2]], matrix)
            for sx, sy, sz in ((1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,-1),
                              (1,-1,1),(1,1,1),(-1,1,1),(-1,-1,1))]


def in_box(point_world, obj, tolerance=1e-7):
    p = transform(point_world, inverse(object_matrix(obj)))
    return all(abs(p[i]) <= float(obj["extent"][i])+tolerance for i in range(3))


def numerical_audit():
    poses = [[0,0,0,0,0,0], [13,-7,2,11,93,-17], [-40,25,4,-9,-138,8]]
    points = [[0,0,0], [1,2,3], [-40,10,8]]
    error = 0.
    cross = 0.
    for pose in poses:
        m = pose_matrix(pose)
        for point in points:
            recovered = transform(transform(point, m), inverse(m))
            error = max(error, *(abs(a-b) for a,b in zip(point,recovered)))
        for world in points:
            a = transform(transform(world, inverse(m)), m)
            b = transform(transform(world, inverse(pose_matrix(poses[1]))), pose_matrix(poses[1]))
            cross = max(cross, *(abs(x-y) for x,y in zip(a,b)))
    yaw = transform([1,0,0], pose_matrix([0,0,0,0,90,0]))
    if error > 1e-10 or cross > 1e-10 or max(abs(a-b) for a,b in zip(yaw,[0,1,0])) > 1e-12:
        raise ValueError("Coordinate numerical gate failed")
    return dict(roundtrip_max_m=error, cross_cav_max_m=cross, yaw90_x_axis=yaw,
                rotation="Rz(yaw) Ry(-pitch) Rx(-roll)", angles="degrees",
                real_data_status="pending_server_audit")
