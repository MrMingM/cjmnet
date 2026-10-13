"""Fixed-face signed residual and analytic Jacobian; matrices are geometric surrogates."""
import numpy as np
from .coordinates import local_xy

def faces(points, theta):
    q = local_xy(points, theta)
    axis = np.argmin(np.abs(np.abs(q) - np.asarray(theta)[2:4] / 2), axis=1)
    sign = np.where(q[np.arange(len(q)), axis] >= 0, 1., -1.)
    return axis, sign

def residual(points, theta, assignment):
    axis, sign = assignment
    q = local_xy(points, theta)
    return q[np.arange(len(q)), axis] - sign * np.asarray(theta)[2 + axis] / 2

def jacobian(points, theta, assignment=None):
    axis, sign = faces(points, theta) if assignment is None else assignment
    q = local_xy(points, theta)
    c, s = np.cos(theta[4]), np.sin(theta[4])
    j = np.zeros((len(q), 5))
    a = axis == 0
    j[a, 0], j[a, 1], j[a, 2], j[a, 4] = -c, -s, -sign[a]/2, q[a, 1]
    j[~a, 0], j[~a, 1], j[~a, 3], j[~a, 4] = s, -c, -sign[~a]/2, -q[~a, 0]
    return j

def finite_jacobian(points, theta, steps):
    fixed = faces(points, theta)
    out = []
    for k, h in enumerate(steps):
        d = np.zeros(5); d[k] = h
        out.append((residual(points, np.asarray(theta)+d, fixed) - residual(points, np.asarray(theta)-d, fixed)) / (2*h))
    return np.array(out).T

def metrics(matrix, count, ridge, rank_tol):
    m = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    if m.shape != (5, 5) or not np.isfinite(m).all():
        raise ValueError('Invalid information matrix')
    eig, vec = np.linalg.eigh(m)
    tol = max(float(eig[-1]), 1.) * rank_tol
    if eig[0] < -tol:
        raise ValueError('Information matrix not PSD')
    eig = np.maximum(eig, 0.)
    eig[eig <= tol] = 0.
    rank = int(np.sum(eig > 0))
    trace = float(np.trace(m))
    for k in range(5):
        v = vec[:, k]
        if v[np.argmax(np.abs(v))] < 0:
            vec[:, k] *= -1
    usable = count >= 2 and rank > 0
    return {'valid': bool(count >= 2 and rank == 5), 'directional_valid': bool(usable),
        'matrix_valid': bool(count > 0), 'rank': rank, 'rank_deficient': rank < 5,
        'diagonal': np.diag(m), **{'info_'+p:float(m[k,k]) for k,p in enumerate(('x','y','l','w','yaw'))},
        'eigenvalues': eig, 'weakest_eigenvector': vec[:, 0],
        'strongest_eigenvector': vec[:, -1],
        'weakest_eigenvector_valid': bool(usable and eig[1]-eig[0] > tol),
        'strongest_eigenvector_valid': bool(usable and eig[-1]-eig[-2] > tol),
        'min_eigenvalue': float(eig[0]), 'max_eigenvalue': float(eig[-1]),
        'condition_number': float(eig[-1]/eig[0]) if rank == 5 else None,
        'condition_valid': rank == 5, 'trace': trace,
        'logdet': float(np.log(eig).sum()) if rank == 5 else None,
        'logdet_valid': rank == 5,
        'logdet_regularized': float(np.log(eig + ridge).sum()),
        'anisotropy': float(np.linalg.norm(m / trace - np.eye(5)/5)) if trace > 0 else None}

def information(points, theta, weights, cfg):
    p = np.asarray(points, float).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite ROI points')
    j = jacobian(p, theta)
    w = np.asarray(weights, float)
    if w.shape != (len(p),) or not np.isfinite(w).all():
        raise ValueError('Weights do not correspond to points')
    w = np.clip(w, cfg['weight_eps'], 1.)
    scale = np.asarray(cfg['parameter_scales'])
    out = {'count': len(p), 'face_counts': np.bincount(faces(p, theta)[0], minlength=2)}
    for name, ww in (('geo', np.ones(len(p))), ('gspr', w)):
        raw = j.T @ (j * ww[:, None])
        norm = raw * scale[:, None] * scale[None, :]
        trace = float(np.trace(norm))
        matrices = {'raw': norm, 'count_normalized': norm/max(len(p), 1),
                    'shape': norm/trace if trace > 0 else np.zeros((5,5))}
        out[name] = {'I_physical': raw, 'I_physical_per_count': raw/max(len(p), 1),
                     'I_physical_per_trace': raw/np.trace(raw) if np.trace(raw)>0 else np.zeros((5,5)),
                     'variants': {k: {'matrix': v, **metrics(v, len(p), cfg['matrix_ridge'], cfg['rank_relative_tolerance'])}
                                  for k, v in matrices.items()}}
    return out

def complementarity(a, b, ridge):
    a, b = np.asarray(a), np.asarray(b)
    def ld(m):
        return float(np.linalg.slogdet(m + ridge*np.eye(5))[1])
    def small(m):
        return max(0., float(np.linalg.eigvalsh((m+m.T)/2)[0]))
    _, va = np.linalg.eigh(a); _, vb = np.linalg.eigh(b)
    ea, eb = np.linalg.eigvalsh(a), np.linalg.eigvalsh(b)
    weak_unique = bool(ea[1]-ea[0]>1e-8*max(1,ea[-1]) and eb[1]-eb[0]>1e-8*max(1,eb[-1]))
    return {'C_logdet': ld(a+b)-max(ld(a),ld(b)),
            'C_weak': small(a+b)-max(small(a),small(b)),
            'weak_alignment': float(abs(va[:,0]@vb[:,0])) if weak_unique else None,
            'weak_alignment_valid': weak_unique}
