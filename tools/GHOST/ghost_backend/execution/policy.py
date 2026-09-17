"""Shared CPU backend relative runtime prior.

The prior ranks work; it is not a wall-time or optimality guarantee. It retains
the measured dense preference on small/medium systems. Both desktop and
execution-node scheduling use it.
"""
import math

MODEL = 'geometry_work_v4'
BACKENDS = ('dense', 'compressed')


def relative_cost(resources, n_angles, mode):
    n=max(1,int(resources['nodes'])); d=max(1,int(resources['system_dofs']))
    angles=max(1,int(n_angles)); kernels=max(1.,resources.get('operator_matrices',3)/3.)
    if resources.get('analytic_zero'):
        return .001
    assembly=7e-7*n*n*kernels
    dense=.025 + assembly + 4e-12*d**3 + 2e-10*d*d*angles
    if mode == 'dense':
        return dense
    if mode == 'compressed':
        return 1.4*dense
    raise ValueError('Unknown automatic backend.')


def available_candidates(candidates):
    """Copy candidates for ranking; every backend runs on any CPU host."""
    return {mode:dict(c) for mode,c in candidates.items()}


def rank_candidates(candidates, budget_gib, margin=.2):
    available=available_candidates(candidates)
    if not available:
        raise RuntimeError('No compatible solver backend is available.')
    for mode,c in available.items():
        if mode not in BACKENDS or any(not math.isfinite(float(c[k])) or c[k] <= 0 for k in ('cost','peak_gb')):
            raise ValueError('Invalid automatic backend forecast.')
    fitting=[m for m,c in available.items() if c['peak_gb'] <= (1-margin)*budget_gib]
    if not fitting:
        fitting=[m for m,c in available.items() if c['peak_gb'] <= budget_gib]
    if not fitting:
        description=', '.join('{} {:.2f} GiB'.format(m,c['peak_gb']) for m,c in available.items())
        raise MemoryError('No compatible backend fits the {:.2f} GiB solve budget ({}).'.format(budget_gib,description))
    return sorted(fitting,key=lambda m:(available[m]['cost'],BACKENDS.index(m)))
