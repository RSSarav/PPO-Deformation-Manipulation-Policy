"""Point-set distances between the current positions of an object's material samples and a goal point set
(notes/A_deformation_math.md, section 6).

Distances are in metres. :func:`ordered` compares sample k with target k (a fixed correspondence); :func:`hungarian`
finds the best one-to-one correspondence; :func:`chamfer` and :func:`coverage` need no correspondence at all.

Only the legacy environments use this module: ``GraspRouteEnv`` and ``SurfaceRouteEnv`` measure their "shape" error
with :func:`ordered` against per-sample targets on the route. The final ``GenericSurfaceEnv`` replaces it with the
correspondence-free coverage of a target surface (``GenericSurfaceEnv.coverage``).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def ordered(points: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None,
            allow_reverse: bool = False) -> float:
    """Mean distance between corresponding points: point k is compared with target k.

    Parameters
    ----------
    points : np.ndarray, shape (K, 3)
        Current sample positions [m].
    targets : np.ndarray, shape (K, 3)
        Target position of each sample [m].
    mask : np.ndarray of bool, shape (K,), optional
        Samples to include; all if omitted.
    allow_reverse : bool
        Also try the reversed order of the points and return the smaller distance.

    Returns
    -------
    float
        Mean distance [m].

    Examples
    --------
    >>> points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> targets = points + np.array([0.0, 0.1, 0.0])
    >>> round(ordered(points, targets), 6), round(ordered(points[::-1], targets), 6)
    (0.1, 0.703325)
    >>> round(ordered(points[::-1], targets, allow_reverse=True), 6)
    0.1
    """
    def mean_error(candidate):
        """Mean (masked) distance of ``candidate`` from the targets [m]."""
        errors = np.linalg.norm(candidate - targets, axis=1)
        return float(errors[mask].mean() if mask is not None else errors.mean())

    return min(mean_error(points), mean_error(points[::-1])) if allow_reverse else mean_error(points)


def hungarian(points: np.ndarray, targets: np.ndarray) -> float:
    """Mean distance under the optimal one-to-one assignment of points to targets (Hungarian algorithm).

    Parameters
    ----------
    points : np.ndarray, shape (K, 3)
        Current sample positions [m].
    targets : np.ndarray, shape (M, 3)
        Target points [m].

    Returns
    -------
    float
        Mean distance of the assigned pairs [m].

    Examples
    --------
    >>> points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> hungarian(points[::-1], points)
    0.0
    """
    distance = np.linalg.norm(points[:, None] - targets[None], axis=2)
    rows, cols = linear_sum_assignment(distance)
    return float(distance[rows, cols].mean())


def chamfer(points: np.ndarray, targets: np.ndarray) -> float:
    """Symmetric Chamfer distance: the mean of the two mean nearest-neighbour distances.

    Parameters
    ----------
    points : np.ndarray, shape (K, 3)
        Current sample positions [m].
    targets : np.ndarray, shape (M, 3)
        Target points [m].

    Returns
    -------
    float
        Chamfer distance [m].

    Examples
    --------
    >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> round(chamfer(points, targets), 6)
    0.166667
    """
    distance = np.linalg.norm(points[:, None] - targets[None], axis=2)
    return 0.5 * float(distance.min(axis=1).mean() + distance.min(axis=0).mean())


def coverage(points: np.ndarray, targets: np.ndarray) -> float:
    """Coverage of a target set (Berenson 2013): the mean distance from each target to its nearest point.

    One-sided: points far from every target cost nothing.

    Parameters
    ----------
    points : np.ndarray, shape (K, 3)
        Current sample positions [m].
    targets : np.ndarray, shape (M, 3)
        Target points [m].

    Returns
    -------
    float
        Mean nearest-point distance of the targets [m].

    Examples
    --------
    >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    >>> round(coverage(points, targets), 6), coverage(targets, points)
    (0.333333, 0.0)
    """
    distance = np.linalg.norm(points[:, None] - targets[None], axis=2)
    return float(distance.min(axis=0).mean())
