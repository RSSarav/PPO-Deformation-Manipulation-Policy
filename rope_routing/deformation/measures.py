"""Generic deformation measures on a material graph (notes/A_deformation_math.md, section 6).

Every measure is computed only from the rest sample positions, the current sample positions, the rest-neighbour edges
and the rest geodesic distances of a :class:`MaterialGraph`; none of them knows what kind of object it measures.
Positions are in metres; strains and stretches are dimensionless.

These point-graph measures predate the fundamental-form descriptor of ``surface_forms.py``. The legacy
``GraspRouteEnv`` observes :func:`node_features`, and the start-state driver (``env/start_states.py``) uses
:func:`node_strain` as a strain guard. The final ``GenericSurfaceEnv`` measures deformation with ``surface_forms.py``
instead: the chord strain of this module reads every bend as compression and cannot see stretch on a peg wrap.
"""

import numpy as np

from .material_graph import MaterialGraph


def edge_green_strain(graph: MaterialGraph, positions: np.ndarray) -> np.ndarray:
    """Green strain of every edge, eps = (l^2 / L^2 - 1) / 2 with current length l and rest length L.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph (rest positions and edges).
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].

    Returns
    -------
    np.ndarray, shape (E,)
        Strain of each edge.

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> edge_green_strain(graph, 1.1 * graph.rest_points).round(4).tolist()
    [0.105, 0.105]
    """
    first, second = graph.edges[:, 0], graph.edges[:, 1]
    rest_sq = np.sum((graph.rest_points[second] - graph.rest_points[first]) ** 2, axis=1)
    current_sq = np.sum((positions[second] - positions[first]) ** 2, axis=1)
    return 0.5 * (current_sq / rest_sq - 1.0)


def node_strain(graph: MaterialGraph, positions: np.ndarray) -> np.ndarray:
    """Mean Green strain of the edges at each sample (0 for samples without edges).

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].

    Returns
    -------
    np.ndarray, shape (K,)
        Strain per sample.

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> node_strain(graph, 1.1 * graph.rest_points).round(4).tolist()
    [0.105, 0.105, 0.105]
    """
    strain = edge_green_strain(graph, positions)
    total = np.zeros(graph.num_samples)
    count = np.zeros(graph.num_samples)
    for edge, (first, second) in enumerate(graph.edges):
        total[first] += strain[edge]
        total[second] += strain[edge]
        count[first] += 1
        count[second] += 1
    return total / np.maximum(count, 1)


def deformation_gradients(graph: MaterialGraph, positions: np.ndarray, sigma: float | None = None):
    """Moving-least-squares deformation gradient of each sample, with its polar rotation and principal stretches.

    For sample i with rest-neighbour offsets X_ij and current offsets x_ij, F_i = B A^+ with
    A = sum_j w_j X_ij X_ij^T, B = sum_j w_j x_ij X_ij^T and Gaussian weights w_j = exp(-|X_ij|^2 / sigma^2). The
    singular value decomposition F_i = U S V^T gives the polar rotation R_i = U V^T and the principal stretches S.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].
    sigma : float, optional
        Kernel width [m]; by default twice the root-mean-square neighbour distance of each sample.

    Returns
    -------
    gradients : np.ndarray, shape (K, 3, 3)
        Deformation gradients F_i (zero for samples without neighbours).
    rotations : np.ndarray, shape (K, 3, 3)
        Polar rotations R_i (identity for samples without neighbours).
    stretch : np.ndarray, shape (K, 3)
        Principal stretches in descending order (ones for samples without neighbours).

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> gradients, rotations, stretch = deformation_gradients(graph, 1.1 * graph.rest_points)
    >>> stretch[:, 0].round(4).tolist()
    [1.1, 1.1, 1.1]
    """
    neighbours = graph.neighbours()
    count = graph.num_samples
    gradients = np.zeros((count, 3, 3))
    rotations = np.tile(np.eye(3), (count, 1, 1))
    stretch = np.ones((count, 3))
    for node in range(count):
        members = neighbours[node]
        if len(members) == 0:
            continue
        rest_offsets = graph.rest_points[members] - graph.rest_points[node]
        offsets = positions[members] - positions[node]
        width_sq = sigma**2 if sigma else np.mean(np.sum(rest_offsets**2, axis=1)) * 4.0
        kernel = np.exp(-np.sum(rest_offsets**2, axis=1) / width_sq)
        rest_moment = np.einsum("e,ea,eb->ab", kernel, rest_offsets, rest_offsets)
        cross_moment = np.einsum("e,ea,eb->ab", kernel, offsets, rest_offsets)
        gradients[node] = cross_moment @ np.linalg.pinv(rest_moment, rcond=1e-6)
        left, singular, right_t = np.linalg.svd(gradients[node])
        rotations[node] = left @ right_t
        stretch[node] = singular
    return gradients, rotations, stretch


def bending(graph: MaterialGraph, positions: np.ndarray, rotations: np.ndarray | None = None) -> np.ndarray:
    """Bending of each sample from its Laplacian coordinates.

    b_i = 2 |delta_i - R_i delta0_i| / h_i^2, where delta_i is the offset of sample i from the mean of its neighbours
    (current and rest, the rest one rotated by R_i) and h_i the mean rest distance to the neighbours. For samples on a
    circle of radius R this is about 1/R. Samples with fewer than two neighbours get 0.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].
    rotations : np.ndarray, shape (K, 3, 3), optional
        Rotations R_i of the rest offsets (e.g. the polar rotations of :func:`deformation_gradients`); none if
        omitted.

    Returns
    -------
    np.ndarray, shape (K,)
        Bending per sample [1/m].

    Examples
    --------
    Seven samples on a circle of radius 0.2 m (1/R = 5 1/m):

    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 0.3, 31), np.zeros(31), np.zeros(31)], 7)
    >>> angle = graph.rest_points[:, 0] / 0.2
    >>> circle = np.c_[0.2 * np.sin(angle), 0.2 * (1 - np.cos(angle)), np.zeros(7)]
    >>> _, rotations, _ = deformation_gradients(graph, circle)
    >>> bending(graph, circle, rotations).round(2).tolist()
    [0.0, 4.97, 4.97, 4.97, 4.97, 4.97, 0.0]
    """
    neighbours = graph.neighbours()
    curvature = np.zeros(graph.num_samples)
    for node, members in enumerate(neighbours):
        if len(members) < 2:
            continue
        delta = positions[node] - positions[members].mean(axis=0)
        rest_delta = graph.rest_points[node] - graph.rest_points[members].mean(axis=0)
        if rotations is not None:
            rest_delta = rotations[node] @ rest_delta
        spacing = np.mean(np.linalg.norm(graph.rest_points[members] - graph.rest_points[node], axis=1))
        curvature[node] = 2.0 * np.linalg.norm(delta - rest_delta) / spacing**2
    return curvature


def tautness_to(graph: MaterialGraph, positions: np.ndarray, idx: int) -> np.ndarray:
    """Tautness of every sample toward a reference sample: tau_i = |x_i - x_idx| / D_i,idx.

    The straight-line distance divided by the rest geodesic distance is 1 when the material between the two samples
    is pulled straight and much less than 1 when it is slack.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].
    idx : int
        Index of the reference sample.

    Returns
    -------
    np.ndarray, shape (K,)
        Tautness per sample (1 for the reference sample itself).

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> tautness_to(graph, graph.rest_points, 0).tolist()
    [1.0, 1.0, 1.0]
    """
    distance = np.linalg.norm(positions - positions[idx], axis=1)
    geodesic = graph.geodesic[:, idx]
    tautness = np.ones(graph.num_samples)
    valid = geodesic > 1e-9
    tautness[valid] = distance[valid] / geodesic[valid]
    return tautness


def overstretch(graph: MaterialGraph, positions: np.ndarray) -> np.ndarray:
    """Largest relative excess of a straight-line distance over the rest geodesic distance, per sample.

    s_i = max_j (|x_i - x_j| - D_ij) / D_ij. A coarse guard: material stretched around an obstacle keeps its
    straight-line distances short and goes unnoticed.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].

    Returns
    -------
    np.ndarray, shape (K,)
        Overstretch per sample (positive when some distance exceeds its rest geodesic distance).

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> overstretch(graph, 1.1 * graph.rest_points).round(3).tolist()
    [0.1, 0.1, 0.1]
    """
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(graph.geodesic > 1e-9, (distance - graph.geodesic) / graph.geodesic, -np.inf)
    return excess.max(axis=1)


def rigidity(graph: MaterialGraph, grasp_idx: int, decay: float) -> np.ndarray:
    """Berenson's diminishing rigidity, w_i = exp(-decay * D_i,grasp).

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    grasp_idx : int
        Index of the grasped sample.
    decay : float
        Decay rate [1/m].

    Returns
    -------
    np.ndarray, shape (K,)
        Rigidity weight per sample, 1 at the grasped sample.

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> rigidity(graph, 1, 2.0).round(4).tolist()
    [0.3679, 1.0, 0.3679]
    """
    return np.exp(-decay * graph.geodesic[:, grasp_idx])


def node_features(graph: MaterialGraph, positions: np.ndarray, grasp_idx: int,
                  anchor_idx: int) -> dict[str, np.ndarray]:
    """Per-sample generic deformation features observed by the legacy ``GraspRouteEnv``.

    Parameters
    ----------
    graph : MaterialGraph
        Material graph.
    positions : np.ndarray, shape (K, 3)
        Current sample positions [m].
    grasp_idx, anchor_idx : int
        Samples toward which the tautness is measured: the one nearest the gripper and the anchored one.

    Returns
    -------
    dict of str to np.ndarray, each of shape (K,)
        "strain" (:func:`node_strain`), "stretch" (largest principal stretch), "bending" (with the polar rotations),
        "taut_grasp", "taut_anchor" (:func:`tautness_to`) and "overstretch".

    Examples
    --------
    >>> from rope_routing.deformation.material_graph import from_polyline
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> sorted(node_features(graph, graph.rest_points, grasp_idx=2, anchor_idx=0))
    ['bending', 'overstretch', 'strain', 'stretch', 'taut_anchor', 'taut_grasp']
    """
    _, rotations, stretch = deformation_gradients(graph, positions)
    return {
        "strain": node_strain(graph, positions),
        "stretch": stretch[:, 0],
        "bending": bending(graph, positions, rotations),
        "taut_grasp": tautness_to(graph, positions, grasp_idx),
        "taut_anchor": tautness_to(graph, positions, anchor_idx),
        "overstretch": overstretch(graph, positions),
    }
