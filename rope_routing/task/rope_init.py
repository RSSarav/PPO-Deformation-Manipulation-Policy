"""Random initial rope layouts on the table, with one end anchored.

A layout is a smooth planar curve for the rope's centreline. It starts at the anchor, heads away from the peg
triangle, and is built by integrating a random smooth curvature profile. A candidate is accepted only if it satisfies
every constraint below (rejection sampling):

* every node keeps at least 1.2 cm of clearance from every peg;
* apart from the first three nodes (at and next to the anchor), no node lies inside the peg triangle enlarged by
  4 cm;
* the rope lies in the arm's reachable annulus around the robot base, with its free end and its grasp region (the
  last ``grasp_region`` metres) at distances from the base that the arm reaches with the tool pointing down;
* the curve neither crosses nor nearly touches itself;
* the curve is not already wrapped around a peg (the bearing from each peg turns by at most 90 degrees).

Coordinates are in metres in the robot base frame (x forward, y left); layouts are planar (x, y). Angles are in
radians.

Every environment draws one layout per episode in ``reset`` (``GraspRouteEnv``, ``SurfaceRouteEnv`` and, through it,
the final ``GenericSurfaceEnv``).
"""

import numpy as np


def _point_in_triangle(point, triangle):
    """Test whether a point lies inside a triangle or on its boundary, using barycentric coordinates.

    Parameters
    ----------
    point : np.ndarray, shape (2,)
        Query point [m].
    triangle : np.ndarray, shape (3, 2)
        Triangle corners [m].

    Returns
    -------
    bool
        True if the point is inside the triangle or on its boundary.

    Examples
    --------
    >>> triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> bool(_point_in_triangle(np.array([0.2, 0.2]), triangle))
    True
    >>> bool(_point_in_triangle(np.array([0.8, 0.8]), triangle))
    False
    """
    corner_a, corner_b, corner_c = triangle
    # Write point - a = coord_c (c - a) + coord_b (b - a) and solve the 2 x 2 normal equations for the coordinates.
    edge_c, edge_b, offset = corner_c - corner_a, corner_b - corner_a, point - corner_a
    dot_cc, dot_cb, dot_bb = edge_c @ edge_c, edge_c @ edge_b, edge_b @ edge_b
    dot_pc, dot_pb = offset @ edge_c, offset @ edge_b
    denominator = dot_cc * dot_bb - dot_cb * dot_cb
    coord_c = (dot_bb * dot_pc - dot_cb * dot_pb) / denominator
    coord_b = (dot_cc * dot_pb - dot_cb * dot_pc) / denominator
    return coord_c >= 0 and coord_b >= 0 and coord_c + coord_b <= 1


def _offset_triangle(pegs, margin):
    """Enlarge the triangle of peg centres by moving each corner ``margin`` further away from the centroid.

    Parameters
    ----------
    pegs : np.ndarray, shape (3, 2)
        Peg centres [m].
    margin : float
        Distance each corner moves outward, along the ray from the centroid [m].

    Returns
    -------
    np.ndarray, shape (3, 2)
        Corners of the enlarged triangle [m].

    Examples
    --------
    >>> _offset_triangle(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0]]), 1.0).round(3).tolist()
    [[1.707, -0.707], [-1.707, -0.707], [0.0, 4.0]]
    """
    centroid = pegs.mean(0)
    return np.array([centroid + (peg - centroid) * (1 + margin / np.linalg.norm(peg - centroid)) for peg in pegs])


def random_layout(rng: np.random.Generator, anchor: np.ndarray, pegs: np.ndarray, length: float, num_nodes: int,
                  peg_radius: float, rope_radius: float, base_xy=(0.0, 0.0), reach=(0.36, 0.70),
                  end_reach=(0.45, 0.66), grasp_region: float = 0.15, max_tries: int = 4000) -> np.ndarray:
    """Sample a random planar rope layout that starts at the anchor and satisfies the module's constraints.

    Each candidate is integrated from a random heading and curvature profile. The initial heading points from the peg
    triangle's centroid to the anchor, perturbed by up to +-1.2 rad. The curvature profile is a sum of four sinusoidal
    harmonics over the rope length (harmonic ``m`` has a normal random amplitude with standard deviation 7/m [1/m] and a
    random phase), clipped to +-18 1/m, a minimum bend radius of about 5.5 cm.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator; the layout is a deterministic function of its state.
    anchor : np.ndarray, shape (2,)
        Position of the anchored rope end (node 0) [m].
    pegs : np.ndarray, shape (n_pegs, 2)
        Peg centres [m].
    length : float
        Rope length [m].
    num_nodes : int
        Number of rope nodes, equally spaced along the rope.
    peg_radius, rope_radius : float
        Peg and rope radii [m].
    base_xy : tuple of float
        Robot base position (x, y) [m].
    reach : tuple of float
        Reachable annulus (inner, outer radius) around the base [m]. The first three nodes may lie inside the inner
        radius, and every node may exceed the outer radius by up to 6 cm.
    end_reach : tuple of float
        Allowed distance range of the free end (last node) from the base [m].
    grasp_region : float
        Length of the rope's free end that must lie between 0.42 m and 0.68 m from the base [m].
    max_tries : int
        Maximum number of candidates before giving up.

    Returns
    -------
    np.ndarray, shape (num_nodes, 2)
        Node positions (x, y) [m]; node 0 is at the anchor.

    Raises
    ------
    RuntimeError
        If no candidate satisfies the constraints within ``max_tries`` attempts.

    Examples
    --------
    >>> from rope_routing.task.geometry import TriangleTask
    >>> task = TriangleTask(pegs=np.array([[0.40, -0.09], [0.58, -0.09], [0.49, 0.066]]), peg_height=0.07,
    ...                     rope_radius=0.005)
    >>> layout = random_layout(np.random.default_rng(0), task.anchor, task.pegs, task.route_length + 0.20, 61,
    ...                        task.peg_radius, 0.005)
    >>> layout.shape, bool(np.allclose(layout[0], task.anchor))
    ((61, 2), True)
    """
    arclength = np.linspace(0.0, length, num_nodes)
    spacing = arclength[1] - arclength[0]
    centroid = pegs.mean(0)
    keep_out = _offset_triangle(pegs, 0.04)
    outward = anchor - centroid
    outward_angle = np.arctan2(outward[1], outward[0])
    base = np.asarray(base_xy)
    for _ in range(max_tries):
        # Candidate curve: random initial heading and smooth random curvature profile, integrated along the rope.
        start_heading = outward_angle + rng.uniform(-1.2, 1.2)
        curvature = np.zeros_like(arclength)
        for harmonic in range(1, 5):
            curvature += (rng.normal(0, 7.0 / harmonic)
                          * np.sin(2 * np.pi * harmonic * arclength / length + rng.uniform(0, 2 * np.pi)))
        curvature = np.clip(curvature, -18.0, 18.0)  # minimum bend radius of about 5.5 cm
        heading = start_heading + np.concatenate([[0.0], np.cumsum(0.5 * (curvature[1:] + curvature[:-1]) * spacing)])
        xy = anchor + np.concatenate(
            [[[0.0, 0.0]], np.cumsum(spacing * np.stack([np.cos(heading[:-1]), np.sin(heading[:-1])], 1), 0)])

        # Peg clearance of every node.
        peg_distance = np.linalg.norm(xy[:, None] - pegs[None], axis=2)
        if peg_distance.min() < peg_radius + rope_radius + 0.012:
            continue
        # Keep-out triangle, except for the first nodes next to the anchor.
        if any(_point_in_triangle(point, keep_out) for point in xy[3:]):
            continue
        # Workspace: the whole rope inside the reachable annulus, the free end comfortably reachable.
        base_distance = np.linalg.norm(xy - base, axis=1)
        if base_distance[3:].min() < reach[0] or base_distance.max() > reach[1] + 0.06:
            continue
        if not (end_reach[0] <= base_distance[-1] <= end_reach[1]):
            continue
        # The grasp region must be comfortably reachable with the tool pointing down.
        tail = arclength >= length - grasp_region
        if base_distance[tail].min() < 0.42 or base_distance[tail].max() > 0.68:
            continue
        # No self-intersection or near self-contact: nodes more than about 6 cm apart along the rope keep a centre
        # distance of at least four rope radii.
        pairwise = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        distant = np.abs(np.arange(num_nodes)[:, None] - np.arange(num_nodes)[None]) > int(0.06 / spacing) + 1
        if pairwise[distant].min() < 4 * rope_radius:
            continue
        # Not already wrapped around a peg: the bearing from each peg turns by at most 90 degrees end to end.
        ok = True
        for peg in pegs:
            bearing = np.unwrap(np.arctan2(xy[:, 1] - peg[1], xy[:, 0] - peg[0]))
            if abs(bearing[-1] - bearing[0]) > np.pi / 2:
                ok = False
                break
        if not ok:
            continue
        return xy
    raise RuntimeError("could not sample a valid rope layout")


def verification_seeds(count: int = 100, start: int = 9_000_000) -> list[int]:
    """Layout seeds of the verification set; training and held-out evaluation never use them.

    ``scripts/verify_rl.py`` uses the same range (``SEED0 = 9_000_000``).

    Parameters
    ----------
    count : int
        Number of seeds.
    start : int
        First seed.

    Returns
    -------
    list of int
        The seeds ``start, ..., start + count - 1``.

    Examples
    --------
    >>> verification_seeds(3)
    [9000000, 9000001, 9000002]
    """
    return list(range(start, start + count))
