"""Task geometry for routing a rope around three pegs.

This module defines the routing task (peg layout, the target route of the rope, and the legacy gripper waypoints) and
the winding-angle measure used to decide how far a curve is wrapped around each peg.

Coordinates are in metres in the robot base frame (x forward, y left, z up). Angles are in radians.

The final, object-agnostic environment (``env/generic_surface_env.py``) uses only ``TriangleTask.route`` (to build the
target surface), ``pegs`` and ``peg_height`` (as unlabelled obstacles), ``anchor`` and ``route_length`` (to build the
rope), and :func:`winding_angles`. The waypoints and the per-sample targets serve the earlier ``SurfaceRouteEnv`` only.
"""

from dataclasses import dataclass, field

import numpy as np


def wrap(angle):
    """Wrap angles to the interval [-pi, pi).

    Parameters
    ----------
    angle : float or np.ndarray
        Angle(s) in radians.

    Returns
    -------
    float or np.ndarray
        The equivalent angle(s) in [-pi, pi).

    Examples
    --------
    >>> round(float(np.degrees(wrap(np.radians(-340.0)))), 6)
    20.0
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def winding_angles(curve: np.ndarray, pegs: np.ndarray, peg_top: float) -> np.ndarray:
    """Signed angle swept by a curve around each peg, counting only the part of the curve below the peg tops.

    Seen from a peg's axis, each point of the curve has a bearing. The winding angle is the sum of the bearing changes
    between consecutive points: 2 pi for a closed loop around the peg, near zero for a curve that passes by.
    Segments with an endpoint at or above ``peg_top`` are ignored, so a curve lifted over a peg does not wrap it.

    Parameters
    ----------
    curve : np.ndarray, shape (n_points, 3)
        Points along the curve, in order [m].
    pegs : np.ndarray, shape (n_pegs, 2) or (n_pegs, 3)
        Peg centres; only x and y are used [m].
    peg_top : float
        Height of the peg tops [m].

    Returns
    -------
    np.ndarray, shape (n_pegs,)
        Winding angle around each peg [rad]; counter-clockwise is positive.

    Examples
    --------
    A quarter circle and a full circle of radius 5 cm around a peg at the origin:

    >>> angles = np.radians(np.linspace(0.0, 90.0, 9))
    >>> quarter = np.c_[0.05 * np.cos(angles), 0.05 * np.sin(angles), np.full(9, 0.01)]
    >>> round(float(np.degrees(winding_angles(quarter, np.zeros((1, 2)), 0.06)[0])), 1)
    90.0
    >>> angles = np.radians(np.linspace(0.0, 360.0, 25))
    >>> loop = np.c_[0.05 * np.cos(angles), 0.05 * np.sin(angles), np.full(25, 0.01)]
    >>> round(float(np.degrees(winding_angles(loop, np.zeros((1, 2)), 0.06)[0])), 1)
    360.0
    """
    result = np.zeros(len(pegs))
    below_top = curve[:, 2] < peg_top
    counted = below_top[:-1] & below_top[1:]
    for index, peg in enumerate(pegs):
        bearing = np.arctan2(curve[:, 1] - peg[1], curve[:, 0] - peg[0])
        result[index] = wrap(np.diff(bearing))[counted].sum()
    return result


@dataclass
class TriangleTask:
    """Routing task: three pegs and the route a rope should follow around them.

    The route starts at the rope's anchored end beside peg A, runs along the outside of the triangle around B, C and
    A (counter-clockwise), and stops ``end_gap`` short of its start. Along the straight edges the rope centre lies
    ``peg_radius + rope_radius`` outside the triangle; around each peg it follows a circular arc of that radius.

    Parameters
    ----------
    pegs : np.ndarray, shape (3, 2)
        Centres of pegs A, B and C in counter-clockwise order [m].
    peg_radius : float
        Peg radius [m].
    peg_height : float
        Peg height [m].
    rope_radius : float
        Rope radius [m].
    anchor_offset : float
        Distance of the anchored rope end from peg A's tangent point, along edge A->B [m].
    end_gap : float
        The route ends this far (along the route) before its start, so the rope does not overlap itself [m].
    waypoint_clearance : float
        Legacy waypoints: distance of each waypoint beyond its peg's surface, radially from the triangle centroid [m].
        Smaller values (0.05, 0.08) let the trailing rope cut inside peg C.
    arc_radius : float
        Legacy waypoints: if positive, the waypoints instead lie on a circle of this radius around the centroid [m].

    Attributes
    ----------
    route : np.ndarray, shape (n_route, 2)
        Dense target route of the rope centre line, starting at the anchor [m].
    route_s : np.ndarray, shape (n_route,)
        Arclength of each route point from the anchor [m].
    waypoints : np.ndarray, shape (n_waypoints, 2)
        Legacy gripper waypoints (``SurfaceRouteEnv`` only) [m].

    Examples
    --------
    >>> pegs = np.array([[0.40, -0.09], [0.58, -0.09], [0.49, 0.066]])
    >>> task = TriangleTask(pegs=pegs, peg_height=0.07, rope_radius=0.005)
    >>> task.route.shape
    (181, 2)
    >>> round(task.route_length, 4)
    0.6149
    >>> task.anchor.round(4).tolist()
    [0.43, -0.11]
    """

    pegs: np.ndarray
    peg_radius: float = 0.015
    peg_height: float = 0.06
    rope_radius: float = 0.006
    anchor_offset: float = 0.03
    end_gap: float = 0.05
    waypoint_clearance: float = 0.10
    arc_radius: float = 0.0
    route: np.ndarray = field(init=False)
    route_s: np.ndarray = field(init=False)
    waypoints: np.ndarray = field(init=False)

    def __post_init__(self):
        pegs = self.pegs
        offset = self.peg_radius + self.rope_radius  # peg centre to rope centre when the rope touches the peg

        # Outward unit normal of each edge (A->B, B->C, C->A) of the counter-clockwise triangle.
        normals = []
        for corner in range(3):
            direction = pegs[(corner + 1) % 3] - pegs[corner]
            direction = direction / np.linalg.norm(direction)
            normals.append(np.array([direction[1], -direction[0]]))

        def segment(start, end, count=40):
            """``count`` evenly spaced points from ``start`` towards ``end`` (``end`` excluded)."""
            fraction = np.linspace(0, 1, count, endpoint=False)
            return start[None] + fraction[:, None] * (end - start)[None]

        def wrap_arc(centre, normal_in, normal_out, count=24):
            """``count`` points on the circle of radius ``offset`` around ``centre``, turning counter-clockwise from
            direction ``normal_in`` to ``normal_out`` (end excluded): the rope bending around a peg."""
            angle_in = np.arctan2(normal_in[1], normal_in[0])
            angle_out = np.arctan2(normal_out[1], normal_out[0])
            angle_out = angle_in + (angle_out - angle_in) % (2 * np.pi)  # always turn the short, counter-clockwise way
            angles = np.linspace(angle_in, angle_out, count, endpoint=False)
            return centre[None] + offset * np.stack([np.cos(angles), np.sin(angles)], 1)

        # Route: anchor -> along A-B -> around B -> along B-C -> around C -> along C-A -> around A -> back to the anchor.
        peg_a, peg_b, peg_c = pegs
        normal_ab, normal_bc, normal_ca = normals
        direction_ab = (peg_b - peg_a) / np.linalg.norm(peg_b - peg_a)
        start = peg_a + offset * normal_ab + self.anchor_offset * direction_ab
        pieces = [
            segment(start, peg_b + offset * normal_ab),
            wrap_arc(peg_b, normal_ab, normal_bc),
            segment(peg_b + offset * normal_bc, peg_c + offset * normal_bc),
            wrap_arc(peg_c, normal_bc, normal_ca),
            segment(peg_c + offset * normal_ca, peg_a + offset * normal_ca),
            wrap_arc(peg_a, normal_ca, normal_ab),
            segment(peg_a + offset * normal_ab, start, 20),
            start[None],
        ]
        route = np.concatenate(pieces)

        # Arclength along the route; drop the last end_gap metres so the route does not close on itself.
        arclength = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
        keep = arclength <= arclength[-1] - self.end_gap
        self.route, self.route_s = route[keep], arclength[keep]

        # Legacy gripper waypoints: beyond each peg (B, C, A), radially from the centroid, then just past the route end.
        centroid = pegs.mean(axis=0)
        beyond_end = self.route[-1] + 0.03 * (self.route[-1] - peg_a) / np.linalg.norm(self.route[-1] - peg_a)
        waypoints = []
        for peg in (peg_b, peg_c, peg_a):
            outward = (peg - centroid) / np.linalg.norm(peg - centroid)
            waypoints.append(peg + (self.peg_radius + self.waypoint_clearance) * outward)
        waypoints.append(beyond_end)
        self.waypoints = np.array(waypoints)

        # Optional legacy alternative: waypoints on a counter-clockwise arc around the centroid, every <= 20 degrees,
        # from just before peg B's bearing to the bearing of the route end.
        if self.arc_radius > 0:
            def bearing(point):
                return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

            angle_start = bearing(peg_b) - 0.25
            angle_end = angle_start + (bearing(self.route[-1]) - angle_start) % (2 * np.pi)
            count = max(4, int(np.ceil((angle_end - angle_start) / np.deg2rad(20))))
            angles = np.linspace(angle_start, angle_end, count + 1)
            circle = centroid[None] + self.arc_radius * np.stack([np.cos(angles), np.sin(angles)], 1)
            self.waypoints = np.concatenate([circle, beyond_end[None]])

    @property
    def anchor(self) -> np.ndarray:
        """Position of the rope's anchored end: the first route point, shape (2,) [m]."""
        return self.route[0]

    @property
    def route_length(self) -> float:
        """Length of the route [m]."""
        return float(self.route_s[-1])

    def targets(self, sample_s: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample goal points on the route (legacy ``SurfaceRouteEnv`` only).

        Parameters
        ----------
        sample_s : np.ndarray, shape (n_samples,)
            Rest arclength of each material sample from the anchored end [m].
        height : float
            Height of the goal points [m].

        Returns
        -------
        goals : np.ndarray, shape (n_samples, 3)
            Route point at each sample's arclength (clamped to the route end) [m].
        on_route : np.ndarray of bool, shape (n_samples,)
            True for samples within the route length; the remaining rope has no goal.
        """
        on_route = sample_s <= self.route_length
        clamped = np.clip(sample_s, 0, self.route_length)
        goals = np.stack([np.interp(clamped, self.route_s, self.route[:, 0]),
                          np.interp(clamped, self.route_s, self.route[:, 1]),
                          np.full_like(clamped, height)], 1)
        return goals, on_route

    def target_windings(self, height: float) -> np.ndarray:
        """Winding angle of the route around each peg, with the route at ``height`` [rad], shape (3,).

        Legacy ``SurfaceRouteEnv`` only; ``GenericSurfaceEnv`` computes its target windings from the target surface.
        """
        route_3d = np.concatenate([self.route, np.full((len(self.route), 1), height)], 1)
        return winding_angles(route_3d, np.concatenate([self.pegs, np.zeros((3, 1))], 1), self.peg_height)
