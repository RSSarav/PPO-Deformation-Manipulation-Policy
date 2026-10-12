"""Tests of the generic point-graph deformation measures (deformation/measures.py), the point-set shape distance and
the winding angle on analytic cases (notes/A_deformation_math.md, section 6)."""

import numpy as np

from rope_routing.deformation import measures, shape_distance
from rope_routing.deformation.material_graph import from_polyline
from rope_routing.task.geometry import winding_angles


def straight(count=21, length=1.0):
    """Material graph of a straight polyline along x with ``count`` samples over ``length`` [m]."""
    rest = np.zeros((count, 3))
    rest[:, 0] = np.linspace(0, length, count)
    return from_polyline(rest, count)


def rot_z(angle):
    """Rotation matrix about z by ``angle`` [rad]."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])


def test_rigid_motion_is_undeformed():
    """A rigid motion gives zero strain, unit stretch and zero bending."""
    graph = straight()
    moved = graph.rest_points @ rot_z(0.7).T + np.array([0.3, -0.2, 0.1])
    assert np.allclose(measures.node_strain(graph, moved), 0, atol=1e-12)
    _, rotations, stretch = measures.deformation_gradients(graph, moved)
    assert np.allclose(stretch[:, 0], 1, atol=1e-9)
    assert np.allclose(measures.bending(graph, moved, rotations), 0, atol=1e-9)


def test_uniform_stretch():
    """A uniform 1% stretch gives the Green strain (s^2 - 1) / 2, stretch s and positive overstretch."""
    graph = straight()
    factor = 1.01
    stretched = graph.rest_points * factor
    assert np.allclose(measures.node_strain(graph, stretched), 0.5 * (factor**2 - 1))
    _, _, stretch = measures.deformation_gradients(graph, stretched)
    assert np.allclose(stretch[:, 0], factor)
    assert np.all(measures.overstretch(graph, stretched) > 0)


def test_affine_F_exact_on_tangent():
    """The MLS deformation gradient reproduces an affine map exactly along the tangent of a 1-D object."""
    graph = straight()
    affine = np.array([[1.2, 0.1, 0], [0.3, 0.9, 0], [0, 0.2, 1.1]])
    gradients, _, _ = measures.deformation_gradients(graph, graph.rest_points @ affine.T)
    tangent = np.array([1.0, 0, 0])
    for gradient in gradients:
        assert np.allclose(gradient @ tangent, affine @ tangent, atol=1e-9)


def test_bending_of_circle_is_one_over_R():
    """A straight rope bent into a quarter circle of radius R has bending 1/R at its interior samples."""
    radius, count = 0.1, 21
    length = 0.5 * np.pi * radius  # quarter circle
    graph = straight(count, length)
    angle = graph.rest_points[:, 0] / radius
    circle = np.stack([radius * np.sin(angle), radius * (1 - np.cos(angle)), np.zeros(count)], 1)
    _, rotations, _ = measures.deformation_gradients(graph, circle)
    curvature = measures.bending(graph, circle, rotations)[1:-1]
    assert np.allclose(curvature, 1 / radius, rtol=0.01)


def test_tautness():
    """Tautness is 1 for the straight rest shape and much less than 1 at the end of a folded rope."""
    graph = straight()
    assert np.allclose(measures.tautness_to(graph, graph.rest_points, 0)[1:], 1)
    folded = graph.rest_points.copy()
    folded[11:, 0] = 2 * folded[10, 0] - folded[11:, 0]
    folded[11:, 1] = 0.01
    assert measures.tautness_to(graph, folded, 0)[-1] < 0.1


def test_shape_distance_reverse_symmetry():
    """With allow_reverse, the ordered distance of the reversed samples equals that of the original order."""
    graph = straight()
    targets = graph.rest_points + np.array([0, 0.1, 0])
    assert np.isclose(shape_distance.ordered(graph.rest_points[::-1], targets, allow_reverse=True), 0.1)


def test_winding_full_loop():
    """A unit circle winds 2 pi around a peg at its centre and nearly 0 around a peg outside it."""
    angle = np.linspace(0, 2 * np.pi, 200)
    loop = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], 1)
    winding = winding_angles(loop, np.array([[0, 0, 0], [5, 0, 0]]), 1.0)
    assert np.isclose(winding[0], 2 * np.pi, atol=1e-6) and abs(winding[1]) < 0.5
