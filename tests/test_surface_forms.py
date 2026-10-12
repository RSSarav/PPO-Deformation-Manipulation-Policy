"""Accuracy tests of the fundamental-form deformation descriptor (rope_routing/deformation/surface_forms.py).

Every check compares against a closed-form surface (sphere, cylinder, torus, helix tube, ellipsoid, flat or bent
sheet) or against an independent re-implementation (mixed areas, area gradient, minimal-rotation frames, per-face
metric). The method is described in notes/A_deformation_math.md, whose section 7 maps each property to its test.
Run with -s to print the measured errors.

Lengths are in metres. The rope constants match the simulated rope: radius R_ROPE, N_ROPE nodes over the rest length
L_ROPE, node spacing H_ROPE.
"""

import time
import warnings

import numpy as np
import pytest

from rope_routing.deformation import surface_forms as sf
from rope_routing.deformation.surface_forms import (SurfaceMeshForms, TubeSurfaceDescriptor, curve_curvature_vectors,
                                                    polyline_window_curvature, ring_radius,
                                                    signed_planar_curvature_of_polyline, tube_surface)

R_ROPE, N_ROPE, L_ROPE = 0.005, 61, 0.81
H_ROPE = L_ROPE / (N_ROPE - 1)  # 13.5 mm node spacing of the simulated rope


# ----------------------------------------------------------------------------------------------------------- shapes
def icosphere(level, radius=1.0):
    """Icosphere: the icosahedron subdivided ``level`` times, every new vertex projected onto the sphere.

    Parameters
    ----------
    level : int
        Number of 1-to-4 subdivisions of the icosahedron.
    radius : float, optional
        Sphere radius [m].

    Returns
    -------
    vertices : np.ndarray, shape (V, 3)
        Vertex positions on the sphere [m].
    faces : np.ndarray of int, shape (F, 3)
        Triangles, oriented with outward normals.
    """
    golden = (1 + 5**0.5) / 2
    vertices = np.array([[-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0], [0, -1, golden],
                         [0, 1, golden], [0, -1, -golden], [0, 1, -golden], [golden, 0, -1], [golden, 0, 1],
                         [-golden, 0, -1], [-golden, 0, 1]], float)
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
             [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
             [8, 6, 7], [9, 8, 1]]
    vertices = list(vertices / np.linalg.norm(vertices, axis=1, keepdims=True))
    for _ in range(level):
        edge_midpoint, new_faces = {}, []

        def midpoint(first, second):
            """Index of the midpoint vertex of edge (first, second) on the unit sphere, created on first use."""
            edge = (min(first, second), max(first, second))
            if edge not in edge_midpoint:
                direction = vertices[first] + vertices[second]
                vertices.append(direction / np.linalg.norm(direction))
                edge_midpoint[edge] = len(vertices) - 1
            return edge_midpoint[edge]

        # Each triangle is split into three corner triangles and the central one.
        for va, vb, vc in faces:
            ab, bc, ca = midpoint(va, vb), midpoint(vb, vc), midpoint(vc, va)
            new_faces += [[va, ab, ca], [vb, bc, ab], [vc, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(vertices) * radius, np.array(faces)


def sheet(nu=41, nv=21, size_u=0.3, size_v=0.2):
    """Flat cloth: a regular grid in the xy plane, two triangles per cell, counter-clockwise seen from +z.

    Parameters
    ----------
    nu, nv : int, optional
        Number of grid points along x (u) and y (v).
    size_u, size_v : float, optional
        Extent of the sheet along x and y [m].

    Returns
    -------
    points : np.ndarray, shape (nu * nv, 3)
        Vertex positions [m], z = 0.
    faces : np.ndarray of int, shape (2 (nu - 1) (nv - 1), 3)
        Triangles.
    grid : np.ndarray of int, shape (nu, nv)
        Vertex index of each grid point.
    """
    coord_u, coord_v = np.meshgrid(np.linspace(0, size_u, nu), np.linspace(0, size_v, nv), indexing="ij")
    points = np.stack([coord_u.ravel(), coord_v.ravel(), np.zeros(coord_u.size)], 1)
    grid = np.arange(nu * nv).reshape(nu, nv)
    # Corner vertices of every grid cell, named by their (u, v) offsets.
    c00, c10, c11, c01 = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(), grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()
    return points, np.concatenate([np.stack([c00, c10, c11], 1), np.stack([c00, c11, c01], 1)]), grid


def arc(radius, count, ang, ccw=True):
    """Points on a circular arc in the xy plane, centred on the origin and starting at (radius, 0, 0).

    Parameters
    ----------
    radius : float
        Arc radius [m].
    count : int
        Number of points, at equal angular steps.
    ang : float
        Swept angle [rad].
    ccw : bool, optional
        Counter-clockwise (left-turning) if True, clockwise otherwise.

    Returns
    -------
    np.ndarray, shape (count, 3)
        Arc points [m].
    """
    th = np.linspace(0, ang, count)
    return np.stack([radius * np.cos(th), (1 if ccw else -1) * radius * np.sin(th), np.zeros(count)], 1)


def straight(num_nodes=N_ROPE, length=L_ROPE):
    """Straight rope along +x from the origin.

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes.
    length : float, optional
        Rope length [m].

    Returns
    -------
    np.ndarray, shape (num_nodes, 3)
        Evenly spaced nodes [m].
    """
    return np.c_[np.linspace(0, length, num_nodes), np.zeros((num_nodes, 2))]


def smooth_curve(seed, kmax=50.0, num_nodes=N_ROPE, length=L_ROPE, z_amplitude=0.003):
    """Random smooth, nearly planar curve with |curvature| up to ``kmax``.

    The planar curvature profile is a random sum of five sine modes along the arclength, scaled to a peak of ``kmax``
    and integrated twice. The curve is resampled at evenly spaced arclength and lifted out of the plane by a small
    sinusoid (three periods along the curve).

    Parameters
    ----------
    seed : int
        Seed of the curvature profile.
    kmax : float, optional
        Peak planar curvature [1/m].
    num_nodes : int, optional
        Number of nodes.
    length : float, optional
        Arclength of the planar curve [m].
    z_amplitude : float, optional
        Amplitude of the out-of-plane sinusoid [m].

    Returns
    -------
    np.ndarray, shape (num_nodes, 3)
        Curve nodes [m].
    """
    rng = np.random.default_rng(seed)
    arclength = np.linspace(0, length, 4000)
    curvature = sum(rng.normal() / mode * np.sin(2 * np.pi * mode * arclength / length + rng.uniform(0, 2 * np.pi))
                    for mode in range(1, 6))
    curvature *= kmax / np.abs(curvature).max()
    # Heading angle by trapezoidal integration of the curvature, then the planar position.
    heading = np.r_[0, np.cumsum(0.5 * (curvature[1:] + curvature[:-1]) * np.diff(arclength))]
    x_coord = np.r_[0, np.cumsum(np.cos(heading[:-1]) * np.diff(arclength))]
    y_coord = np.r_[0, np.cumsum(np.sin(heading[:-1]) * np.diff(arclength))]
    node_arclength = np.linspace(0, length, num_nodes)
    return np.stack([np.interp(node_arclength, arclength, x_coord), np.interp(node_arclength, arclength, y_coord),
                     z_amplitude * np.sin(6 * np.pi * node_arclength / length)], 1)


def random_rotation(seed):
    """Random proper rotation: QR decomposition of a Gaussian matrix, with signs fixed so that det = +1.

    Parameters
    ----------
    seed : int
        Seed of the Gaussian matrix.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix.
    """
    ortho, upper = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    ortho = ortho * np.sign(np.diag(upper))
    return ortho * np.linalg.det(ortho)


def rope_descriptor(sample_s=None, num_nodes=N_ROPE, length=L_ROPE):
    """TubeSurfaceDescriptor of a straight rest rope of radius R_ROPE.

    Parameters
    ----------
    sample_s : np.ndarray, optional
        Rest arclengths of the material samples [m]; 21 evenly spaced samples, ends included, if None.
    num_nodes : int, optional
        Number of rope nodes.
    length : float, optional
        Rest length [m].

    Returns
    -------
    TubeSurfaceDescriptor
    """
    return TubeSurfaceDescriptor(straight(num_nodes, length), R_ROPE,
                                 np.linspace(0, length, 21) if sample_s is None else sample_s)


def torus_reference(vertices, major_radius, minor_radius):
    """Analytic Gaussian and mean curvature of a torus around the z axis at each vertex.

    phi is the vertex's angle around the tube, 0 on the outer equator.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, 3)
        Points on the torus [m].
    major_radius : float
        Radius R of the tube's centre circle [m].
    minor_radius : float
        Tube radius r [m].

    Returns
    -------
    K : np.ndarray, shape (n,)
        Gaussian curvature cos(phi) / (r (R + r cos(phi))) [1/m^2].
    H : np.ndarray, shape (n,)
        Mean curvature (R + 2 r cos(phi)) / (2 r (R + r cos(phi))) [1/m].
    """
    phi = np.arctan2(vertices[:, 2], np.linalg.norm(vertices[:, :2], axis=1) - major_radius)
    cos_phi = np.cos(phi)
    return (cos_phi / (minor_radius * (major_radius + minor_radius * cos_phi)),
            (major_radius + 2 * minor_radius * cos_phi) / (2 * minor_radius * (major_radius + minor_radius * cos_phi)))


def turning_polyline(psi, num_nodes=N_ROPE, spacing=H_ROPE, lengths=None):
    """Planar polyline from the origin along +x, turning left by ``psi`` at every interior node.

    Parameters
    ----------
    psi : float or np.ndarray, shape (num_nodes - 2,)
        Turning angle at each interior node [rad].
    num_nodes : int, optional
        Number of nodes.
    spacing : float, optional
        Segment length [m], used when ``lengths`` is None.
    lengths : np.ndarray, shape (num_nodes - 1,), optional
        Individual segment lengths [m].

    Returns
    -------
    np.ndarray, shape (num_nodes, 3)
        Polyline nodes [m].
    """
    ang = np.r_[0.0, np.cumsum(np.broadcast_to(psi, (num_nodes - 2,)))]
    seg_len = np.full(num_nodes - 1, spacing) if lengths is None else lengths
    return np.r_[np.zeros((1, 3)),
                 np.cumsum(seg_len[:, None] * np.c_[np.cos(ang), np.sin(ang), np.zeros(num_nodes - 1)], 0)]


def window_turning(centerline, desc):
    """Exact per-sample reference for the tube identity, from the centreline polyline alone.

    For each sample: (t(hi) - t(lo)) divided by the current length of its rings' Voronoi cells, where t(lo) is the
    tangent of the segment entering the sample's first ring and t(hi) that of the segment leaving its last ring.

    Parameters
    ----------
    centerline : np.ndarray, shape (N, 3)
        Rope nodes [m].
    desc : TubeSurfaceDescriptor
        Descriptor that defines the samples and their rings.

    Returns
    -------
    np.ndarray, shape (num_samples, 3)
        Window mean of the curvature vector [1/m]; NaN for samples without rings or containing an end node.
    """
    segments = np.diff(centerline, axis=0)
    seg_len = np.linalg.norm(segments, axis=1)
    tangents = segments / seg_len[:, None]
    vor = np.r_[seg_len[0] / 2, 0.5 * (seg_len[:-1] + seg_len[1:]), seg_len[-1] / 2]
    rs = desc.groups[: len(centerline) * desc.n_around: desc.n_around]
    out = np.full((desc.num_samples, 3), np.nan)
    for sample in range(desc.num_samples):
        rings = np.flatnonzero(rs == sample)
        if len(rings) and rings[0] > 0 and rings[-1] < len(centerline) - 1:
            out[sample] = (tangents[rings[-1]] - tangents[rings[0] - 1]) / vor[rings].sum()
    return out


# ----------------------------------------------------------------------------------------------------------- sphere
def test_icosphere_area_curvature_and_gauss_bonnet():
    """Icospheres of radius 0.3 m: area, H = 1/R and K = 1/R^2 converge with refinement, the angle defects sum to
    4 pi (Gauss-Bonnet with chi = 2) and the vertex normals point outward."""
    radius, err = 0.3, {}
    for level in (3, 4):
        vertices, faces = icosphere(level, radius)
        fm = SurfaceMeshForms(faces, vertices)
        out = fm(vertices)
        err[level] = (abs(out["area"] / (4 * np.pi * radius**2) - 1),
                      np.mean(np.abs(out["H_v"] - 1 / radius)) * radius,
                      np.mean(np.abs(out["K_v"] - 1 / radius**2)) * radius**2)
        assert fm.closed and fm.euler_characteristic == 2
        assert abs(np.sum(2 * np.pi - (2 * np.pi - out["K_v"] * out["A_v"])) - 4 * np.pi) < 1e-9  # sum of defects
        assert abs(out["gauss_bonnet_residual"]) < 1e-9
        assert np.all(np.einsum("ij,ij->i", out["n_v"], vertices) > 0)  # outward
        print(f"\nicosphere level {level}: area {err[level][0]:.3%}, mean|H-1/R|R {err[level][1]:.4%}, "
              f"mean|K-1/R^2|R^2 {err[level][2]:.3%}")
    assert err[4][0] < 0.01 and err[4][1] < 0.02 and err[4][2] < 0.03
    assert all(err[4][metric] < err[3][metric] for metric in range(3))


def test_face_second_fundamental_form_on_sphere_and_ellipsoid():
    """Per-face second fundamental form: II = I / R, H = 1/R and K = 1/R^2 on a sphere. On an oblate ellipsoid (the
    boundary of a squashed ball) the vertex H and K at the pole are within 1% and the area within 0.2% of the
    analytic values."""
    radius = 0.3
    vertices, faces = icosphere(4, radius)
    fm = SurfaceMeshForms(faces, vertices)
    face_forms = fm.face_second_fundamental_form(vertices)
    assert np.allclose(face_forms["b_f"], face_forms["g_f"] / radius, atol=1e-9)  # II = I / R on a sphere
    assert (np.allclose(face_forms["H_f"], 1 / radius) and np.allclose(face_forms["K_f"], 1 / radius**2)
            and np.allclose(face_forms["beta_f"], 0))
    # A soft ball squashed into an oblate ellipsoid (semi-axes R, R, polar_axis): the boundary of a volumetric body.
    polar_axis = 0.6 * radius
    ellipsoid = vertices * np.array([1, 1, polar_axis / radius])
    out = fm(ellipsoid)
    pole = np.argmax(ellipsoid[:, 2])
    eccentricity = np.sqrt(1 - (polar_axis / radius) ** 2)
    area = 2 * np.pi * radius**2 * (1 + (1 - eccentricity**2) / eccentricity * np.arctanh(eccentricity))
    print(f"\nellipsoid pole: H {out['H_v'][pole] / (polar_axis / radius**2) - 1:+.3%}, "
          f"K {out['K_v'][pole] / (polar_axis**2 / radius**4) - 1:+.3%}; "
          f"area ratio {out['area'] / out['area_rest']:.5f} vs {area / (4 * np.pi * radius**2):.5f}")
    assert (abs(out["H_v"][pole] / (polar_axis / radius**2) - 1) < 0.01
            and abs(out["K_v"][pole] / (polar_axis**2 / radius**4) - 1) < 0.01)
    assert abs(out["area"] / area - 1) < 0.002 and abs(out["gauss_bonnet_residual"]) < 1e-9


def jittered_sphere(level, seed, radial=0.15, tangential=0.03):
    """Unit icosphere with randomly displaced vertices: an irregular closed mesh with many obtuse faces.

    Parameters
    ----------
    level : int
        Subdivision level of the icosphere.
    seed : int
        Seed of the displacements.
    radial : float, optional
        Standard deviation of the relative radial scaling of each vertex.
    tangential : float, optional
        Standard deviation of an additional displacement in every coordinate.

    Returns
    -------
    vertices : np.ndarray, shape (V, 3)
        Rest vertices on the unit sphere.
    faces : np.ndarray of int, shape (F, 3)
        Triangles.
    jittered : np.ndarray, shape (V, 3)
        Displaced vertices.
    """
    rng = np.random.default_rng(seed)
    vertices, faces = icosphere(level)
    return vertices, faces, (vertices * (1 + radial * rng.normal(size=(len(vertices), 1)))
                             + tangential * rng.normal(size=vertices.shape))


def test_mixed_voronoi_areas_on_obtuse_triangles():
    """Per-face loop of Meyer et al.'s (2003, Fig. 4) mixed-area rule, written independently, on a jittered sphere where
    about half the faces are obtuse. Mixed areas partition the surface exactly."""
    vertices, faces, jittered = jittered_sphere(3, 0)
    out = SurfaceMeshForms(faces, vertices)(jittered)
    mixed_area, n_obtuse = np.zeros(len(vertices)), 0
    for face in faces:
        corners = jittered[face]
        # The two edges leaving each corner, and the interior angle there.
        edges = [(corners[(corner + 1) % 3] - corners[corner], corners[(corner + 2) % 3] - corners[corner])
                 for corner in range(3)]
        ang = [np.arccos(first @ second / np.linalg.norm(first) / np.linalg.norm(second)) for first, second in edges]
        area = 0.5 * np.linalg.norm(np.cross(*edges[0]))
        if max(ang) > np.pi / 2:
            n_obtuse += 1
            mixed_area[face] += [area / 2 if angle > np.pi / 2 else area / 4 for angle in ang]
        else:  # Voronoi area of corner P: 1/8 (|PR|^2 cot Q + |PQ|^2 cot R), with Q and R the next two corners
            mixed_area[face] += [(edges[corner][1] @ edges[corner][1] / np.tan(ang[(corner + 1) % 3])
                                  + edges[corner][0] @ edges[corner][0] / np.tan(ang[(corner + 2) % 3])) / 8
                                 for corner in range(3)]
    print(f"\nmixed areas: {n_obtuse}/{len(faces)} obtuse faces, "
          f"max rel err {np.max(np.abs(out['A_v'] / mixed_area - 1)):.1e}")
    assert n_obtuse > len(faces) // 4
    np.testing.assert_allclose(out["A_v"], mixed_area, rtol=1e-12)
    assert abs(out["A_v"].sum() / out["area"] - 1) < 1e-12 and abs(out["A_bary_v"].sum() / out["area"] - 1) < 1e-12


def test_cotan_laplacian_is_the_area_gradient():
    """Mean curvature is the area gradient: |grad_x Area| = 2 |H| A_v, pointing along +n where H > 0 (central
    differences of the total area on an irregular closed mesh)."""
    vertices, faces, jittered = jittered_sphere(2, 1, radial=0.1)
    out = SurfaceMeshForms(faces, vertices)(jittered)

    def area(points):
        """Total area of the mesh with vertex positions ``points``."""
        return 0.5 * np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                             points[faces[:, 2]] - points[faces[:, 0]]), axis=1).sum()

    err = []
    for vertex in range(0, len(vertices), 7):
        gradient = np.zeros(3)
        for axis in range(3):
            dx = np.zeros_like(jittered)
            dx[vertex, axis] = 1e-6
            gradient[axis] = (area(jittered + dx) - area(jittered - dx)) / 2e-6
        err.append(abs(2 * abs(out["H_v"][vertex]) * out["A_v"][vertex] / np.linalg.norm(gradient) - 1))
        assert np.sign(gradient @ out["n_v"][vertex]) == np.sign(out["H_v"][vertex])
    print(f"\ncotan Laplacian vs finite-difference area gradient: max rel err {max(err):.1e}")
    assert max(err) < 1e-6


def test_gauss_image_moment():
    """Kn_v = first moment of each vertex's Gauss image (integral of K n over its star): sums to 0 on a closed surface
    (the Gauss map of a closed surface covers the sphere evenly), converges to K n A_v = (A_v / R^2) x / R on a
    sphere."""
    radius, err = 0.3, {}
    for level in (3, 4):
        vertices, faces = icosphere(level, radius)
        out = SurfaceMeshForms(faces, vertices)(vertices)
        ref = (out["A_v"] / radius**2)[:, None] * vertices / radius
        err[level] = np.max(np.linalg.norm(out["Kn_v"] - ref, axis=1) / np.linalg.norm(ref, axis=1))
        assert np.linalg.norm(out["Kn_v"].sum(0)) < 1e-12
    print(f"\nsphere Gauss-image moment vs K n A_v: max rel err level 3 {err[3]:.2e}, level 4 {err[4]:.2e}")
    assert err[4] < 0.005 and err[4] < err[3]
    vertices, faces, jittered = jittered_sphere(2, 2)
    assert np.linalg.norm(SurfaceMeshForms(faces, vertices)(jittered)["Kn_v"].sum(0)) < 1e-12


# ----------------------------------------------------------------------------------------------------------- tubes
def test_straight_tube():
    """Straight tube: vertex and face counts, interior H = 1/(2r) and K = 0 within 2%, lateral area 2 pi r L within
    1%, and a closed, consistently outward-oriented convex surface (Gauss-Bonnet with chi = 2)."""
    centerline = straight()
    vertices, faces, ring = tube_surface(centerline, R_ROPE)
    n_around = 12
    assert (vertices.shape == (N_ROPE * n_around + 2 * (3 * n_around + 1), 3)
            and len(faces) == 2 * (N_ROPE - 1) * n_around + 2 * (2 * 3 * n_around + n_around))
    assert ring.min() == 0 and ring.max() == N_ROPE - 1
    fm = SurfaceMeshForms(faces, vertices)
    out = fm(vertices)
    inner = (ring >= 2) & (ring <= N_ROPE - 3)
    lateral = np.all(faces < N_ROPE * n_around, axis=1)
    triangles = vertices[faces[lateral]]
    A_lat = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                                 axis=1).sum()
    H_err = np.mean(out["H_v"][inner]) * 2 * R_ROPE - 1
    print(f"\nstraight tube: interior H {H_err:+.3%} of 1/(2r), "
          f"max|K| r^2 {np.abs(out['K_v'][inner]).max() * R_ROPE**2:.1e}, "
          f"lateral area {A_lat / (2 * np.pi * R_ROPE * L_ROPE) - 1:+.1e}")
    assert np.all(np.abs(out["H_v"][inner] * 2 * R_ROPE - 1) < 0.02)
    assert np.all(np.abs(out["K_v"][inner]) * R_ROPE**2 < 0.02)
    assert abs(A_lat / (2 * np.pi * R_ROPE * L_ROPE) - 1) < 0.01
    assert fm.closed and abs(out["gauss_bonnet_residual"]) < 1e-9
    assert abs(np.sum(fm.rest["K_v"] * fm.rest["A_v"]) - 4 * np.pi) < 1e-9
    vol = np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6
    assert vol > 0  # outward orientation...
    half_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    assert len(np.unique(half_edges, axis=0)) == len(half_edges)  # ...and consistent (no flipped cap)
    assert np.all(out["H_v"] > 0)  # convex everywhere with outward normals, caps included


def test_bent_tube_area_is_exact_and_quads_are_planar():
    """Mitered rings: every segment is an exact prism, so area = 2 pi r * centreline length for any bending."""
    centerline = smooth_curve(4)
    vertices, faces, ring = tube_surface(centerline, R_ROPE)
    lateral = np.all(faces < N_ROPE * 12, axis=1)
    triangles = vertices[faces[lateral]]
    face_area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                                     axis=1)
    length = np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()
    assert abs(face_area.sum() / (2 * np.pi * R_ROPE * length) - 1) < 1e-12
    # The two triangles of each lateral quad are coplanar (zero dihedral on the diagonal); faces come per segment as
    # n triangles (a_j, a_j+1, b_j) then n triangles (a_j+1, b_j+1, b_j).
    nf = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    nf = (nf / np.linalg.norm(nf, axis=1, keepdims=True)).reshape(N_ROPE - 1, 2, 12, 3)
    assert np.min(np.einsum("ijk,ijk->ij", nf[:, 0], nf[:, 1])) > 1 - 1e-12


@pytest.mark.parametrize("radius", [0.05, 0.021])
def test_torus_patch_per_vertex_K_and_H(radius):
    """Tube around a circular arc = torus patch. Fine node spacing (4 mm) tests the estimators; the rope's own 13.5 mm
    spacing (38.6 deg per node at R = 2.1 cm) is checked with looser bounds."""
    for spacing, k_tol, h_tol in ((0.004, 0.05, 0.03), (H_ROPE, 0.05, 0.07)):
        ang = 1.5 * np.pi
        num_nodes = int(round(radius * ang / spacing)) + 1
        vertices, faces, ring = tube_surface(arc(radius, num_nodes, ang), R_ROPE)
        out = SurfaceMeshForms(faces, vertices)(vertices)
        # Lateral vertices away from the end caps.
        sel = (ring >= 2) & (ring <= num_nodes - 3) & (np.arange(len(vertices)) < num_nodes * 12)
        K_ref, H_ref = torus_reference(vertices[sel], radius, R_ROPE)
        k_err = np.mean(np.abs(out["K_v"][sel] - K_ref)) / np.abs(K_ref).max()
        h_err = np.mean(np.abs(out["H_v"][sel] - H_ref) / H_ref)
        print(f"\ntorus R={radius} spacing {spacing * 1e3:.1f} mm: per-vertex K err {k_err:.2%} of max|K|, "
              f"H err {h_err:.2%}")
        assert k_err < k_tol and h_err < h_tol


def unstretched_arc(radius, ccw=True, num_nodes=N_ROPE, spacing=H_ROPE):
    """Rope nodes on a circle with every chord equal to the rest segment length (no stretch).

    Parameters
    ----------
    radius : float
        Circle radius [m].
    ccw : bool, optional
        Left-turning if True, right-turning otherwise.
    num_nodes : int, optional
        Number of nodes.
    spacing : float, optional
        Chord length [m].

    Returns
    -------
    np.ndarray, shape (num_nodes, 3)
        Nodes [m] in the xy plane.
    """
    return arc(radius, num_nodes, (num_nodes - 1) * 2 * np.arcsin(spacing / (2 * radius)), ccw)


@pytest.mark.parametrize("radius", [0.05, 0.021])
def test_arc_curvature_vector_and_sign(radius):
    """Per-sample curvature vector on an arc of radius R. With one ring per sample it is (1/R) toward the centre (the
    discrete curvature 2 sin(psi/2) / h of nodes on a circle is exactly 1/R). With the default 4 cm samples it is the
    window MEAN of kappa N, which is shorter than 1/R because N turns inside the window (2 sin(dth/2) / dth); the
    reference is the exact window mean of the rope polyline."""
    rest = straight()
    sgn = {True: 1, False: -1}
    for ccw in (True, False):
        centerline = unstretched_arc(radius, ccw)
        dens = TubeSurfaceDescriptor(rest, R_ROPE, np.linspace(0, L_ROPE, N_ROPE))(centerline)
        curv_vec = dens["curvature_vector"][1:-1]
        err1 = np.max(np.linalg.norm(curv_vec - -centerline[1:-1] / radius**2, axis=1)) * radius
        assert err1 < 1e-9 and np.all(np.abs(dens["signed_planar_curvature"][1:-1] * radius - sgn[ccw]) < 1e-9)
        desc = rope_descriptor()
        out = desc(centerline)
        ref, ref_spc = polyline_window_curvature(centerline, desc.sample_window)
        err2 = np.max(np.linalg.norm(out["curvature_vector"] - ref, axis=1) / np.linalg.norm(ref, axis=1))
        spc_err = np.max(np.abs(out["signed_planar_curvature"] / ref_spc - 1))
        kabs = out["K_abs"][1:-1] * np.pi * R_ROPE / 2 * radius
        print(f"\narc R={radius} ccw={ccw}: one ring/sample max err {err1:.1e}; 4 cm samples vs window mean {err2:.1e} "
              f"(signed {spc_err:.1e}; window mean = {np.mean(np.linalg.norm(ref[1:-1], axis=1)) * radius:.3f}/R); "
              f"K_abs pi r/2 = {kabs.mean():.3f}/R")
        assert err2 < 1e-9 and spc_err < 1e-9 and np.all(np.sign(out["signed_planar_curvature"]) == sgn[ccw])
        assert np.all(np.abs(kabs - 1) < 0.04)  # K_abs = 2 |kappa| / (pi r): unsigned, no averaging loss


@pytest.mark.parametrize("deg", [1.0, 37.5, 80.0, 104.0])
def test_tube_identity_is_exact_for_any_turn(deg):
    """Discrete tube identity: the Gauss image of a ring's star is the band between the great circles normal to the two
    segments, so integral K n over ring k = -pi (t_k - t_{k-1}) exactly, and with the 2 pi r perimeter the ring area is
    2 pi r times its Voronoi length. Holds up to the fold limit 2 rho tan(psi/2) = h (106 degrees per node here)."""
    centerline = turning_polyline(np.radians(deg))
    vertices, faces, ring = tube_surface(centerline, R_ROPE)
    out = SurfaceMeshForms(faces, vertices)(vertices)
    tangents = np.diff(centerline, axis=0) / H_ROPE
    lateral = np.arange(len(vertices)) < N_ROPE * 12
    err = max(np.linalg.norm(out["Kn_v"][lateral & (ring == node)].sum(0)
                             + np.pi * (tangents[node] - tangents[node - 1])) / np.pi
              / np.linalg.norm(tangents[node] - tangents[node - 1]) for node in range(1, N_ROPE - 1))
    desc = rope_descriptor()
    curv_vec = desc(centerline)["curvature_vector"]
    err2 = np.max(np.linalg.norm(curv_vec - window_turning(centerline, desc), axis=1)
                  / np.linalg.norm(window_turning(centerline, desc), axis=1))
    print(f"\ntube identity at {deg} deg/node: ring moment rel err {err:.1e}, per-sample curvature_vector {err2:.1e}")
    assert err < 1e-12 and err2 < 1e-12


def test_frames_are_rotation_minimizing():
    """Ring frames = discrete parallel transport: each step is the minimal rotation t_i -> t_{i+1} (independent
    Rodrigues implementation), so the frame does not twist about the tangent. On a finely sampled helix the frame turns
    against the Frenet normal at exactly -torsion, the continuous rotation-minimizing frame (Bishop 1975)."""
    sharp = turning_polyline(np.radians(100.0) * np.sin(np.arange(N_ROPE - 2)))  # left/right turns up to 100 deg
    sharp[:, 2] = 0.002 * np.cos(np.arange(N_ROPE))  # ... and out of plane
    for centerline in (smooth_curve(3), sharp):
        tangents, _, _, normals = sf._frames(centerline)
        # Rodrigues rotation of the previous normal about before x after, by the angle between the two tangents.
        ref = [normals[0]]
        for before, after in zip(tangents[:-1], tangents[1:]):
            ax = np.cross(before, after)
            sine, unit_axis = np.linalg.norm(ax), ax / max(np.linalg.norm(ax), 1e-300)
            prev, ang = ref[-1], np.arctan2(sine, before @ after)
            ref.append(prev * np.cos(ang) + np.cross(unit_axis, prev) * np.sin(ang)
                       + unit_axis * (unit_axis @ prev) * (1 - np.cos(ang)))
        assert np.abs(normals - np.array(ref)).max() < 1e-12
        assert (np.abs(np.einsum("ij,ij->i", normals, tangents)).max() < 1e-12
                and np.abs(np.linalg.norm(normals, axis=1) - 1).max() < 1e-12)
    # Helix (radius cos th, radius sin th, rise th) with torsion tau = rise / (radius^2 + rise^2).
    radius, rise = 0.03, 0.01
    tau = rise / (radius * radius + rise * rise)
    th = np.linspace(0, 4 * np.pi, 4001)
    tangents, _, _, normals = sf._frames(np.stack([radius * np.cos(th), radius * np.sin(th), rise * th], 1))
    tm = 0.5 * (th[1:] + th[:-1])
    nF = -np.c_[np.cos(tm), np.sin(tm), np.zeros_like(tm)]  # Frenet normal at the segment midpoints
    ang = np.unwrap(np.arctan2(np.einsum("ij,ij->i", normals, np.cross(tangents, nF)),
                               np.einsum("ij,ij->i", normals, nF)))
    rate = np.polyfit(np.hypot(radius, rise) * tm, ang, 1)[0]
    print(f"\nhelix frame twist rate {rate:.6f} vs -torsion {-tau:.6f}")
    assert abs(rate / -tau - 1) < 1e-4


def test_helix_curvature():
    """Helix with chords equal to the rest segment length: the one-ring-per-sample curvature vector matches the smooth
    helix within 1% and points to the axis, the 4 cm samples equal the exact window mean of the polyline, K_abs gives
    |kappa| within 5%, and the area ratio stays 1."""
    from scipy.optimize import brentq

    # Helix (radius cos t, radius sin t, rise t): curvature radius / (radius^2 + rise^2) = 30 1/m, torsion
    # rise / (radius^2 + rise^2) = 10 1/m.
    radius, rise = 0.03, 0.01
    kappa = radius / (radius * radius + rise * rise)
    # Parameter step whose chord equals the rest segment length.
    dt = brentq(lambda step: np.hypot(2 * radius * np.sin(step / 2), rise * step) - H_ROPE, 1e-6, 1.0)
    phase = dt * np.arange(N_ROPE)
    centerline = np.stack([radius * np.cos(phase), radius * np.sin(phase), rise * phase], 1)
    dens = TubeSurfaceDescriptor(straight(), R_ROPE, np.linspace(0, L_ROPE, N_ROPE))(centerline)
    curv_vec = dens["curvature_vector"][1:-1]
    axis_dir = -np.c_[np.cos(phase[1:-1]), np.sin(phase[1:-1]), 0 * phase[1:-1]]
    err1 = np.max(np.abs(np.linalg.norm(curv_vec, axis=1) / kappa - 1))  # polyline (13.5 mm chords) vs smooth helix
    assert err1 < 0.01 and np.min(np.einsum("ij,ij->i", curv_vec, axis_dir) / np.linalg.norm(curv_vec, axis=1)) > 0.99
    desc = rope_descriptor()
    out = desc(centerline)
    ref = polyline_window_curvature(centerline, desc.sample_window)[0]
    err2 = np.max(np.linalg.norm(out["curvature_vector"] - ref, axis=1) / np.linalg.norm(ref, axis=1))
    err3 = np.max(np.abs(out["K_abs"][1:-1] * np.pi * R_ROPE / 2 / kappa - 1))
    print(f"\nhelix: one ring/sample |kN| err {err1:.2%}; 4 cm samples vs window mean {err2:.1e} (mean = "
          f"{np.mean(np.linalg.norm(ref[1:-1], axis=1)) / kappa:.3f} kappa); K_abs -> |kappa| err {err3:.2%}; "
          f"area ratio {out['area_ratio'].min():.12f}")
    assert err2 < 1e-9 and err3 < 0.05 and np.allclose(out["area_ratio"], 1, atol=1e-9)


def test_rest_and_end_samples_are_clean():
    """At rest every curvature statistic is that of a straight tube; the end rings (cap junctions) are excluded."""
    desc = rope_descriptor()
    out = desc(straight())
    assert np.allclose(out["area_ratio"], 1, atol=1e-12) and np.allclose(out["K_abs"] * R_ROPE**2, 0, atol=1e-9)
    assert np.allclose(out["curvature_vector"], 0, atol=1e-6) and np.allclose(out["H"], out["H_rest"])
    assert abs(out["gauss_bonnet_residual"]) < 1e-9 and abs(out["willmore"] / out["willmore_rest"] - 1) < 1e-12
    assert (np.all(desc.groups[desc.ring_of_vertex == 0] == -1)
            and np.all(desc.groups[desc.ring_of_vertex == N_ROPE - 1] == -1))
    np.testing.assert_allclose(desc.sample_window[1:-1], np.stack([np.arange(1, 20) - 0.5, np.arange(1, 20) + 0.5], 1)
                               * L_ROPE / 20, atol=1e-12)
    assert np.allclose(desc.sample_window[[0, -1]],
                       [[0.5 * H_ROPE, 1.5 * H_ROPE], [L_ROPE - 1.5 * H_ROPE, L_ROPE - 0.5 * H_ROPE]])
    for key in ("area_ratio", "H", "H_rest", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2",
                "signed_planar_curvature"):
        assert out[key].shape == (21,), key
    for key in ("normal", "moment", "curvature_vector", "tangent"):
        assert out[key].shape == (21, 3), key
    for key in ("area_ratio_total", "willmore", "willmore_rest", "gauss_bonnet_residual", "n_invalid"):
        assert np.isscalar(out[key]), key
    assert out["valid"].shape == (21,) and out["valid"].all() and out["n_invalid"] == 0 and np.all(out["fold_max"] == 0)


# ----------------------------------------------------------------------------------------------------------- invariance
@pytest.mark.parametrize("lam", [1.05, 0.97])
def test_uniform_stretch_area_ratio(lam):
    """Uniform stretch by lam: area_ratio = lam on a straight rope, the exact axial stretch of the segments on a curved
    rope, and the lateral faces stretch along the axis only."""
    desc = rope_descriptor()
    out = desc(straight() * lam)
    assert np.all(np.abs(out["area_ratio"] / lam - 1) < 0.005) and abs(out["area_ratio_total"] / lam - 1) < 0.01
    # Curved rope: area ratio = the axial stretch of the centreline segments, exactly, whatever the bending.
    centerline = smooth_curve(1)
    base = desc(centerline)["area_ratio"]
    seg_len = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    vor = np.r_[seg_len[0] / 2, 0.5 * (seg_len[:-1] + seg_len[1:]), seg_len[-1] / 2] / H_ROPE
    rs = desc.groups[: N_ROPE * 12 : 12]
    ref = np.array([vor[rs == sample].sum() / np.sum(rs == sample) for sample in range(21)])
    assert (np.max(np.abs(base - ref)) < 1e-9
            and np.max(np.abs(desc(centerline * lam)["area_ratio"] - lam * ref)) < 1e-9)
    # Radius fixed: lateral faces stretch along the axis only (principal stretches lam and 1).
    desc(straight() * lam)
    lateral = np.all(desc.faces < N_ROPE * 12, axis=1)
    assert np.allclose(np.sort(desc.last_out["stretch_f"][lateral], axis=1), np.sort([lam, 1.0]), atol=1e-9)


def test_rigid_motion_invariance():
    """Under a rigid motion the scalar outputs are invariant and the vector outputs rotate with it; the signed planar
    curvature is invariant under rotations about world z; the same holds for the generic mesh forms."""
    rotation, shift = random_rotation(7), np.array([0.3, -1.2, 0.5])
    centerline = smooth_curve(2)
    desc = rope_descriptor()
    out, moved = desc(centerline), desc(centerline @ rotation.T + shift)
    for key in ("area_ratio", "H", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2", "area_ratio_total", "willmore"):
        np.testing.assert_allclose(moved[key], out[key], rtol=1e-8, atol=1e-8, err_msg=key)
    for key in ("curvature_vector", "moment", "tangent"):
        np.testing.assert_allclose(moved[key], out[key] @ rotation.T, rtol=1e-8, atol=1e-8, err_msg=key)
    # The signed planar curvature is measured about world z (task frame): invariant under rotations about z.
    cos_z, sin_z = np.cos(0.8), np.sin(0.8)
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    np.testing.assert_allclose(desc(centerline @ rot_z.T + shift)["signed_planar_curvature"],
                               out["signed_planar_curvature"], atol=1e-8)
    # The generic mesh forms themselves.
    vertices, faces, _ = tube_surface(centerline, R_ROPE)
    fm = SurfaceMeshForms(faces, vertices)
    mesh_out, mesh_moved = fm(vertices), fm(vertices @ rotation.T + shift)
    for key in ("A_v", "H_v", "K_v", "k1_v", "k2_v", "J_f", "E_f", "stretch_f"):
        np.testing.assert_allclose(mesh_moved[key], mesh_out[key], rtol=1e-8, atol=1e-8, err_msg=key)
    np.testing.assert_allclose(mesh_moved["n_v"], mesh_out["n_v"] @ rotation.T, atol=1e-10)
    np.testing.assert_allclose(mesh_moved["Kn_v"], mesh_out["Kn_v"] @ rotation.T, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rigid_motion_with_nonuniform_stretch(seed):
    """A bent 3-D rope whose segments are stretched non-uniformly (-3% .. +5%), then moved rigidly: area_ratio is the
    exact per-sample stretch, curvature_vector the exact window turning per current length, both
    invariant/equivariant."""
    rng = np.random.default_rng(seed)
    lam = 1 + np.interp(np.arange(N_ROPE - 1), np.linspace(0, N_ROPE - 2, 6), rng.uniform(-0.03, 0.05, 6))
    psi = np.radians(40.0) * np.sin(np.linspace(0, 3 * np.pi, N_ROPE - 2) + rng.uniform(0, 6))
    centerline = turning_polyline(psi, lengths=H_ROPE * lam)
    # Out of plane (the stretch changes).
    centerline[:, 2] = 0.01 * np.sin(np.linspace(0, 2 * np.pi, N_ROPE) + rng.uniform(0, 6))
    desc = rope_descriptor()
    out = desc(centerline)
    seg_len = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    vor = np.r_[seg_len[0] / 2, 0.5 * (seg_len[:-1] + seg_len[1:]), seg_len[-1] / 2]
    rs = desc.groups[: N_ROPE * 12: 12]
    ref = np.array([vor[rs == sample].sum() / (np.sum(rs == sample) * H_ROPE) for sample in range(21)])
    np.testing.assert_allclose(out["area_ratio"], ref, rtol=1e-12)
    np.testing.assert_allclose(out["curvature_vector"], window_turning(centerline, desc), rtol=1e-9, atol=1e-9)
    rotation, shift = random_rotation(seed + 10), rng.normal(size=3)
    moved = desc(centerline @ rotation.T + shift)
    for key in ("area_ratio", "H", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2", "area_ratio_total", "willmore"):
        np.testing.assert_allclose(moved[key], out[key], rtol=1e-8, atol=1e-8, err_msg=key)
    for key in ("curvature_vector", "moment", "tangent"):
        np.testing.assert_allclose(moved[key], out[key] @ rotation.T, rtol=1e-8, atol=1e-8, err_msg=key)
    print(f"\nnon-uniform stretch seed {seed}: area_ratio {out['area_ratio'].min():.4f}..{out['area_ratio'].max():.4f} "
          f"(exact), max |kappa| window {np.linalg.norm(out['curvature_vector'], axis=1).max():.1f} 1/m")


def test_sample_windows_tile_the_rope():
    """For any ascending sample set: every interior ring belongs to its nearest sample, the windows of non-empty samples
    are contiguous (no gaps, no overlaps) and cover the rope between the excluded end rings; empty samples give NaN."""
    rng = np.random.default_rng(3)
    node_arclength = np.arange(N_ROPE) * H_ROPE
    for _ in range(100):
        sample_s = np.sort(rng.uniform(0, L_ROPE, rng.integers(2, 40)))
        desc = TubeSurfaceDescriptor(straight(), R_ROPE, sample_s)
        rs = desc.groups[: N_ROPE * 12: 12]
        assert np.all(rs[1:-1] == np.argmin(np.abs(node_arclength[1:-1, None] - sample_s[None]), axis=1))
        ok = np.isin(np.arange(len(sample_s)), rs)
        windows = desc.sample_window[ok]
        assert np.all(np.isnan(desc.sample_window[~ok])) and np.all(windows[1:, 0] == windows[:-1, 1])
        assert np.isclose(windows[0, 0], 0.5 * H_ROPE if rs[0] < 0 else 0.0)
        assert np.isclose(windows[-1, 1], L_ROPE - 0.5 * H_ROPE if rs[-1] < 0 else L_ROPE)
        out = desc(smooth_curve(0))
        assert np.all(np.isnan(out["area_ratio"][~ok])) and np.all(np.isfinite(out["area_ratio"][ok]))


def test_group_stats_with_empty_groups():
    """Leading, interior and trailing empty groups: every statistic of the non-empty groups equals a brute-force
    per-group computation (K_min/K_max use reduceat, which must not drop the last vertex of a group), empty ones NaN."""
    vertices, faces, jittered = jittered_sphere(2, 4)
    fm = SurfaceMeshForms(faces, vertices)
    out = fm(jittered)
    labels = np.array([1, 2, 4])[np.arange(len(vertices)) % 3]
    labels[::17] = -1
    st = fm.group_stats(out, labels, 6)
    for group in (1, 2, 4):
        members = labels == group
        area = out["A_bary_v"][members].sum()
        assert st["K_min"][group] == out["K_v"][members].min() and st["K_max"][group] == out["K_v"][members].max()
        assert np.isclose(st["area_ratio"][group], area / fm.A0_bary_v[members].sum())
        assert np.isclose(st["H"][group], np.sum(out["H_v"][members] * out["A_v"][members]) / area)
        assert np.allclose(st["moment"][group], out["Kn_v"][members].sum(0) / area)
    for group in (0, 3, 5):
        assert np.isnan(st["K_min"][group]) and np.isnan(st["K_max"][group]) and np.isnan(st["area_ratio"][group])


# ----------------------------------------------------------------------------------------------------------- cloth
def test_object_agnostic_cloth_bending_and_stretch():
    """The same SurfaceMeshForms on an open cloth sheet: isometric bending into a cylinder (interior H = -1/(2R) within
    3%, K = 0, face area ratio = chord / arc) and an in-plane stretch (area ratio, principal stretches and right
    Cauchy-Green invariants exact)."""
    flat, faces, _ = sheet()
    fm = SurfaceMeshForms(faces, flat)
    assert not fm.closed and fm.boundary_v.sum() == 2 * (41 + 21) - 4
    # Isometric bending into a cylinder of radius R (curls toward +z, the side the normal points to: H = -1/(2R)).
    radius = 0.1
    coord_u, coord_v = flat[:, 0], flat[:, 1]
    bent = np.stack([radius * np.sin(coord_u / radius), coord_v, radius * (1 - np.cos(coord_u / radius))], 1)
    out = fm(bent)
    inner = ~fm.boundary_v
    h_err = np.max(np.abs(out["H_v"][inner] * -2 * radius - 1))
    print(f"\ncloth cylinder: interior H err {h_err:.2e}, "
          f"max|K| R^2 {np.abs(out['K_v'][inner]).max() * radius * radius:.1e}, "
          f"area ratio {out['area'] / out['area_rest']:.6f}")
    assert h_err < 0.03 and np.all(np.abs(out["K_v"][inner]) * radius * radius < 1e-6)
    # Chord / arc per column of faces.
    assert np.allclose(out["J_f"], 2 * radius * np.sin(0.3 / 40 / (2 * radius)) / (0.3 / 40))
    face_forms = fm.face_second_fundamental_form(bent)  # faces away from the boundary (one-sided normals there)
    inner_faces = ~np.any(fm.boundary_v[faces], axis=1)
    assert (np.allclose(face_forms["H_f"][inner_faces], -0.5 / radius, rtol=0.03)
            and np.max(np.abs(face_forms["K_f"][inner_faces])) * radius * radius < 1e-6)
    # In-plane stretch (l1, l2): area ratio l1 l2 and principal stretches (l1, l2), exactly.
    l1, l2 = 1.08, 0.95
    out = fm(flat * np.array([l1, l2, 1.0]))
    assert np.allclose(out["J_f"], l1 * l2) and np.allclose(out["area_ratio_v"], l1 * l2)
    assert np.allclose(out["stretch_f"], [l1, l2])
    Cg = np.eye(2) + 2 * out["E_f"]  # right Cauchy-Green tensor per face: invariants l1^2 + l2^2 and (l1 l2)^2
    assert np.allclose(np.trace(Cg, axis1=1, axis2=2), l1**2 + l2**2) and np.allclose(np.linalg.det(Cg), (l1 * l2) ** 2)
    stats = fm.group_stats(out, np.zeros(len(flat), int), 1)
    assert np.isclose(stats["area_ratio"][0], l1 * l2) and np.allclose(out["H_v"][inner], 0, atol=1e-9)


def test_cloth_random_deformation_and_rigid_motion():
    """Smooth non-uniform 3-D deformation of the sheet plus a random rigid motion. Per face, the metric g must have the
    invariants of C = F^T F with F = Ds Dm^-1 built independently in the sheet's own (x, y) rest coordinates: tr C,
    det C = J^2, principal stretches = sqrt(eig C)."""
    flat, faces, _ = sheet()
    fm = SurfaceMeshForms(faces, flat)
    coord_u, coord_v = flat[:, 0], flat[:, 1]
    deformed = np.stack([coord_u + 0.02 * np.sin(9 * coord_v) + 0.3 * coord_u * coord_v,
                         coord_v * (1 + 0.1 * np.cos(7 * coord_u)), 0.04 * np.sin(11 * coord_u) * coord_v], 1)
    rotation, shift = random_rotation(5), np.array([0.2, -0.4, 1.0])
    out = fm(deformed @ rotation.T + shift)
    # Edge matrices per face: rest (m, 2, 2) in the sheet's (x, y) coordinates, deformed (m, 3, 2).
    Dm = np.stack([flat[faces[:, 1], :2] - flat[faces[:, 0], :2], flat[faces[:, 2], :2] - flat[faces[:, 0], :2]], 2)
    Ds = np.stack([deformed[faces[:, 1]] - deformed[faces[:, 0]], deformed[faces[:, 2]] - deformed[faces[:, 0]]], 2)
    Fd = Ds @ np.linalg.inv(Dm)
    Cg = np.einsum("fia,fib->fab", Fd, Fd)
    metric = np.eye(2) + 2 * out["E_f"]
    np.testing.assert_allclose(np.trace(metric, axis1=1, axis2=2), np.trace(Cg, axis1=1, axis2=2), rtol=1e-10)
    np.testing.assert_allclose(np.linalg.det(metric), np.linalg.det(Cg), rtol=1e-9)
    np.testing.assert_allclose(out["J_f"] ** 2, np.linalg.det(Cg), rtol=1e-9)
    np.testing.assert_allclose(out["stretch_f"], np.sqrt(np.linalg.eigvalsh(Cg)[:, ::-1]), rtol=1e-8)
    print(f"\ncloth random deformation: J_f {out['J_f'].min():.3f}..{out['J_f'].max():.3f}, per-face metric invariants "
          f"match an independent F = Ds Dm^-1; area ratio {out['area'] / out['area_rest']:.4f}")
    assert abs(out["gauss_bonnet_residual"]) < 1e-9


# ---------------------------------------------------------------------------------------------------------- consistency
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tube_curvature_matches_centreline_curvature(seed):
    """curvature_vector = -2 r * moment of the tube vs the centreline's discrete curvature vectors averaged over the
    same rings (weighted by Voronoi length): equal up to round-off, by the exact discrete tube identity."""
    centerline = smooth_curve(seed)
    desc = rope_descriptor()
    out = desc(centerline)
    kc = curve_curvature_vectors(centerline)
    seg_len = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    vor = np.r_[seg_len[0] / 2, 0.5 * (seg_len[:-1] + seg_len[1:]), seg_len[-1] / 2]
    rs = desc.groups[: N_ROPE * 12 : 12]
    ref = np.array([(kc[rs == sample] * vor[rs == sample, None]).sum(0) / vor[rs == sample].sum()
                    for sample in range(21)])
    mag = np.linalg.norm(ref, axis=1)
    err = np.linalg.norm(out["curvature_vector"] - ref, axis=1) / np.maximum(mag, 1e-12)
    print(f"\nrandom curve {seed}: max |kappa| {np.linalg.norm(kc, axis=1).max():.1f} 1/m, max rel err (|k|>5) "
          f"{err[mag > 5].max():.1e}, max abs err {np.abs(out['curvature_vector'] - ref).max():.1e} 1/m")
    assert np.linalg.norm(kc, axis=1).max() > 30 and err[mag > 5].max() < 1e-9
    assert np.abs(out["curvature_vector"] - ref).max() < 1e-9


def test_polyline_helpers():
    """signed_planar_curvature_of_polyline, curve_curvature_vectors and polyline_window_curvature on a finely sampled
    semicircle: 1/R with the turning sign, the window mean 2 sin(dth/2) / dth of a window turning by dth, zero for a
    window beyond the polyline, and no change from a repeated point."""
    radius = 0.05
    for ccw, sgn in ((True, 1), (False, -1)):
        points = arc(radius, 200, np.pi, ccw)
        signed = signed_planar_curvature_of_polyline(points[:, :2])
        assert np.allclose(signed[1:-1], sgn / radius, rtol=1e-9) and signed[0] == 0 and signed[-1] == 0
        assert np.allclose(np.linalg.norm(curve_curvature_vectors(points)[1:-1], axis=1), 1 / radius, rtol=1e-9)
        arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        kv, spc = polyline_window_curvature(points, np.array([[0.01, 0.05], [0.1, 0.12]]))
        dth = np.array([0.04, 0.02]) / radius
        assert np.allclose(spc * radius, sgn * np.sin(dth / 2) / (dth / 2), rtol=2e-3)
    kv, spc = polyline_window_curvature(points, np.array([[arclength[-1] + 0.1, arclength[-1] + 0.2]]))
    assert np.all(kv == 0) and spc[0] == 0
    windows = np.array([[0.01, 0.05], [0.1, 0.12]])  # a repeated point (zero-length segment) changes nothing
    np.testing.assert_allclose(polyline_window_curvature(np.insert(points, 50, points[50], axis=0), windows)[0],
                               polyline_window_curvature(points, windows)[0], atol=1e-12)


def test_target_profile_from_route_matches_rope_on_route():
    """polyline_window_curvature(route, sample_window) is the per-sample target of signed_planar_curvature: a rope
    lying exactly on the routing path (sampled at its 13.5 mm node spacing) reproduces it."""
    from rope_routing.env.grasp_route_env import make_task

    task = make_task()
    route = task.route
    arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    num_nodes = int(arclength[-1] / H_ROPE) + 1
    node_arclength = np.arange(num_nodes) * H_ROPE
    centerline = np.stack([np.interp(node_arclength, arclength, route[:, 0]),
                           np.interp(node_arclength, arclength, route[:, 1]), np.full(num_nodes, 0.005)], 1)
    desc = TubeSurfaceDescriptor(np.c_[node_arclength, np.zeros((num_nodes, 2))], R_ROPE,
                                 np.linspace(0, node_arclength[-1], 16))
    out = desc(centerline)
    _, target = polyline_window_curvature(route, desc.sample_window)
    err = np.abs(out["signed_planar_curvature"] - target)
    print(f"\nroute target profile: max {np.abs(target).max():.1f} 1/m, max |rope - target| {err.max():.2f} 1/m")
    assert err.max() < 0.1 * np.abs(target).max()


# ----------------------------------------------------------------------------------------------------------- speed
def test_performance():
    """One descriptor call (61 nodes, 12-gon, 3 cap rings) takes less than 3 ms, also with two folds replaced by the
    closed forms (best of 5 runs of 50 calls)."""
    desc = rope_descriptor()
    centerline = smooth_curve(0)
    desc(centerline)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            desc(centerline)
        best = min(best, (time.perf_counter() - t0) / 50)
    print(f"\nTubeSurfaceDescriptor call (N=61, n_around=12, cap_rings=3, {len(desc.faces)} faces): "
          f"{best * 1e3:.3f} ms")
    assert best < 3e-3
    centerline = hairpin(150.0, (10, 40))  # folded: closed forms substituted for 4 samples and the globals
    assert desc(centerline)["n_invalid"] == 4
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            desc(centerline)
        best = min(best, (time.perf_counter() - t0) / 50)
    print(f"... with two folds: {best * 1e3:.3f} ms")
    assert best < 3e-3


def test_ring_radius_area_preserving():
    """With area_preserving the 12-gon's perimeter equals the circle's 2 pi r (the tube then has the smooth tube's
    lateral area); without it the radius is returned unchanged."""
    assert np.isclose(2 * 12 * ring_radius(1.0, 12) * np.sin(np.pi / 12), 2 * np.pi)
    assert ring_radius(1.0, 12, area_preserving=False) == 1.0


# ----------------------------------------------------------------------------------------------------------- robustness
def test_degenerate_rope_input():
    """Coincident nodes, a 180-degree fold, a 120+ degree kink, a vertical rope and a fully collapsed rope: no exception
    and no numpy warning. Valid geometry gives finite outputs; NaN input gives NaN output (the env must sanitize)."""
    desc = rope_descriptor()
    cases = {}
    centerline = straight().copy()
    centerline[30] = centerline[31]
    cases["coincident"] = centerline
    centerline = straight().copy()
    centerline[31:] = 2 * centerline[30] - centerline[31:] + np.array([0, 1e-12, 0])
    cases["fold"] = centerline
    cases["kink"] = turning_polyline(np.radians(np.where(np.arange(N_ROPE - 2) == 30, 130.0, 0.0)))
    cases["vertical"] = np.c_[np.zeros((N_ROPE, 2)), np.linspace(0, L_ROPE, N_ROPE)]
    cases["collapsed"] = np.zeros((N_ROPE, 3))
    centerline = smooth_curve(0)
    centerline[10, 1] = np.nan
    cases["nan"] = centerline
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = {case: desc(shape) for case, shape in cases.items()}
    for case in ("coincident", "fold", "kink", "vertical"):
        assert all(np.all(np.isfinite(value)) for value in out[case].values()), case
    assert np.isnan(out["nan"]["area_ratio"]).any() and np.all(np.isfinite(out["nan"]["H_rest"]))
    np.testing.assert_allclose(out["vertical"]["curvature_vector"], 0, atol=1e-9)
    np.testing.assert_allclose(out["vertical"]["area_ratio"], 1, atol=1e-12)
    # Folds are flagged: a zero-length segment, the 180-degree fold and the kink fold the tube; NaN is never valid.
    assert out["vertical"]["valid"].all() and out["vertical"]["n_invalid"] == 0
    for case in ("coincident", "fold", "kink"):
        assert out[case]["n_invalid"] >= 1 and not out[case]["valid"][10], case
    assert not out["nan"]["valid"][3]


# ----------------------------------------------------------------------------------------------------------- folds
PER_SAMPLE = ("area_ratio", "H", "H_rest", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2", "signed_planar_curvature",
              "normal", "moment", "curvature_vector", "tangent")
GLOBALS = ("area_ratio_total", "willmore", "willmore_rest", "gauss_bonnet_residual")
CLOSED = ("area_ratio", "H", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2", "signed_planar_curvature", "moment",
          "curvature_vector")


def legacy_call(desc, centerline):
    """Frozen copy of TubeSurfaceDescriptor.__call__ before fold handling (mesh values for every sample).

    Parameters
    ----------
    desc : TubeSurfaceDescriptor
        Descriptor whose mesh, samples and cached interpolation weights are used.
    centerline : array_like, shape (N, 3)
        Rope nodes [m].

    Returns
    -------
    dict
        The descriptor's outputs except "valid", "fold_max" and "n_invalid".
    """
    points = np.asarray(centerline, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        vertices, tangents, bisectors = sf._tube_vertices(points, desc.rho, desc._cs, desc._sn, *desc._cap)
        out = desc.forms(vertices)
        st = desc.forms.group_stats(out, desc.groups, desc.num_samples)
        curv = -2.0 * desc.radius * st["moment"]
        tan = sf._slerp_tangent(tangents, desc._ti, desc._tf)
        tw = sf._slerp_tangent(tangents, desc._wi, desc._wf)
    return dict(area_ratio=st["area_ratio"], H=st["H"], H_rest=desc.H_rest, K_mean=st["K_mean"], K_abs=st["K_abs"],
                K_min=st["K_min"], K_max=st["K_max"], k1=st["k1"], k2=st["k2"],
                signed_planar_curvature=tw[:, 0] * curv[:, 1] - tw[:, 1] * curv[:, 0],
                normal=st["normal"], moment=st["moment"], curvature_vector=curv, tangent=tan,
                area_ratio_total=out["area"] / out["area_rest"], willmore=out["willmore"],
                willmore_rest=out["willmore_rest"], gauss_bonnet_residual=out["gauss_bonnet_residual"])


def smooth_test_shapes():
    """The smooth rope shapes of the tests above (none folds the 12-gon tube).

    Returns
    -------
    dict of str to np.ndarray, shape (N_ROPE, 3)
        Rope nodes [m] by shape name.
    """
    shapes = {"straight": straight(), "stretched": straight() * 1.05, "compressed": straight() * 0.97,
              "vertical": np.c_[np.zeros((N_ROPE, 2)), np.linspace(0, L_ROPE, N_ROPE)]}
    for seed in range(5):
        shapes[f"smooth{seed}"] = smooth_curve(seed)
    for arc_radius in (0.05, 0.021):
        shapes[f"arc{arc_radius}"] = unstretched_arc(arc_radius)
    for deg in (1.0, 37.5, 80.0, 104.0):
        shapes[f"turn{deg}"] = turning_polyline(np.radians(deg))
    radius, rise = 0.03, 0.01
    th = np.arange(N_ROPE) * 0.3
    shapes["helix"] = np.stack([radius * np.cos(th), radius * np.sin(th), rise * th], 1)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        lam = 1 + np.interp(np.arange(N_ROPE - 1), np.linspace(0, N_ROPE - 2, 6), rng.uniform(-0.03, 0.05, 6))
        psi = np.radians(40.0) * np.sin(np.linspace(0, 3 * np.pi, N_ROPE - 2) + rng.uniform(0, 6))
        centerline = turning_polyline(psi, lengths=H_ROPE * lam)
        centerline[:, 2] = 0.01 * np.sin(np.linspace(0, 2 * np.pi, N_ROPE) + rng.uniform(0, 6))
        shapes[f"nonuniform{seed}"] = centerline
    sharp = turning_polyline(np.radians(100.0) * np.sin(np.arange(N_ROPE - 2)))
    sharp[:, 2] = 0.002 * np.cos(np.arange(N_ROPE))
    shapes["alternating100"] = sharp
    return shapes


def hairpin(deg, nodes=(30,), plane_twist=0.0, lengths=None):
    """Straight rope along +x with a turn of ``deg`` at each node in ``nodes``.

    Parameters
    ----------
    deg : float
        Turning angle at each listed node [deg].
    nodes : tuple of int, optional
        Interior nodes that turn.
    plane_twist : float, optional
        Rotation of the bending plane about the tangent from one turning node to the next [rad] (3-D folds).
    lengths : np.ndarray, shape (N_ROPE - 1,), optional
        Segment lengths [m]; H_ROPE each if None.

    Returns
    -------
    np.ndarray, shape (N_ROPE, 3)
        Rope nodes [m], starting at the origin.
    """
    tangents = np.zeros((N_ROPE - 1, 3))
    tangents[0] = [1.0, 0.0, 0.0]
    binormal = np.array([0.0, 0.0, 1.0])  # binormal of the bend
    for node in range(1, N_ROPE - 1):
        if node in nodes:
            # Rodrigues rotation of the incoming tangent about the binormal, then of the binormal about the new
            # tangent by plane_twist.
            ang = np.radians(deg)
            prev = tangents[node - 1]
            tangents[node] = (prev * np.cos(ang) + np.cross(binormal, prev) * np.sin(ang)
                              + binormal * (binormal @ prev) * (1 - np.cos(ang)))
            cos_twist, sin_twist = np.cos(plane_twist), np.sin(plane_twist)
            binormal = (binormal * cos_twist + np.cross(tangents[node], binormal) * sin_twist
                        + tangents[node] * (tangents[node] @ binormal) * (1 - cos_twist))
        else:
            tangents[node] = tangents[node - 1]
    seg_len = np.full(N_ROPE - 1, H_ROPE) if lengths is None else lengths
    return np.r_[np.zeros((1, 3)), np.cumsum(seg_len[:, None] * tangents, 0)]


def check_fold_outputs(desc, centerline, smooth_ok=False):
    """Assert the fold contract on one rope shape.

    No warning and finite outputs; invalid samples equal the closed forms exactly, valid samples equal the
    pre-fold-handling mesh values to 1e-12, and the globals come from the mesh if and only if every sample is valid.

    Parameters
    ----------
    desc : TubeSurfaceDescriptor
        Descriptor under test.
    centerline : np.ndarray, shape (N, 3)
        Rope nodes [m].
    smooth_ok : bool, optional
        Unused.

    Returns
    -------
    out : dict
        Descriptor outputs.
    sm : dict
        Smooth-tube closed forms (``desc.smooth_tube_forms``).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = desc(centerline)
        sm = desc.smooth_tube_forms(centerline)
    ref = legacy_call(desc, centerline)
    assert all(np.all(np.isfinite(value)) for value in out.values())
    valid = out["valid"]
    assert (out["valid"].dtype == bool and out["fold_max"].shape == (desc.num_samples,)
            and out["n_invalid"] == np.count_nonzero(~valid))
    for key in PER_SAMPLE:
        np.testing.assert_allclose(out[key][valid], ref[key][valid], rtol=1e-12, atol=1e-12, err_msg=key)
    for key in CLOSED:
        assert np.array_equal(out[key][~valid], sm[key][~valid]), key
    assert np.array_equal(out["tangent"], ref["tangent"]) and np.array_equal(out["H_rest"], ref["H_rest"])
    bad_normal = ~np.all(np.isfinite(ref["normal"]), axis=1)
    np.testing.assert_array_equal(out["normal"], np.where((~valid & bad_normal)[:, None], sm["normal"], ref["normal"]))
    assert out["gauss_bonnet_residual"] == ref["gauss_bonnet_residual"]
    if valid.all():
        for key in GLOBALS:
            assert out[key] == ref[key], key
    else:
        assert out["area_ratio_total"] == sm["area_ratio_total"] and out["willmore"] == sm["willmore"]
    return out, sm


def test_valid_everywhere_and_unchanged_on_smooth_shapes():
    """On every smooth test shape all samples are valid and every output equals the pre-fold-handling implementation
    (frozen in legacy_call) to 1e-12, for three sample layouts."""
    for sample_s in (None, np.linspace(0, L_ROPE, N_ROPE), np.sort(np.random.default_rng(0).uniform(0, L_ROPE, 13))):
        desc = rope_descriptor(sample_s)
        for name, centerline in smooth_test_shapes().items():
            out = desc(centerline)
            ref = legacy_call(desc, centerline)
            assert out["valid"].all() and out["n_invalid"] == 0, name
            assert np.all(out["fold_max"] < np.radians(106.0)), name
            for key in PER_SAMPLE + GLOBALS:
                np.testing.assert_allclose(out[key], ref[key], rtol=1e-12, atol=1e-12, err_msg=f"{name} {key}")


def test_fold_limit_is_the_code_s_prism_condition():
    """Validity follows the prism condition rho (f(psi_i) + f(psi_{i+1})) < l_i with the code's capped miter factor
    f = sin(psi/2) / max(cos(psi/2), 1/2), plus psi < 120 degrees: uniform turns fold at 2 atan(h / (2 rho)) = 106.3
    degrees per node, two adjacent equal turns at the same angle, a lone turn only past the 120-degree miter cap; a
    shorter segment folds earlier."""
    desc = rope_descriptor()
    lim = 2 * np.arctan(H_ROPE / (2 * desc.rho))
    for eps, ok in ((-1e-6, True), (1e-6, False)):
        assert desc(turning_polyline(lim + eps))["valid"].all() == ok
        assert desc(hairpin(np.degrees(lim + eps), nodes=(30, 31)))["valid"].all() == ok
        assert desc(hairpin(120.0 + 1e3 * eps, nodes=(30,)))["valid"].all() == ok
    assert desc(hairpin(119.0, nodes=(30,)))["valid"].all()
    # Segment 30 between the two turns shortened to 70%: the pair folds at 2 atan(0.7 h / (2 rho)).
    seg_lengths = np.full(N_ROPE - 1, H_ROPE)
    seg_lengths[30] *= 0.7
    lim2 = np.degrees(2 * np.arctan(0.7 * H_ROPE / (2 * desc.rho)))
    assert desc(hairpin(lim2 - 0.01, nodes=(30, 31), lengths=seg_lengths))["valid"].all()
    out = desc(hairpin(lim2 + 0.01, nodes=(30, 31), lengths=seg_lengths))
    assert out["n_invalid"] == 1 and not out["valid"][10] and abs(out["fold_max"][10] - np.radians(lim2 + 0.01)) < 1e-9
    print(f"\nfold limit: uniform {np.degrees(lim):.2f} deg/node, lone turn 120 deg, "
          f"pair on a 70% segment {lim2:.2f} deg")


@pytest.mark.parametrize("deg", [110.0, 130.0, 150.0, 175.0])
@pytest.mark.parametrize("twist", [0.0, 0.9])
def test_hairpin_folds(deg, twist):
    """Hairpins (3 consecutive nodes turning by deg, planar or with the bending plane rotating 0.9 rad per node) at the
    middle and next to each end, and the whole rope turning by deg per node: the samples touching the fold are flagged,
    their outputs are the closed forms exactly, the others keep the mesh values; all finite."""
    desc = rope_descriptor()
    rs = desc.groups[: N_ROPE * 12: 12]
    for nodes in ((29, 30, 31), (2, 3, 4), (56, 57, 58)):
        centerline = hairpin(deg, nodes, twist) @ random_rotation(int(deg)).T
        out, sm = check_fold_outputs(desc, centerline)
        # Folded segments: between two turned nodes (deg > 106.3), and next to any node past the 120-degree miter cap.
        seg_bad = np.zeros(N_ROPE - 1, bool)
        for node in nodes:
            seg_bad[[node - 1, node]] |= deg > 120 or (node - 1 in nodes, node + 1 in nodes)
        # A sample's rings j own half of segments j-1 and j.
        expect = np.array([seg_bad[np.clip(np.r_[np.flatnonzero(rs == sample) - 1, np.flatnonzero(rs == sample)],
                                           0, N_ROPE - 2)].any()
                           for sample in range(desc.num_samples)])
        np.testing.assert_array_equal(out["valid"], ~expect)
        assert np.isclose(out["fold_max"][~out["valid"]].max(), np.radians(deg)) and out["n_invalid"] in (1, 2, 3)
    centerline = turning_polyline(np.radians(deg))
    out, sm = check_fold_outputs(desc, centerline)
    assert out["n_invalid"] == desc.num_samples
    # Closed-form values: smooth-tube H, k1, k2, K_mean; bounded extremes (r kappa clamped to 0.9).
    radius = R_ROPE
    assert np.allclose(out["H"] * 2 * radius, 1) and np.allclose(out["k1"] * radius, 1) and np.all(out["k2"] == 0)
    assert (np.all(out["K_mean"] == 0) and np.all(out["K_min"] * radius * radius >= -9 - 1e-9)
            and np.all(out["K_max"] * radius * radius <= 0.9 / 1.9 + 1e-9))
    np.testing.assert_allclose(out["area_ratio"], 1, atol=1e-12)
    ref = window_turning(centerline, desc)
    np.testing.assert_allclose(out["curvature_vector"][1:-1], ref[1:-1], rtol=1e-12, atol=1e-9)


def test_180_degree_folds():
    """A rope doubled back on itself: exactly 180 degrees at one node (any miter), nearly so, and spread over two
    nodes (two 90-degree turns, which the 12-gon tube follows: valid)."""
    desc = rope_descriptor()
    centerline = straight().copy()
    centerline[31:] = 2 * centerline[30] - centerline[31:]
    for centerline in (centerline, centerline + np.array([0, 1e-12, 0]) * (np.arange(N_ROPE) > 30)[:, None],
                       hairpin(179.9, (30,)) @ random_rotation(3).T):
        out, sm = check_fold_outputs(desc, centerline)
        assert np.degrees(np.max(out["fold_max"])) > 179.8 and out["n_invalid"] == 1 and not out["valid"][10]
    out, sm = check_fold_outputs(desc, hairpin(90.0, (30, 31)))
    assert out["valid"].all() and np.isclose(np.degrees(np.max(out["fold_max"])), 90)


def real_looking_fold(rng):
    """A smooth carried rope with 1-3 hairpins (grasp point, free end, middle): a few adjacent nodes turning 60-178
    degrees each in random planes, random +-3% segment stretch and out-of-plane sag.

    Parameters
    ----------
    rng : np.random.Generator
        Source of all random choices.

    Returns
    -------
    np.ndarray, shape (N_ROPE, 3)
        Rope nodes [m] after a random rigid motion.
    """
    psi = np.radians(rng.uniform(0, 30, N_ROPE - 2)) * np.sign(rng.normal(size=N_ROPE - 2))
    for _ in range(rng.integers(1, 4)):
        start = rng.integers(1, N_ROPE - 4)
        width = rng.integers(1, 4)
        psi[start: start + width] = np.radians(rng.uniform(60, 178, len(psi[start: start + width])))
    centerline = turning_polyline(psi, lengths=H_ROPE * rng.uniform(0.97, 1.03, N_ROPE - 1))
    centerline[:, 2] = rng.uniform(0, 0.03) * np.sin(np.linspace(0, rng.uniform(1, 6), N_ROPE) + rng.uniform(0, 6))
    return centerline @ random_rotation(int(rng.integers(1 << 30))).T + rng.normal(size=3)


def test_random_real_looking_folds_and_rigid_invariance():
    """200 random folded ropes: the fold contract holds; validity and fold_max are rigid-motion invariant and the
    outputs keep their invariance/equivariance through the closed forms."""
    rng = np.random.default_rng(11)
    for sample_s in (None, np.linspace(0, L_ROPE, N_ROPE)):
        desc = rope_descriptor(sample_s)
        n_inv = n_tot = 0
        for _ in range(100):
            centerline = real_looking_fold(rng)
            out, _ = check_fold_outputs(desc, centerline)
            n_inv, n_tot = n_inv + out["n_invalid"], n_tot + desc.num_samples
            rotation, shift = random_rotation(int(rng.integers(1000))), rng.normal(size=3)
            moved = desc(centerline @ rotation.T + shift)
            assert np.array_equal(moved["valid"], out["valid"])
            np.testing.assert_allclose(moved["fold_max"], out["fold_max"], atol=1e-9)
            for key in ("area_ratio", "H", "K_mean", "K_abs", "K_min", "K_max", "k1", "k2", "area_ratio_total",
                        "willmore"):
                np.testing.assert_allclose(moved[key], out[key], rtol=1e-8, atol=1e-8 / R_ROPE**2, err_msg=key)
            for key in ("curvature_vector", "moment", "tangent"):
                np.testing.assert_allclose(moved[key], out[key] @ rotation.T, rtol=1e-8, atol=1e-8, err_msg=key)
        print(f"\nrandom folds ({desc.num_samples} samples): {n_inv / n_tot:.1%} of samples invalid")
        assert 0.02 < n_inv / n_tot < 0.8


@pytest.mark.parametrize("seed", range(6))
def test_closed_forms_match_mesh_on_valid_samples(seed):
    """Property: on random smooth ropes (curvature up to 150 1/m, stretch, 3-D, rigid motion), on every valid sample
    the smooth-tube closed forms reproduce the mesh's exact quantities: area_ratio to 1e-9, curvature_vector,
    moment and signed curvature to round-off, area_ratio_total exactly; K_abs and H to the 12-gon's accuracy."""
    rng = np.random.default_rng(seed)
    centerline = smooth_curve(seed, kmax=rng.uniform(20, 150), z_amplitude=rng.uniform(0, 0.02))
    centerline = centerline[0] + (centerline - centerline[0]) * rng.uniform(0.97, 1.04)
    centerline = centerline @ random_rotation(seed + 20).T + rng.normal(size=3)
    for sample_s in (None, np.linspace(0, L_ROPE, N_ROPE)):
        desc = rope_descriptor(sample_s)
        out, sm = desc(centerline), desc.smooth_tube_forms(centerline)
        assert out["valid"].mean() > 0.8  # 150 1/m x 13.5 mm = 116 degrees per node can fold a few samples
        # A sample made of an end ring alone also holds cap faces (class docstring): not a tube strip.
        rs = desc.groups[: N_ROPE * 12: 12]
        mask = ~np.isin(np.arange(desc.num_samples), rs[[0, -1]]) & out["valid"]
        out = {key: value[mask] if np.ndim(value) else value for key, value in out.items()}
        sm = {key: value[mask] if np.ndim(value) else value for key, value in sm.items()}
        np.testing.assert_allclose(sm["area_ratio"], out["area_ratio"], rtol=0, atol=1e-9)
        curv_vec = out["curvature_vector"]
        rel = (np.linalg.norm(sm["curvature_vector"] - curv_vec, axis=1)
               / np.maximum(np.linalg.norm(curv_vec, axis=1), 1e-12))
        mag = np.linalg.norm(curv_vec, axis=1)
        assert (np.abs(sm["curvature_vector"] - curv_vec).max() < 1e-9
                and (rel[mag > 5].max() if (mag > 5).any() else 0) < 1e-9)
        np.testing.assert_allclose(sm["moment"], out["moment"], atol=1e-9 / R_ROPE)
        np.testing.assert_allclose(sm["signed_planar_curvature"], out["signed_planar_curvature"], atol=1e-9)
        assert abs(sm["area_ratio_total"] - out["area_ratio_total"]) < 1e-12
        big = sm["K_abs"] * R_ROPE**2 > 1e-3
        ratio = out["K_abs"][big] / sm["K_abs"][big]
        assert np.all((ratio > 0.94) & (ratio < 1.03)), ratio
        # The 12-gon's H drifts low at sharp bends (TubeSurfaceDescriptor docstring, row H).
        np.testing.assert_allclose(out["H"] * 2 * R_ROPE, 1, atol=0.15)
    print(f"\nclosed forms vs mesh (seed {seed}): K_abs mesh/closed {ratio.min():.3f}..{ratio.max():.3f}")


def test_closed_form_globals_at_rest():
    """The smooth-tube globals are anchored to the mesh on the rest shape and track the length."""
    desc = rope_descriptor()
    sm = desc.smooth_tube_forms(straight())
    assert abs(sm["willmore"] / desc.forms.willmore_rest - 1) < 1e-12 and abs(sm["area_ratio_total"] - 1) < 1e-12
    np.testing.assert_allclose(sm["area_ratio"], 1, atol=1e-12)
    assert np.allclose(sm["curvature_vector"], 0) and np.all(sm["normal"] == 0) and np.all(sm["K_abs"] == 0)
    lam = 1.04
    sm = desc.smooth_tube_forms(straight() * lam)
    lat = 2 * np.pi * R_ROPE * L_ROPE
    assert np.isclose(sm["area_ratio_total"], (lat * lam + desc.forms.area_rest - lat) / desc.forms.area_rest,
                      rtol=1e-12)
    # Bending raises the smooth-tube Willmore energy by about (pi r / 4) int kappa^2 ds.
    centerline = unstretched_arc(0.05)
    dW = desc.smooth_tube_forms(centerline)["willmore"] - desc.forms.willmore_rest
    assert np.isclose(dW, np.pi * R_ROPE / 4 * L_ROPE / 0.05**2, rtol=0.02)


def random_3d_turns(rng):
    """Rope turning at every node by a random angle (up to 60-179 degrees) in a random plane, segments stretched or
    compressed to 0.6-1.3 h: many folds of every kind, including turn planes that differ at the two ends of a
    segment.

    Parameters
    ----------
    rng : np.random.Generator
        Source of all random choices.

    Returns
    -------
    np.ndarray, shape (N_ROPE, 3)
        Rope nodes [m], starting at the origin along +x.
    """
    psi = np.radians(rng.uniform(0, rng.choice([60, 100, 115, 130, 179]), N_ROPE - 2))
    tangents = np.zeros((N_ROPE - 1, 3))
    tangents[0] = [1.0, 0.0, 0.0]
    for node in range(1, N_ROPE - 1):
        # Random direction perpendicular to the previous tangent: the turn plane at this node.
        perp = rng.normal(size=3)
        perp = perp - (perp @ tangents[node - 1]) * tangents[node - 1]
        tangents[node] = (np.cos(psi[node - 1]) * tangents[node - 1]
                          + np.sin(psi[node - 1]) * perp / np.linalg.norm(perp))
    return np.r_[np.zeros((1, 3)), np.cumsum(H_ROPE * rng.uniform(0.6, 1.3, N_ROPE - 1)[:, None] * tangents, 0)]


def test_validity_rule_is_sufficient_for_exactness():
    """Property (reviewer, 2026-10-12): wherever a sample is flagged valid, the mesh's exact quantities really are
    exact, i.e. the mesh area_ratio and curvature_vector equal the smooth-tube closed forms to round-off, on ropes with
    random 3-D turns of up to 179 degrees and 0.6-1.3x segment lengths (rules 1-2 bound the worst generator over all
    directions, so they may flag a sample whose 12-gon happens not to fold: conservative, never permissive)."""
    desc = rope_descriptor()
    rs = desc.groups[: N_ROPE * 12: 12]
    # End-ring-only samples hold cap faces (class docstring).
    interior = ~np.isin(np.arange(desc.num_samples), rs[[0, -1]])
    rng = np.random.default_rng(5)
    n_valid = n_invalid = 0
    for _ in range(300):
        centerline = random_3d_turns(rng)
        out, sm, ref = desc(centerline), desc.smooth_tube_forms(centerline), legacy_call(desc, centerline)
        mask = out["valid"] & interior
        np.testing.assert_allclose(ref["area_ratio"][mask], sm["area_ratio"][mask], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ref["curvature_vector"][mask], sm["curvature_vector"][mask], rtol=0, atol=1e-9)
        n_valid, n_invalid = n_valid + mask.sum(), n_invalid + (~out["valid"]).sum()
    assert n_valid > 1000 and n_invalid > 1000
