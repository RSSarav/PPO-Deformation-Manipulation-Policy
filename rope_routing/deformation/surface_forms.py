"""Surface area and curvature as the description of deformation: the first and second fundamental forms of an object's
boundary surface (notes/A_deformation_math.md).

Definition
----------
A body with rest boundary surface S0 is deformed by phi into S = phi(S0). At every material point p of S0:

    first fundamental form   I  = g = (d phi)^T (d phi) on the tangent plane T_p S0  (metric: lengths, angles, area)
    second fundamental form  II = b = dx . dn                                        (how the normal turns: curvature)
    shape operator           Sh = g^-1 b, mean curvature H = tr(Sh) / 2, Gaussian curvature K = det(Sh),
                             principal curvatures k1,2 = H +- sqrt(H^2 - K)

Deformation is the pair (membrane strain E = (g - g0) / 2, bending strain beta = b - b0), as in Koiter shell theory and
Discrete Shells (Grinspun et al. 2003). Area changes as dA = J dA0 with the areal stretch J = sqrt(det g / det g0). By
Bonnet's theorem, (I, II) subject to the Gauss-Codazzi equations determine S up to a rigid motion, so the pair is a
complete description of shape; the positions of the material points fix the remaining rigid motion. Signs: n is the
outward normal and b = dx . dn, so a sphere of radius R has H = 1/R and K = 1/R^2.

Object-agnostic design
----------------------
Every deformable object is a triangle mesh of its boundary surface with FIXED material connectivity: cloth (the
sheet), volumetric soft bodies (the boundary of the tetrahedral mesh), ropes, cables and rods (the tube around the
simulated centreline; a discrete elastic rod keeps rigid circular cross-sections, so the tube IS the rod's simulated
surface). :class:`SurfaceMeshForms` never knows which object it is measuring; :func:`tube_surface` and
:class:`TubeSurfaceDescriptor` are the only rod-specific adapters.

Discrete estimators
-------------------
Vectorised, with the index structures precomputed, so that a call is cheap (about 1.5 ms for the rope):

* first form per face: F_f = Ds Dm^-1, g_f = F_f^T F_f = Dm^-T G Dm^-1 (G = Gram matrix of the current edges), the
  membrane strain E_f, the area ratio J_f and the principal stretches;
* vertex areas: mixed Voronoi areas (Meyer, Desbrun, Schroeder, Barr 2003; obtuse-triangle rule);
* H: cotangent Laplacian, (1 / (2 A_v)) sum_j (cot a_ij + cot b_ij)(x_i - x_j) = 2 H n;
* K: angle defect, (2 pi - sum theta) / A_v (pi - sum theta at boundary vertices);
* k1, k2: H +- sqrt(max(H^2 - K, 0));
* K n: first moment of the vertex's Gauss image, 1/2 sum_j theta_ij unit(x_i - x_j) with the dihedral angles theta;
* II per face: b_f = sym(F_f^T dn_f) from vertex-normal differences (Rusinkiewicz 2004), an optional cross-check;
* N_f: area-weighted face normal 2 A_f n_f (for orientation checks: an inverted face points inward).

Tube identities
---------------
Derived in notes A. With tube radius r and the curvature vector kappa N of the centreline, per centreline length ds:

    area of a ring strip           = 2 pi r ds   (for any bending)  ->  per-sample area ratio = axial stretch
    integral of H over the strip   = pi ds                          ->  area-mean H = 1/(2r), independent of bending
    integral of K n over the strip = -pi kappa N ds                 ->  moment = -kappa N / (2r), kappa N = -2 r moment
    integral of |K| over the strip = 4 |kappa| ds                   ->  K_abs = 2 |kappa| / (pi r)

The first and third identities hold exactly on the mitred, area-preserving polygonal tube of
:class:`TubeSurfaceDescriptor` while it does not fold; where it folds, the descriptor substitutes these smooth-tube
closed forms.

Units: lengths in metres, mean and principal curvatures in 1/m, Gaussian curvature in 1/m^2, areas in m^2, angles in
radians.

Use in the system: ``SurfaceRouteEnv`` and the final ``GenericSurfaceEnv`` evaluate a :class:`TubeSurfaceDescriptor` on
the simulated rope at every step. ``GenericSurfaceEnv`` observes its per-sample outputs and, for its coverage reward,
also reads the vertex-level surface of the last call (``last_V`` and ``last_out["K_v"]``).
:func:`polyline_window_curvature` provides the curvature targets of the legacy ``SurfaceRouteEnv``; the remaining curve
helpers serve tests and cross-validation.
"""

import numpy as np
from scipy.sparse import csr_matrix

TWO_PI = 2.0 * np.pi


def _rownorm(vectors):
    """Euclidean norm along the last axis."""
    return np.sqrt(np.einsum("...i,...i->...", vectors, vectors))


def _unit(vectors):
    """Vectors normalised along the last axis (zero vectors stay zero)."""
    return vectors / np.maximum(_rownorm(vectors), 1e-300)[..., None]


class SurfaceMeshForms:
    """Fundamental-form measures of a triangle mesh with fixed connectivity, relative to its rest shape.

    The constructor precomputes everything that depends only on the connectivity and the rest shape: the sparse matrix
    that sums per-corner quantities onto vertices, the boundary vertices, the Euler characteristic, the interior edges
    with their two faces (for the Gauss-image moment), the inverse rest edge matrix of every face in an orthonormal rest
    face basis, and the rest areas. Calling the object on current vertex positions returns the per-vertex and per-face
    measures (see :meth:`__call__`).

    Faces must be consistently oriented: counter-clockwise seen from outside for a closed surface (outward normals).

    Parameters
    ----------
    faces : array_like of int, shape (m, 3)
        Vertex indices of each triangle.
    V_rest : array_like, shape (n, 3)
        Rest vertex positions [m].

    Attributes
    ----------
    faces : np.ndarray of int64, shape (m, 3)
        Triangles.
    num_vertices, num_faces : int
        Number of vertices n and of faces m.
    V_rest : np.ndarray, shape (n, 3)
        Rest vertex positions [m].
    boundary_v : np.ndarray of bool, shape (n,)
        Vertices on a boundary edge (an edge with a single face).
    closed : bool
        Whether every edge has exactly two faces.
    euler_characteristic : int
        Vertices - edges + faces, counting the vertices used by a face.
    A0_v, A0_f, A0_bary_v : np.ndarray
        Rest mixed Voronoi vertex areas (n,), face areas (m,) and barycentric vertex areas (n,) [m^2].
    area_rest : float
        Rest surface area [m^2].
    willmore_rest : float
        Rest Willmore energy, sum over vertices of H_v^2 A_v.
    rest : dict
        Output of :meth:`__call__` on the rest shape.

    Examples
    --------
    A regular octahedron, scaled by 2:

    >>> vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    >>> faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    >>> forms = SurfaceMeshForms(faces, vertices)
    >>> forms.closed, forms.euler_characteristic, round(forms.area_rest, 4)
    (True, 2, 6.9282)
    >>> out = forms(2.0 * vertices)
    >>> out["area_ratio_v"].round(6).tolist()
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    >>> round(float(out["K_v"][0] * out["A_v"][0]), 6), round(out["gauss_bonnet_residual"], 12)  # defect 2 pi / 3
    (2.094395, 0.0)
    """

    def __init__(self, faces, V_rest):
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        V_rest = np.array(V_rest, dtype=np.float64)
        num_vertices, num_faces = len(V_rest), len(faces)
        self.faces, self.num_vertices, self.num_faces, self.V_rest = faces, num_vertices, num_faces, V_rest
        self._f0, self._f1, self._f2 = faces[:, 0].copy(), faces[:, 1].copy(), faces[:, 2].copy()
        # Corner c = k * m + f is corner k of face f; one sparse product sums every per-corner quantity onto vertices.
        self._scatter = csr_matrix((np.ones(3 * num_faces), (faces.T.ravel(), np.arange(3 * num_faces))),
                                   shape=(num_vertices, 3 * num_faces))

        # Edges, boundary vertices and the Euler characteristic.
        edge_pairs = np.sort(np.concatenate([faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]]), axis=1)
        edges, count = np.unique(edge_pairs, axis=0, return_counts=True)
        self.boundary_v = np.zeros(num_vertices, bool)
        self.boundary_v[edges[count == 1].ravel()] = True
        self.closed = bool(np.all(count == 2))
        self.euler_characteristic = int(len(np.unique(faces)) - len(edges) + num_faces)
        # The angle sum of a flat neighbourhood: 2 pi inside, pi on the boundary (discrete Gauss-Bonnet).
        self._flat = np.where(self.boundary_v, np.pi, TWO_PI)

        # Interior edges i < j with the face f that holds the half-edge i -> j and its twin face g (j -> i), for the
        # Gauss-image moment. Half-edge k * m + f runs faces[f, k] -> faces[f, k + 1], which is the edge opposite corner
        # k + 2 (index _eh into the per-call edge arrays). _edge_scatter scatters an edge quantity q_e onto the edge's
        # endpoints as -q_e at i and +q_e at j.
        half_from, half_to, half_face = faces.T.ravel(), faces[:, [1, 2, 0]].T.ravel(), np.tile(np.arange(num_faces), 3)
        key, twin = half_from * num_vertices + half_to, half_to * num_vertices + half_from
        order = np.argsort(key)
        match = np.minimum(np.searchsorted(key[order], twin), 3 * num_faces - 1)
        interior = np.flatnonzero((key[order][match] == twin) & (half_from < half_to))
        self._ef, self._eg = half_face[interior], half_face[order][match][interior]
        self._eh = ((interior // num_faces + 2) % 3) * num_faces + half_face[interior]
        num_edges = len(interior)
        self._edge_scatter = csr_matrix((np.r_[-np.ones(num_edges), np.ones(num_edges)],
                                         (np.r_[half_from[interior], half_to[interior]],
                                          np.tile(np.arange(num_edges), 2))),
                                        shape=(num_vertices, num_edges))

        # Orthonormal rest face basis: basis1 along x1 - x0, basis2 = n x basis1. The rest edges expressed in it, Dm,
        # form an upper triangular matrix, so P = Dm^-1 = [[p00, p01], [0, p11]].
        edge_a = V_rest[self._f1] - V_rest[self._f0]
        edge_c = V_rest[self._f2] - V_rest[self._f0]
        len_a = _rownorm(edge_a)
        basis1 = edge_a / len_a[:, None]
        basis2 = np.cross(_unit(np.cross(edge_a, edge_c)), basis1)
        coord1, coord2 = np.einsum("ij,ij->i", edge_c, basis1), np.einsum("ij,ij->i", edge_c, basis2)
        self._p00, self._p01, self._p11 = 1.0 / len_a, -coord1 / (len_a * coord2), 1.0 / coord2
        self._group_cache = None
        self._b0_f = None

        # Rest areas and rest Willmore energy.
        rest_parts = self._compute(V_rest)
        self.A0_v, self.A0_f, self.A0_bary_v = rest_parts["A_v"], rest_parts["A_f"], rest_parts["A_bary_v"]
        self.area_rest = float(rest_parts["A_f"].sum())
        self.willmore_rest = float(np.sum(rest_parts["H_v"] ** 2 * rest_parts["A_v"]))
        self.rest = self(V_rest)

    # ------------------------------------------------------------------------------------------------ per-call core
    def _compute(self, vertices):
        """Vertex areas, normals and curvatures, and the per-face quantities that :meth:`__call__` reuses.

        Parameters
        ----------
        vertices : array_like, shape (n, 3)
            Current vertex positions [m].

        Returns
        -------
        dict
            "A_v", "n_v", "H_v", "K_v", "k1_v", "k2_v", "Kn_v", "A_bary_v", "defect" per vertex; "A_f", "N_f" per
            face; "L" (squared edge lengths) and "d" (dot products of the two edges at each corner), shape (3, m).
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        corner0, corner1, corner2 = vertices[self._f0], vertices[self._f1], vertices[self._f2]
        edges = np.stack([corner2 - corner1, corner0 - corner2, corner1 - corner0])  # (3, m, 3): edge opposite corner k
        edge_sq = np.einsum("kfi,kfi->kf", edges, edges)  # squared edge lengths
        area_normal = np.cross(edges[2], -edges[1])  # (x1 - x0) x (x2 - x0) = 2 A_f n_f
        double_area = np.maximum(_rownorm(area_normal), 1e-300)
        dots = 0.5 * edge_sq.sum(0) - edge_sq  # law of cosines: dot product of the two edges leaving corner k
        cot = dots / double_area
        theta = np.arctan2(double_area, dots)  # (3, m) corner angles

        # Mixed Voronoi area per corner (Meyer et al. 2003): the Voronoi split if the face is non-obtuse, otherwise
        # A/2 for the obtuse corner and A/4 for the other two (the circumcentre would lie outside the face).
        face_area = 0.5 * double_area
        voronoi_area = 0.125 * (edge_sq[[1, 2, 0]] * cot[[1, 2, 0]] + edge_sq[[2, 0, 1]] * cot[[2, 0, 1]])
        obtuse = dots < 0
        corner_area = np.where(obtuse.any(0), np.where(obtuse, 0.5, 0.25) * face_area, voronoi_area)

        # Cotangent Laplacian: corner k adds cot(opposite angle) (x_k - x_j) for each of its two edges.
        weighted = cot[:, :, None] * edges
        laplacian = weighted[[1, 2, 0]] - weighted[[2, 0, 1]]

        # Sum every per-corner quantity onto the vertices with one sparse product.
        cols = np.empty((3 * self.num_faces, 9))
        cols[:, 0], cols[:, 1] = corner_area.ravel(), theta.ravel()
        cols[:, 2:5] = laplacian.reshape(-1, 3)
        cols[:, 5:8] = np.tile(area_normal, (3, 1))  # area-weighted vertex normals
        cols[:, 8] = np.tile(face_area / 3.0, 3)  # barycentric areas: region areas for group statistics
        sums = self._scatter @ cols
        vertex_area = np.maximum(sums[:, 0], 1e-300)
        vertex_normal = _unit(sums[:, 5:8])

        # Mean curvature from the cotangent Laplacian (= 2 H n), Gaussian curvature from the angle defect.
        curv_normal = sums[:, 2:5] / (2.0 * vertex_area[:, None])  # = 2 H n
        mean_curv = 0.5 * _rownorm(curv_normal) * np.sign(np.einsum("ij,ij->i", curv_normal, vertex_normal))
        if self.boundary_v.any():  # at boundary vertices the cotangent vector also holds the in-plane boundary turning
            mean_curv = np.where(self.boundary_v, 0.5 * np.einsum("ij,ij->i", curv_normal, vertex_normal), mean_curv)
        defect = self._flat - sums[:, 1]
        gauss_curv = defect / vertex_area
        disc = np.sqrt(np.maximum(mean_curv * mean_curv - gauss_curv, 0.0))

        # The integral of K n over the vertex star is the first moment of its Gauss image, the spherical polygon of the
        # face normals: 1/2 the sum over its arcs of (arc angle) x (arc pole) (Stokes: int n dA = 1/2 loop n x dn on the
        # sphere). Each arc is the dihedral angle theta_e of an edge and its pole is the unit edge vector, so the star
        # of x_i gets 1/2 sum_j theta_ij unit(x_i - x_j) (theta > 0 where convex). This is exact for the polyhedral
        # surface; defect * n_v only approximates it, with a shape-dependent error (on a 12-gon tube ring: -1.1% for
        # gentle bends, where it tends to n sin(pi/n) / pi, +0.1% at a 37.5-degree turn per node, +4.2% at 80 degrees).
        unit_normal = area_normal / double_area[:, None]
        normal_f, normal_g = unit_normal[self._ef], unit_normal[self._eg]
        edge_dir = edges.reshape(-1, 3)[self._eh] / np.maximum(np.sqrt(edge_sq.ravel()[self._eh]), 1e-300)[:, None]
        dihedral = np.arctan2(np.einsum("ij,ij->i", np.cross(normal_f, normal_g), edge_dir),
                              np.einsum("ij,ij->i", normal_f, normal_g))
        gauss_moment = self._edge_scatter @ (0.5 * dihedral[:, None] * edge_dir)
        return dict(A_v=vertex_area, A_f=face_area, L=edge_sq, d=dots, n_v=vertex_normal, H_v=mean_curv, K_v=gauss_curv,
                    k1_v=mean_curv + disc, k2_v=mean_curv - disc, defect=defect, A_bary_v=sums[:, 8],
                    Kn_v=gauss_moment, N_f=area_normal)

    def __call__(self, vertices) -> dict:
        """Fundamental-form measures of the mesh at the given vertex positions.

        Parameters
        ----------
        vertices : array_like, shape (n, 3)
            Current vertex positions [m].

        Returns
        -------
        dict
            Per vertex (n,): "A_v" mixed Voronoi area [m^2], "area_ratio_v" (A_v / A0_v), "n_v" unit normal (n, 3),
            "H_v" mean curvature [1/m], "K_v" Gaussian curvature [1/m^2], "k1_v", "k2_v" principal curvatures [1/m],
            "boundary_v", "A_bary_v" barycentric area [m^2], "Kn_v" integral of K n over the vertex star (n, 3).
            Per face (m,): "J_f" area ratio, "E_f" membrane strain in the rest face basis (m, 2, 2), "stretch_f"
            principal stretches (m, 2), "N_f" area-weighted normal 2 A_f n_f (m, 3) [m^2].
            Globals: "area", "area_rest" [m^2], "willmore", "willmore_rest" (sum of H_v^2 A_v), and
            "gauss_bonnet_residual" [rad].
        """
        parts = self._compute(vertices)
        # First fundamental form per face in the rest face basis: g = P^T G P with the Gram matrix
        # G = [[a.a, a.c], [a.c, c.c]] of a = x1 - x0 and c = x2 - x0. G needs only squared edge lengths: the metric
        # IS the edge lengths.
        edge_sq, dots = parts["L"], parts["d"]
        p00, p01, p11 = self._p00, self._p01, self._p11
        sq_a, sq_c, dot_ac = edge_sq[2], edge_sq[1], dots[0]
        gp01 = sq_a * p01 + dot_ac * p11  # entry (0, 1) of G P
        g00, g01, g11 = sq_a * p00 * p00, p00 * gp01, p01 * gp01 + p11 * (dot_ac * p01 + sq_c * p11)
        strain = np.empty((self.num_faces, 2, 2))
        strain[:, 0, 0], strain[:, 1, 1] = 0.5 * (g00 - 1.0), 0.5 * (g11 - 1.0)
        strain[:, 0, 1] = strain[:, 1, 0] = 0.5 * g01
        # Principal stretches: square roots of the eigenvalues of g.
        half_tr = 0.5 * (g00 + g11)
        half_gap = np.sqrt(0.25 * (g00 - g11) ** 2 + g01 * g01)
        stretch = np.sqrt(np.maximum(np.stack([half_tr + half_gap, half_tr - half_gap], 1), 0.0))
        vertex_area = parts["A_v"]
        # Discrete Gauss-Bonnet: the angle defects (pi - sum theta on the boundary) sum to 2 pi chi, an identity for
        # any manifold mesh, so the residual checks the mesh and the angle computation, not the accuracy.
        gauss_bonnet = float(parts["defect"].sum() - TWO_PI * self.euler_characteristic)
        return dict(A_v=vertex_area, area_ratio_v=vertex_area / self.A0_v, n_v=parts["n_v"], H_v=parts["H_v"],
                    K_v=parts["K_v"], k1_v=parts["k1_v"], k2_v=parts["k2_v"], boundary_v=self.boundary_v,
                    A_bary_v=parts["A_bary_v"], Kn_v=parts["Kn_v"], J_f=parts["A_f"] / self.A0_f, E_f=strain,
                    stretch_f=stretch, N_f=parts["N_f"], area=float(parts["A_f"].sum()), area_rest=self.area_rest,
                    willmore=float(np.sum(parts["H_v"] ** 2 * vertex_area)), willmore_rest=self.willmore_rest,
                    gauss_bonnet_residual=gauss_bonnet)

    # ------------------------------------------------------------------------------------------------ aggregation
    def _groups(self, groups, n_groups):
        """Membership matrix (n_groups, n), the vertices sorted by group, and the start of each NON-EMPTY group in that
        order (min and max with ``reduceat`` need strictly increasing in-range starts).

        Cached while the same ``groups`` array object is reused (an array modified in place keeps the stale cache).
        """
        cache = self._group_cache
        if cache is not None and cache[0] is groups and cache[1] == n_groups:
            return cache[2:]
        group_ids = np.asarray(groups, dtype=np.int64)
        members = np.flatnonzero(group_ids >= 0)
        membership = csr_matrix((np.ones(len(members)), (group_ids[members], members)),
                                shape=(n_groups, self.num_vertices))
        order = members[np.argsort(group_ids[members], kind="stable")]
        counts = np.bincount(group_ids[members], minlength=n_groups)
        nonempty = counts > 0
        starts = np.r_[0, np.cumsum(counts)[:-1]][nonempty]
        self._group_cache = (groups, n_groups, membership, order, starts, nonempty)
        return membership, order, starts, nonempty

    def group_stats(self, out: dict, groups, n_groups: int) -> dict:
        """Area-weighted statistics per vertex group, e.g. per material sample: integral over the group's region
        divided by the region's area.

        The integrals use the vertex estimators (H_v A_v, and the angle defect K_v A_v, do not depend on how area is
        attributed); the region area is the sum of the barycentric vertex areas (A_f / 3 per corner). Barycentric areas
        split every face evenly, so on a tube each ring gets exactly half of each adjacent segment prism (the
        generator-length differences cancel around the ring), and the per-sample area ratio equals the axial stretch
        for any bending; summed mixed Voronoi areas drift between neighbouring samples at sharp bends.

        Parameters
        ----------
        out : dict
            Output of :meth:`__call__`.
        groups : np.ndarray of int, shape (n,)
            Group of each vertex; -1 for none.
        n_groups : int
            Number of groups G.

        Returns
        -------
        dict of np.ndarray
            Per group (G,): "area_ratio", "H" [1/m], "K_mean", "K_abs", "K_min", "K_max" [1/m^2] (min and max over
            the group's vertices), "k1", "k2" [1/m]; "normal" (G, 3): normalised mean normal, zero if it vanishes;
            "moment" (G, 3): integral of K n over the region divided by its area [1/m^2], using the exact first moment
            of the region's Gauss image (Kn_v). "normal" is ill-conditioned on closed cross-sections such as a tube
            ring, where the mean normal before normalisation is only -r kappa N / 2; use "moment" there. Empty groups
            give NaN.
        """
        membership, order, starts, nonempty = self._groups(groups, n_groups)
        area, gauss_curv, vertex_normal = out["A_v"], out["K_v"], out["n_v"]
        gauss_area = gauss_curv * area
        cols = np.column_stack([out["A_bary_v"], self.A0_bary_v, out["H_v"] * area, gauss_area, np.abs(gauss_area),
                                out["k1_v"] * area, out["k2_v"] * area, vertex_normal * area[:, None], out["Kn_v"]])
        sums = membership @ cols
        group_area = np.where(nonempty, sums[:, 0], np.nan)
        kmin, kmax = np.full(n_groups, np.nan), np.full(n_groups, np.nan)
        if len(order):
            sorted_curv = gauss_curv[order]
            kmin[nonempty] = np.minimum.reduceat(sorted_curv, starts)
            kmax[nonempty] = np.maximum.reduceat(sorted_curv, starts)
        normal_sum = sums[:, 7:10]
        normal_len = _rownorm(normal_sum)
        normal = np.where((normal_len > 1e-9 * np.abs(sums[:, 0]))[:, None],
                          normal_sum / np.maximum(normal_len, 1e-300)[:, None], 0.0)
        return dict(area_ratio=group_area / sums[:, 1], H=sums[:, 2] / group_area, K_mean=sums[:, 3] / group_area,
                    K_abs=sums[:, 4] / group_area, K_min=kmin, K_max=kmax, k1=sums[:, 5] / group_area,
                    k2=sums[:, 6] / group_area, normal=normal, moment=sums[:, 10:13] / group_area[:, None])

    # ------------------------------------------------------------------------------------------------ cross-check
    def _face_gb(self, vertices, vertex_normal):
        """Metric g and symmetrised second fundamental form b of every face in the rest face basis, with the normal
        field fitted linearly over the face from its three vertex normals."""
        edge_a, edge_c = vertices[self._f1] - vertices[self._f0], vertices[self._f2] - vertices[self._f0]
        dn_a, dn_c = (vertex_normal[self._f1] - vertex_normal[self._f0],
                      vertex_normal[self._f2] - vertex_normal[self._f0])
        rest_inv = np.zeros((self.num_faces, 2, 2))
        rest_inv[:, 0, 0], rest_inv[:, 0, 1], rest_inv[:, 1, 1] = self._p00, self._p01, self._p11
        tangent_map = np.stack([edge_a, edge_c], 2) @ rest_inv  # (m, 3, 2): F_f, the tangent map in the rest face basis
        normal_map = np.stack([dn_a, dn_c], 2) @ rest_inv  # linear fit of the normal field over the face
        metric = np.einsum("fia,fib->fab", tangent_map, tangent_map)
        second = np.einsum("fia,fib->fab", tangent_map, normal_map)
        return metric, 0.5 * (second + second.transpose(0, 2, 1))

    def _max_normals(self, vertices):
        """Vertex normals with Max's (1999) weights sin(theta) / (|e_a| |e_b|) per corner.

        These are exact for vertices on a sphere, as in Rusinkiewicz's curvature code; area-weighted normals make the
        per-face fit only first-order accurate.
        """
        corner0, corner1, corner2 = vertices[self._f0], vertices[self._f1], vertices[self._f2]
        edges = np.stack([corner2 - corner1, corner0 - corner2, corner1 - corner0])
        edge_sq = np.einsum("kfi,kfi->kf", edges, edges)
        area_normal = np.cross(edges[2], -edges[1])
        weight = 1.0 / np.maximum(edge_sq[[1, 2, 0]] * edge_sq[[2, 0, 1]], 1e-300)  # corner k: 1 / (|e_a|^2 |e_b|^2)
        return _unit(self._scatter @ (weight[:, :, None] * area_normal[None]).reshape(-1, 3))

    def face_second_fundamental_form(self, vertices) -> dict:
        """Per-face second fundamental form fitted to the vertex normals (Rusinkiewicz 2004).

        Computes b_f = sym(dx^T dn) in the rest face basis from the three vertex normals of each face, its
        shape-operator invariants, and the bending strain beta_f = b_f - b0_f. This is not on the fast path: it
        cross-checks the vertex estimators and gives the (E, beta) pair of the definition explicitly.

        Parameters
        ----------
        vertices : array_like, shape (n, 3)
            Current vertex positions [m].

        Returns
        -------
        dict
            "g_f" metric (m, 2, 2), "b_f" second fundamental form (m, 2, 2) [1/m], "beta_f" bending strain
            (m, 2, 2) [1/m], "H_f" mean curvature (m,) [1/m] and "K_f" Gaussian curvature (m,) [1/m^2].
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        metric, second = self._face_gb(vertices, self._max_normals(vertices))
        if self._b0_f is None:
            self._b0_f = self._face_gb(self.V_rest, self._max_normals(self.V_rest))[1]
        det_metric = metric[:, 0, 0] * metric[:, 1, 1] - metric[:, 0, 1] ** 2
        # H = tr(g^-1 b) / 2 and K = det(b) / det(g) for 2 x 2 symmetric g and b.
        mean_curv = 0.5 * (metric[:, 1, 1] * second[:, 0, 0] - 2.0 * metric[:, 0, 1] * second[:, 0, 1]
                           + metric[:, 0, 0] * second[:, 1, 1]) / det_metric
        gauss_curv = (second[:, 0, 0] * second[:, 1, 1] - second[:, 0, 1] ** 2) / det_metric
        return dict(g_f=metric, b_f=second, beta_f=second - self._b0_f, H_f=mean_curv, K_f=gauss_curv)


# ======================================================================================================== tube adapter
def ring_radius(radius: float, n_around: int = 12, area_preserving: bool = True) -> float:
    """Circumradius of the polygonal tube cross-section.

    With ``area_preserving`` the n-gon has the perimeter 2 pi r of the circle, so the polyhedral tube has exactly the
    smooth tube's area 2 pi r L for any bending (every segment is an exact prism, see :func:`tube_surface`).

    Parameters
    ----------
    radius : float
        Tube radius r [m].
    n_around : int
        Number of polygon sides.
    area_preserving : bool
        Scale the polygon to the circle's perimeter; otherwise return ``radius``.

    Returns
    -------
    float
        Circumradius rho [m].

    Examples
    --------
    >>> round(ring_radius(0.005), 7)
    0.0050576
    """
    return float(radius * np.pi / (n_around * np.sin(np.pi / n_around)) if area_preserving else radius)


def _perpendicular(tangents):
    """Unit vectors perpendicular to the rows of ``tangents`` (any such vectors)."""
    axis = np.zeros_like(tangents)
    axis[np.arange(len(tangents)), np.argmin(np.abs(tangents), axis=1)] = 1.0
    return _unit(np.cross(tangents, axis))


def _initial_normal(centerline, tangent0):
    """Reference normal at node 0, perpendicular to ``tangent0``.

    It must be a rotation-EQUIVARIANT function of the centreline (a fixed world axis would make the measures depend on
    the rope's orientation): the normal of the best-fit plane (the smallest principal axis), oriented by the curve's
    vector area. For a planar curve every ring is then mirror-symmetric about the plane; with an even ``n_around`` a
    sign flip only relabels the ring vertices.
    """
    if np.all(np.isfinite(centerline)):
        centered = centerline - centerline.mean(0)
        axes = np.linalg.eigh(centered.T @ centered)[1]
        vector_area = np.cross(centered[:-1], centered[1:]).sum(0)
        for axis in (axes[:, 0], axes[:, 1]):
            normal = axis - (axis @ tangent0) * tangent0
            length = np.linalg.norm(normal)
            if length > 0.5:  # true for at least one of two orthonormal axes
                return normal / length * (1.0 if normal @ vector_area >= 0.0 else -1.0)
    return np.array([0.0, 0.0, 1.0])  # non-finite centreline: the outputs are NaN anyway


def _frames(centerline):
    """Segment tangents, node bisectors, half-turn cosines and rotation-minimising normals of a polyline.

    Double reflection (Wang, Juettler, Zheng, Liu 2008) at every interior node: reflect across the mitre plane
    (normal T_i), then across the plane normal to t_i. For a polyline the composite is exactly the minimal rotation
    that takes t_{i-1} to t_i about their binormal (discrete parallel transport, as in discrete elastic rods), so the
    frame never twists, and the cross-section of segment i-1 cut by the mitre plane coincides with that of segment i:
    each segment is an exact prism.

    Parameters
    ----------
    centerline : np.ndarray, shape (N, 3)
        Polyline nodes [m].

    Returns
    -------
    tangents : np.ndarray, shape (N - 1, 3)
        Unit segment tangents t.
    bisectors : np.ndarray, shape (N - 2, 3)
        Unit bisectors T of consecutive tangents at the interior nodes (the mitre-plane normals).
    cos_half : np.ndarray, shape (N - 2,)
        Cosine of half the turning angle at each interior node.
    normals : np.ndarray, shape (N - 1, 3)
        Rotation-minimising unit normal u of each segment.
    """
    delta = np.diff(centerline, axis=0)
    tangents = delta / np.maximum(_rownorm(delta), 1e-300)[:, None]
    bisector_sum = tangents[:-1] + tangents[1:]
    sum_len = _rownorm(bisector_sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        bisectors = np.where((sum_len > 1e-9)[:, None], bisector_sum / sum_len[:, None],
                             _perpendicular(tangents[:-1]))  # a 180-degree fold accepts any mitre
    next_tangent = tangents[1:]
    identity = np.eye(3)
    # Composite double reflection (I - 2 t t^T)(I - 2 T T^T) at every interior node.
    transport = (identity - 2.0 * next_tangent[:, :, None] * next_tangent[:, None, :]
                 - 2.0 * bisectors[:, :, None] * bisectors[:, None, :]
                 + 4.0 * np.einsum("ij,ij->i", next_tangent, bisectors)[:, None, None]
                 * next_tangent[:, :, None] * bisectors[:, None, :])
    span = 1
    while span < len(transport):  # inclusive prefix product Q_i ... Q_1 (Hillis-Steele scan, log2(N) batched matmuls)
        transport = np.concatenate([transport[:span], transport[span:] @ transport[:-span]])
        span *= 2
    normal0 = _initial_normal(centerline, tangents[0])
    normals = np.empty_like(tangents)
    normals[0] = normal0
    normals[1:] = transport @ normal0
    return tangents, bisectors, 0.5 * sum_len, normals


def _tube_vertices(centerline, rho, cos_phi, sin_phi, cap_cos, cap_sin):
    """Vertices of the mitred polygonal tube (vertex layout in :func:`_tube_topology`).

    Parameters
    ----------
    centerline : np.ndarray, shape (N, 3)
        Centreline nodes [m].
    rho : float
        Circumradius of the cross-section [m].
    cos_phi, sin_phi : np.ndarray, shape (n,)
        Cosine and sine of the angle of each ring vertex around the tube.
    cap_cos, cap_sin : np.ndarray, shape (cap_rings,)
        Cap ring profile from :func:`_cap_profile`.

    Returns
    -------
    vertices : np.ndarray, shape (N n + 2 (cap_rings n + 1), 3)
        Ring vertices, then the start cap (rings and pole), then the end cap [m].
    tangents, bisectors : np.ndarray
        Segment tangents and node bisectors from :func:`_frames`, for reuse.
    """
    tangents, bisectors, cos_half, normals = _frames(centerline)
    binormals = np.cross(tangents, normals)
    # (N - 1, n, 3) offsets of the cross-section vertices in each segment's frame.
    offsets = rho * (cos_phi[None, :, None] * normals[:, None, :] + sin_phi[None, :, None] * binormals[:, None, :])
    num_nodes, n_around = len(centerline), len(cos_phi)
    rings = np.empty((num_nodes, n_around, 3))
    rings[0] = centerline[0] + offsets[0]
    rings[-1] = centerline[-1] + offsets[-1]
    # Interior ring = the cross-section of segment i-1 slid along t_{i-1} onto the mitre (bisector) plane at node i.
    # The prisms stay exact while rho (tan(psi_i / 2) + tan(psi_{i+1} / 2)) < segment length (106.3 degrees per node on
    # a uniformly bent rope: 13.5 mm segments, rho 5.06 mm); sharper turns fold the tube onto itself. The mitre factor
    # 1/cos(psi/2) is capped at 2 (120 degrees) so that a kink cannot throw ring vertices far from the centreline;
    # beyond it the ring leaves the mitre plane. TubeSurfaceDescriptor flags both cases ("valid") and substitutes
    # closed forms.
    inner = offsets[:-1]
    proj = np.einsum("inc,ic->in", inner, bisectors) / np.maximum(cos_half, 0.5)[:, None]
    rings[1:-1] = centerline[1:-1, None, :] + inner - proj[..., None] * tangents[:-1, None, :]
    # Hemispherical end caps: cap ring k has radius rho cos(alpha_k) and sits rho sin(alpha_k) beyond the end node.
    lift = rho * cap_sin[:, None, None]
    cap_start = centerline[0] + cap_cos[:, None, None] * offsets[0][None] - lift * tangents[0]
    cap_end = centerline[-1] + cap_cos[:, None, None] * offsets[-1][None] + lift * tangents[-1]
    vertices = np.concatenate([rings.reshape(-1, 3), cap_start.reshape(-1, 3),
                               (centerline[0] - rho * tangents[0])[None], cap_end.reshape(-1, 3),
                               (centerline[-1] + rho * tangents[-1])[None]])
    return vertices, tangents, bisectors


def _tube_topology(num_nodes, n_around, cap_rings):
    """Faces (outward, counter-clockwise seen from outside) and ring index of every vertex of the tube.

    Vertex layout: ``num_nodes`` rings of ``n_around`` vertices, then the start cap (``cap_rings`` rings and the pole),
    then the end cap (``cap_rings`` rings and the pole). Cap vertices get ring 0 or ``num_nodes - 1``.

    Returns
    -------
    faces : np.ndarray of int64, shape (2 n (N - 1) + 2 (2 n cap_rings + n), 3)
        Triangles: two per quad of the lateral surface and the cap strips, one per fan sector at each pole.
    ring : np.ndarray of int64, shape (n_vertices,)
        Centreline node (ring) of every vertex.
    """
    around = np.arange(n_around)

    def strip(back, front):  # back = ring further back along the centreline, front = ring further ahead
        back_next, front_next = np.roll(back, -1, axis=-1), np.roll(front, -1, axis=-1)
        return np.concatenate([np.stack([back, back_next, front], -1), np.stack([back_next, front_next, front], -1)],
                              -2).reshape(-1, 3)

    rings = np.arange(num_nodes)[:, None] * n_around + around  # (N, n)
    start_base = num_nodes * n_around
    start_cap = start_base + np.arange(cap_rings)[:, None] * n_around + around  # start cap ring k at row k - 1
    start_pole = start_base + cap_rings * n_around
    end_base = start_pole + 1
    end_cap = end_base + np.arange(cap_rings)[:, None] * n_around + around
    end_pole = end_base + cap_rings * n_around
    start_seq = np.concatenate([start_cap[::-1], rings[:1]])  # back to front: cap ring c, ..., cap ring 1, ring 0
    end_seq = np.concatenate([rings[-1:], end_cap])  # ring N - 1, cap ring 1, ..., cap ring c
    lateral = strip(rings[:-1], rings[1:])
    start_strips = strip(start_seq[:-1], start_seq[1:]) if cap_rings else np.zeros((0, 3), np.int64)
    end_strips = strip(end_seq[:-1], end_seq[1:]) if cap_rings else np.zeros((0, 3), np.int64)
    start_ring = start_seq[0]
    start_fan = np.stack([np.full(n_around, start_pole), np.roll(start_ring, -1), start_ring], 1)
    end_ring = end_seq[-1]
    end_fan = np.stack([end_ring, np.roll(end_ring, -1), np.full(n_around, end_pole)], 1)
    faces = np.concatenate([lateral, start_strips, start_fan, end_strips, end_fan]).astype(np.int64)
    ring = np.concatenate([np.repeat(np.arange(num_nodes), n_around), np.zeros(cap_rings * n_around + 1, np.int64),
                           np.full(cap_rings * n_around + 1, num_nodes - 1)])
    return faces, ring


def _cap_profile(cap_rings):
    """Cosines and sines of the cap ring latitudes alpha_k = k pi / (2 (cap_rings + 1)), k = 1 .. cap_rings."""
    alpha = np.arange(1, cap_rings + 1) * (0.5 * np.pi / (cap_rings + 1))
    return np.cos(alpha), np.sin(alpha)


def tube_surface(centerline, radius, n_around=12, cap_rings=3, area_preserving=True):
    """Closed triangle mesh of the tube of a given radius around a polyline centreline.

    One ring (a regular ``n_around``-gon) per centreline node, oriented by rotation-minimising frames (double
    reflection, Wang et al. 2008) from a deterministic, rotation-equivariant initial normal. Interior rings lie in the
    mitre (bisector) plane, so every segment is an exact polygonal prism and, with ``area_preserving`` (n-gon perimeter
    2 pi r), the lateral area is exactly 2 pi r times the centreline length, unless a turn is sharp enough to fold the
    tube (see :func:`_tube_vertices`). Hemispherical end caps (``cap_rings`` rings and a pole) close the surface; the
    normals point outward. The connectivity depends only on (N, ``n_around``, ``cap_rings``).

    Parameters
    ----------
    centerline : array_like, shape (N, 3)
        Centreline nodes [m].
    radius : float
        Tube radius [m].
    n_around : int
        Vertices per ring.
    cap_rings : int
        Rings per end cap, excluding the pole.
    area_preserving : bool
        Use the area-preserving ring radius (:func:`ring_radius`).

    Returns
    -------
    vertices : np.ndarray, shape (n_vertices, 3)
        Vertex positions [m].
    faces : np.ndarray of int64, shape (n_faces, 3)
        Triangles.
    ring : np.ndarray of int64, shape (n_vertices,)
        Centreline node of each vertex (0 or N - 1 for cap vertices).

    Examples
    --------
    >>> line = np.c_[np.linspace(0.0, 0.8, 61), np.zeros(61), np.zeros(61)]
    >>> vertices, faces, ring = tube_surface(line, 0.005)
    >>> vertices.shape, faces.shape, int(ring.max())
    ((806, 3), (1608, 3), 60)
    >>> forms = SurfaceMeshForms(faces, vertices)
    >>> forms.closed, forms.euler_characteristic
    (True, 2)
    """
    points = np.asarray(centerline, dtype=np.float64)
    phi = TWO_PI * np.arange(n_around) / n_around
    vertices, _, _ = _tube_vertices(points, ring_radius(radius, n_around, area_preserving), np.cos(phi), np.sin(phi),
                                    *_cap_profile(cap_rings))
    faces, ring = _tube_topology(len(points), n_around, cap_rings)
    return vertices, faces, ring


class TubeSurfaceDescriptor:
    """Per-material-sample surface-area and curvature measures of a rod or rope, computed on its tube surface.

    Every call builds the mitred 12-gon tube around the current centreline, evaluates :class:`SurfaceMeshForms` on it
    and averages the vertex measures over each material sample's rings (:meth:`SurfaceMeshForms.group_stats`).

    Sample windows. ``sample_s`` holds the ascending rest arclengths of the material samples from node 0. Each ring
    belongs to the sample whose window contains it; the windows split the rope at the midpoints between consecutive
    samples. Cap vertices belong to no sample, and neither do the two end rings: a hemispherical cap closes the surface
    there, so their angle defects hold the cap's Gaussian curvature (an end ring stays in its sample if it is the only
    ring there). ``sample_window[k]`` is the rest-arclength interval actually covered by sample k's rings (their
    Voronoi cells along the centreline); for samples on nodes it is the midpoint split. Use it for targets:
    ``polyline_window_curvature(route, desc.sample_window)``.

    Outputs for a rope (smooth-tube values with r = radius and kappa = centreline curvature, per sample window; in
    brackets what the 12-gon tube at the rope's 13.5 mm node spacing measures instead, notes A):

    ================  =============================================================================================
    area_ratio        axial stretch of the window's segments (exact for the mitred tube, whatever the bending)
    H                 ~ 1/(2r) whatever the bending (area mean of H over the rings); H_rest is the rest value
                      [1/(2 rho) when straight, drifting down by 2.6% at a 2.1 cm bend: no bending information]
    K_mean            ~ 0: the Gaussian curvature of a full ring integrates to zero
    K_abs             2 |kappa| / (pi r): unsigned bending without averaging loss [-3.1% to +2.0%]
    K_max, K_min      ~ kappa / (r (1 + r kappa)) outside the bend, -kappa / (r (1 - r kappa)) inside
                      [K_max +1.5% to +22%, K_min -3% to -9% on gentle arcs, depending on the bend and the ring
                      orientation; at sharp node turns the mesh K_min departs from the smooth tube's while still
                      valid: r^2 K_min -2.2 instead of -1.4 at 89 degrees per node, -25 instead of -2.2 at the
                      106.3-degree limit]
    k1, k2            area means of the principal curvatures: ~ 1/r (around the tube) and ~ 0 (along it)
    curvature_vector  window MEAN of kappa N = (T(hi) - T(lo)) / length = -2 r moment [exact, see below]; shorter
                      than |kappa| by 2 sin(dth/2) / dth when the rope turns by dth inside the window (0.86 for a
                      4 cm window on a 2.1 cm arc)
    signed_planar_    the signed value of curvature_vector about +z (left turn > 0), taken about the tangent at the
    curvature         middle of the window
    tangent           unit centreline tangent at the sample; "normal" is ill-conditioned on a tube (see group_stats)
    willmore          sum of H^2 A ~ pi L / (2r sqrt(1 - r^2 kappa^2)) + caps: bending adds ~ (pi r / 4) int kappa^2
                      ds [not resolved: 21-25% of that increase on 5-20 cm arcs, a decrease at 2.1 cm; not used as a
                      feature]
    ================  =============================================================================================

    Tube identity. kappa N = -2 r moment holds EXACTLY on the discrete tube for any bending that does not fold it
    (turns below 106 degrees per node on the rope, see :func:`_tube_vertices`). The Gauss image of ring k's star is
    the band between the great circles normal to t_{k-1} and t_k, whose first moment is -pi (t_k - t_{k-1}): only the
    generator edges (dihedral angle 2 pi / n) contribute, the diagonals of the planar quads have none, and the ring
    edges cancel within the ring. The ring's area is 2 pi r times its Voronoi length because the n-gon perimeter is
    2 pi r. Hence curvature_vector = (t(hi) - t(lo)) / (current window length): exactly the window mean of the discrete
    curvature vector of the centreline polyline.

    Folds (notes A). The simulated rope can turn more sharply than the 12-gon tube can follow (hairpins at the gripper
    and at the free end: 8.5-19% of sample-steps before the frame-consistent reset, none since in 6400 steps, but
    nothing prevents them). The prisms then fold through each other and the mesh values become artefacts (area_ratio
    off by up to 15%, curvature by up to 170 1/m, r^2 K_min down to -32). Per sample:

    * ``valid`` (K,) bool: every prism touching the sample's rings is exact, i.e. for each such segment i,
      rho (f(psi_i) + f(psi_{i+1})) < l_i with f(psi) = sin(psi/2) / max(cos(psi/2), 1/2) (the shortest generator
      keeps a positive length), psi < 120 degrees at both ends (the ring lies in its mitre plane), and none of the
      segment's faces is inverted;
    * ``fold_max`` (K,): largest node turning angle [rad] on those segments' nodes (the window's rings plus one node on
      either side).

    Invalid samples take the smooth tube's closed forms (:meth:`smooth_tube_forms`, which equal the mesh values for
    area_ratio and curvature_vector on valid samples), except "normal" (kept while finite) and "tangent". If any sample
    is invalid, the globals are the smooth tube's as well: area_ratio_total = (2 pi r L + caps) / rest area, and
    willmore = sum over node cells of pi l_j / (2 r sqrt(1 - (r kappa_j)^2)) plus a constant for the caps fitted to
    willmore_rest on the rest shape. ``n_invalid`` counts the invalid samples; gauss_bonnet_residual always comes from
    the mesh.

    Samples should be at least two node spacings apart near the ends: an end ring is excluded only if its sample has
    other rings, and a sample made of the end ring alone reads the cap's curvature (K_abs ~ 5000 1/m^2 at rest).
    Empty samples (no ring) read NaN, with valid = True and fold_max = NaN.

    Parameters
    ----------
    rest_centerline : array_like, shape (N, 3)
        Rest centreline nodes [m].
    radius : float
        Tube (rope) radius r [m].
    sample_s : array_like, shape (K,)
        Ascending rest arclengths of the material samples from node 0 [m].
    n_around : int
        Vertices per ring.
    cap_rings : int
        Rings per end cap, excluding the pole.

    Attributes
    ----------
    num_nodes, num_samples : int
        Number of centreline nodes N and of material samples K.
    radius, rho : float
        Tube radius r and ring circumradius rho [m].
    faces : np.ndarray of int64, shape (n_faces, 3)
        Tube triangles (fixed connectivity).
    ring_of_vertex : np.ndarray of int64, shape (n_vertices,)
        Centreline node of each vertex.
    forms : SurfaceMeshForms
        The mesh estimator on the rest tube.
    groups : np.ndarray of int64, shape (n_vertices,)
        Sample of each vertex; -1 for cap and excluded end-ring vertices.
    sample_window : np.ndarray, shape (K, 2)
        Rest-arclength interval covered by each sample's rings [m]; NaN for empty samples.
    rest_stats : dict
        ``group_stats`` of the rest tube.
    H_rest : np.ndarray, shape (K,)
        Rest mean curvature per sample [1/m].
    length_rest : float
        Rest centreline length [m].
    last_V, last_out : np.ndarray, dict
        Tube vertices [m] and ``SurfaceMeshForms`` output of the last call (None before the first call).

    Examples
    --------
    A straight rope of 61 nodes with 21 samples, then stretched by 1%: the per-sample area ratio is the axial stretch.

    >>> rest = np.c_[np.linspace(0.0, 0.8, 61), np.zeros(61), np.zeros(61)]
    >>> desc = TubeSurfaceDescriptor(rest, 0.005, np.linspace(0.0, 0.8, 21))
    >>> out = desc(rest)
    >>> bool(np.allclose(out["area_ratio"], 1.0)), bool(out["valid"].all()), out["n_invalid"]
    (True, True, 0)
    >>> bool(np.allclose(desc(1.01 * rest)["area_ratio"], 1.01))
    True

    The same rope bent into a counter-clockwise arc of radius 0.2 m (curvature 5 1/m):

    >>> angle = np.linspace(0.0, 0.8, 61) / 0.2
    >>> arc = np.c_[0.2 * np.sin(angle), 0.2 * (1.0 - np.cos(angle)), np.zeros(61)]
    >>> out = desc(arc)
    >>> curvature = out["signed_planar_curvature"][1:-1].round(2)
    >>> float(curvature.min()), float(curvature.max())
    (4.99, 4.99)
    """

    def __init__(self, rest_centerline, radius, sample_s, n_around=12, cap_rings=3):
        rest_line = np.asarray(rest_centerline, dtype=np.float64)
        num_nodes = len(rest_line)
        self.num_nodes, self.radius, self.n_around, self.cap_rings = num_nodes, float(radius), n_around, cap_rings

        # The rest tube and its mesh estimator.
        self.rho = ring_radius(radius, n_around, True)
        phi = TWO_PI * np.arange(n_around) / n_around
        self._cs, self._sn = np.cos(phi), np.sin(phi)
        self._cap = _cap_profile(cap_rings)
        rest_vertices, _, _ = _tube_vertices(rest_line, self.rho, self._cs, self._sn, *self._cap)
        self.faces, self.ring_of_vertex = _tube_topology(num_nodes, n_around, cap_rings)
        self.forms = SurfaceMeshForms(self.faces, rest_vertices)

        # Assign every ring to the sample whose window contains it; end rings leave a sample that has other rings.
        node_arc = np.r_[0.0, np.cumsum(_rownorm(np.diff(rest_line, axis=0)))]
        sample_arc = np.asarray(sample_s, dtype=np.float64)
        self.num_samples = len(sample_arc)
        ring_sample = np.searchsorted(0.5 * (sample_arc[1:] + sample_arc[:-1]), node_arc, side="right")
        for end in (0, num_nodes - 1):
            if np.count_nonzero(ring_sample == ring_sample[end]) > 1:
                ring_sample[end] = -1
        self.groups = np.full(len(rest_vertices), -1, dtype=np.int64)
        self.groups[: num_nodes * n_around] = np.repeat(ring_sample, n_around)
        cell = np.r_[node_arc[0], 0.5 * (node_arc[1:] + node_arc[:-1]), node_arc[-1]]
        self.sample_window = np.full((self.num_samples, 2), np.nan)
        for sample in range(self.num_samples):
            rings = np.flatnonzero(ring_sample == sample)
            if len(rings):
                self.sample_window[sample] = cell[rings.min()], cell[rings.max() + 1]
        self._ti, self._tf = _cell_coords(node_arc, sample_arc)  # material position of each sample in the node cells
        # The curvature is measured over the window, so its sign is taken about the tangent at the window's middle
        # (= the sample for interior samples; the end samples' windows sit one ring in, past the excluded end ring).
        self._wi, self._wf = _cell_coords(node_arc, np.nan_to_num(self.sample_window.mean(1), nan=0.0))
        self.rest_stats = self.forms.group_stats(self.forms.rest, self.groups, self.num_samples)
        self.H_rest = self.rest_stats["H"]
        self.last_V = self.last_out = None

        # Fold bookkeeping (see __call__): the rings, segments and nodes that each sample's measures depend on.
        num_samples, node_index = self.num_samples, np.arange(num_nodes)
        self._next = np.r_[1:n_around, 0]
        self._Mw = np.zeros((num_samples, num_nodes))
        assigned = ring_sample >= 0
        self._Mw[ring_sample[assigned], node_index[assigned]] = 1.0  # rings of sample k (window sums)
        self._empty = self._Mw.sum(1) == 0
        first_ring = np.where(self._empty, 0, np.argmax(self._Mw > 0, axis=1))
        last_ring = np.where(self._empty, 0, num_nodes - 1 - np.argmax(self._Mw[:, ::-1] > 0, axis=1))
        # A ring's star spans the segments on both sides, whose prisms depend on the mitres at their far nodes too.
        self._lo, self._hi = np.maximum(first_ring - 1, 0), np.minimum(last_ring, num_nodes - 2)  # first/last segment
        segment = np.arange(num_nodes - 1)
        self._Mseg = ((segment >= self._lo[:, None]) & (segment <= self._hi[:, None])
                      & ~self._empty[:, None]).astype(float)
        self._Mnode = (node_index >= self._lo[:, None]) & (node_index <= self._hi[:, None] + 1) & ~self._empty[:, None]
        self._Lw0 = self._Mw @ _voronoi_lengths(_rownorm(np.diff(rest_line, axis=0)))
        self.length_rest = float(node_arc[-1])

        # The smooth-tube globals are anchored to the mesh on the rest shape: the caps are rigid, so they add constants
        # (the caps' area exactly; for Willmore the constant also absorbs the 12-gon's ~1% low H on the straight tube).
        self._area_caps = self.forms.area_rest - TWO_PI * self.radius * self.length_rest
        self._willmore_caps = 0.0
        self._willmore_caps = self.forms.willmore_rest - self.smooth_tube_forms(rest_line)["willmore"]

    # ------------------------------------------------------------------------------------------------ folds
    def _node_turns(self, tangents):
        """Turning angle psi, mitre offset factor f = sin(psi/2) / max(cos(psi/2), 1/2) (as applied by
        :func:`_tube_vertices`) and cos(psi/2) per node, each of shape (N,); psi = f = 0 at the two end nodes, whose
        rings are not mitred."""
        cos_half, sin_half = 0.5 * _rownorm(tangents[:-1] + tangents[1:]), 0.5 * _rownorm(tangents[1:] - tangents[:-1])
        psi, miter, cos_full = np.zeros(self.num_nodes), np.zeros(self.num_nodes), np.ones(self.num_nodes)
        psi[1:-1] = 2.0 * np.arctan2(sin_half, cos_half)
        miter[1:-1] = sin_half / np.maximum(cos_half, 0.5)
        cos_full[1:-1] = cos_half
        return psi, miter, cos_full

    def smooth_tube_forms(self, centerline) -> dict:
        """Closed-form measures of the SMOOTH tube of radius r around the centreline polyline (notes A).

        These are the values that :meth:`__call__` substitutes where the 12-gon tube folds. On unfolded samples
        area_ratio and curvature_vector equal the mesh values to round-off (both are exact there); the curvature
        statistics are the smooth tube's, which the 12-gon reproduces to a few percent. Per sample (window = the
        Voronoi cells of the sample's rings):

        ================  ===========================================================================================
        area_ratio        current window length / rest window length (a ring strip has area 2 pi r ds)
        curvature_vector  (t(hi) - t(lo)) / current window length: window mean of kappa N; moment = -kappa N / (2r)
        K_mean = 0        the integral of K over a ring strip is 0
        K_abs             2 <|kappa|> / (pi r), <|kappa|> = total turning of the window's nodes / window length (the
                          integral of |K| over a strip is 4 |kappa| ds)
        K_min, K_max      -kappa / (r (1 - r kappa)) and kappa / (r (1 + r kappa)): inner and outer extremes at the
                          window's sharpest node, kappa_j = psi_j / (Voronoi length of node j), r kappa clamped to 0.9
        H = 1/(2r)        the integral of H over a strip is pi ds
        k1 = 1/r, k2 = 0  area means of the principal curvatures: 1/r around the tube everywhere; along it
                          -kappa cos(v) / (1 - r kappa cos(v)), which integrates to 0 against
                          dA = r (1 - r kappa cos v) dv ds, so (k1 + k2) / 2 = H and k1 k2 integrates to 0 = K_mean
        normal            -unit(kappa N): the strip's mean normal is -pi r^2 kappa N ds (zero when straight)
        ================  ===========================================================================================

        Globals: area_ratio_total = (2 pi r L + caps' area) / rest area; willmore = sum over node cells of
        pi l_j / (2 r sqrt(1 - (r kappa_j)^2)) (the strip integral of H^2) plus the caps' constant.

        Parameters
        ----------
        centerline : array_like, shape (N, 3)
            Current centreline nodes [m].

        Returns
        -------
        dict
            The per-sample arrays above, plus "signed_planar_curvature", "moment" and the globals.
        """
        points = np.asarray(centerline, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            tangents = _frames(points)[0]
            return self._closed_forms(tangents, self._node_turns(tangents)[0], _rownorm(np.diff(points, axis=0)),
                                      _slerp_tangent(tangents, self._wi, self._wf))

    def _closed_forms(self, tangents, psi, seg_len, window_tangent):
        """Smooth-tube measures from the segment tangents, node turning angles, segment lengths and window-middle
        tangents (see :meth:`smooth_tube_forms`)."""
        radius = self.radius
        voronoi = _voronoi_lengths(seg_len)
        window_len = np.where(self._empty, np.nan, self._Mw @ voronoi)
        curv = (tangents[self._hi] - tangents[self._lo]) / window_len[:, None]
        rk_node = np.minimum(radius * psi / np.maximum(voronoi, 1e-300), 0.9)  # r kappa per node
        rk_max = (self._Mw * rk_node).max(1)
        curv_len = _rownorm(curv)
        full = np.where(self._empty, np.nan, 1.0)  # empty samples stay NaN, as on the mesh
        return dict(area_ratio=window_len / self._Lw0, H=full / (2.0 * radius), K_mean=0.0 * full,
                    K_abs=2.0 * (self._Mw @ psi) / window_len / (np.pi * radius),
                    K_min=-rk_max / (radius * radius * (1.0 - rk_max)) * full,
                    K_max=rk_max / (radius * radius * (1.0 + rk_max)) * full, k1=full / radius, k2=0.0 * full,
                    signed_planar_curvature=window_tangent[:, 0] * curv[:, 1] - window_tangent[:, 1] * curv[:, 0],
                    normal=np.where((curv_len > 1e-12)[:, None], -curv / np.maximum(curv_len, 1e-300)[:, None], 0.0),
                    moment=-curv / (2.0 * radius), curvature_vector=curv,
                    area_ratio_total=float((TWO_PI * radius * seg_len.sum() + self._area_caps) / self.forms.area_rest),
                    willmore=float(np.sum(np.pi * voronoi / (2.0 * radius * np.sqrt(1.0 - rk_node * rk_node))))
                    + self._willmore_caps)

    def __call__(self, centerline) -> dict:
        """Per-sample measures of the rope's tube surface at the given centreline.

        Also stores the tube vertices and the vertex-level ``SurfaceMeshForms`` output in ``last_V`` and ``last_out``.

        Parameters
        ----------
        centerline : array_like, shape (N, 3)
            Current centreline nodes [m]. Degenerate input yields NaN or inf without warnings.

        Returns
        -------
        dict
            Per sample (K,): "area_ratio", "H", "H_rest" [1/m], "K_mean", "K_abs", "K_min", "K_max" [1/m^2], "k1",
            "k2" [1/m], "signed_planar_curvature" [1/m], "valid" (bool), "fold_max" [rad]; per sample (K, 3):
            "normal", "moment" [1/m^2], "curvature_vector" [1/m], "tangent"; globals: "area_ratio_total",
            "willmore", "willmore_rest", "gauss_bonnet_residual" [rad] and "n_invalid".
        """
        points = np.asarray(centerline, dtype=np.float64)
        num_nodes, n_around = self.num_nodes, self.n_around
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):  # degenerate input -> NaN/inf, no warning
            vertices, tangents, bisectors = _tube_vertices(points, self.rho, self._cs, self._sn, *self._cap)
            out = self.forms(vertices)
            stats = self.forms.group_stats(out, self.groups, self.num_samples)
            curv = -2.0 * self.radius * stats["moment"]  # tube identity: kappa N = -2 r moment
            tangent = _slerp_tangent(tangents, self._ti, self._tf)
            window_tangent = _slerp_tangent(tangents, self._wi, self._wf)

            # Fold detection. Segment i's prism is exact while (a) both its rings lie in their mitre planes, i.e. the
            # turns at nodes i and i+1 stay below 120 degrees, where _tube_vertices caps the mitre factor, and (b) its
            # generators keep a positive length: ring vertices sit rho f(psi) cos(.) off the node along the axis, so
            # the shortest generator is l_i - rho (f(psi_i) + f(psi_{i+1})) with f = sin(psi/2) / max(cos(psi/2), 1/2)
            # (= tan(psi/2) below 120 degrees). (c) Safety net: no lateral face of the segment points inward (its
            # area-weighted normal against the outward radial direction of its quad).
            psi, miter, cos_half = self._node_turns(tangents)
            seg_len = _rownorm(np.diff(points, axis=0))
            node_ok = cos_half > 0.5
            seg_ok = (self.rho * (miter[:-1] + miter[1:]) < seg_len) & node_ok[:-1] & node_ok[1:]
            ring_offsets = (vertices[: (num_nodes - 1) * n_around].reshape(num_nodes - 1, n_around, 3)
                            - points[:-1, None, :])
            radial = ring_offsets + ring_offsets[:, self._next]  # 2 (midpoint of ring edge j, j+1 - node i): quad j's
            radial = np.concatenate([radial, radial], 1)  # radial direction, shared by its two triangles (n + n faces)
            area_normal = out["N_f"][: 2 * (num_nodes - 1) * n_around].reshape(num_nodes - 1, 2 * n_around, 3)
            outward = (np.einsum("skc,skc->sk", area_normal, radial)  # N_f . (radial projected normal to t_i)
                       - np.einsum("skc,sc->sk", area_normal, tangents) * np.einsum("skc,sc->sk", radial, tangents))
            seg_ok &= np.all(outward > 0.0, axis=1)
            valid = (self._Mseg @ ~seg_ok) == 0
            fold_max = np.where(self._Mnode, psi, -np.inf).max(1)
            fold_max[self._empty] = np.nan
        self.last_V, self.last_out = vertices, out
        result = dict(area_ratio=stats["area_ratio"], H=stats["H"], H_rest=self.H_rest, K_mean=stats["K_mean"],
                      K_abs=stats["K_abs"], K_min=stats["K_min"], K_max=stats["K_max"], k1=stats["k1"], k2=stats["k2"],
                      signed_planar_curvature=window_tangent[:, 0] * curv[:, 1] - window_tangent[:, 1] * curv[:, 0],
                      normal=stats["normal"], moment=stats["moment"], curvature_vector=curv, tangent=tangent,
                      area_ratio_total=out["area"] / out["area_rest"], willmore=out["willmore"],
                      willmore_rest=out["willmore_rest"], gauss_bonnet_residual=out["gauss_bonnet_residual"],
                      valid=valid, fold_max=fold_max, n_invalid=int(np.count_nonzero(~valid)))
        if not np.all(seg_ok):  # some prism folded: its samples and the globals take the smooth-tube closed forms
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                smooth = self._closed_forms(tangents, psi, seg_len, window_tangent)
            folded = ~valid
            for key, value in smooth.items():
                if np.ndim(value) == 0:
                    result[key] = value
                elif key == "normal":  # ill-conditioned on a tube anyway (group_stats): keep the mesh value if finite
                    result[key] = np.where((folded & ~np.all(np.isfinite(result[key]), axis=1))[:, None], value,
                                           result[key])
                else:
                    result[key] = np.where(folded[:, None] if value.ndim == 2 else folded, value, result[key])
        return result


# ======================================================================================================== curve helpers
def _voronoi_lengths(seg_len):
    """Voronoi length of each polyline node: half of each adjacent segment.

    Examples
    --------
    >>> _voronoi_lengths(np.array([1.0, 2.0, 3.0])).tolist()
    [0.5, 1.5, 2.5, 1.5]
    """
    return np.r_[0.5 * seg_len[:1], 0.5 * (seg_len[:-1] + seg_len[1:]), 0.5 * seg_len[-1:]]


def _cell_coords(node_arc, query_arc):
    """Locate arclengths within the Voronoi cells of a polyline's vertices.

    For each arclength s, return the index i and the fraction f in [0, 1] such that s lies between the midpoints of
    segments i and i+1 (the Voronoi cell of vertex i+1), clamped at the two ends.

    Examples
    --------
    >>> cell, fraction = _cell_coords(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.5, 1.75]))
    >>> cell.tolist(), fraction.tolist()
    ([0, 1], [0.0, 0.25])
    """
    midpoints = 0.5 * (node_arc[:-1] + node_arc[1:])
    if len(midpoints) < 2:
        return np.zeros(len(query_arc), np.int64), np.zeros(len(query_arc))
    cell = np.clip(np.searchsorted(midpoints, query_arc, side="right") - 1, 0, len(midpoints) - 2)
    return cell, np.clip((query_arc - midpoints[cell]) / (midpoints[cell + 1] - midpoints[cell]), 0.0, 1.0)


def _slerp_tangent(tangents, cell, fraction):
    """Unit tangent that rotates uniformly from t_i to t_{i+1} across the Voronoi cell of vertex i+1.

    The tangent is continuous, equal to the segment direction at segment midpoints and to the bisector at
    (equal-segment) vertices, where a polyline turns. Window means of curvature use T(hi) - T(lo), so they match the
    tube's ring sums exactly at segment midpoints.
    """
    if len(tangents) < 2:
        return np.repeat(tangents[:1], len(cell), axis=0)
    start, end = tangents[cell], tangents[cell + 1]
    omega = np.arccos(np.clip(np.einsum("ij,ij->i", start, end), -1.0, 1.0))
    sin_omega = np.sin(omega)
    small = sin_omega < 1e-9
    weight_start = np.where(small, 1.0 - fraction, np.sin((1.0 - fraction) * omega) / np.where(small, 1.0, sin_omega))
    weight_end = np.where(small, fraction, np.sin(fraction * omega) / np.where(small, 1.0, sin_omega))
    return _unit(weight_start[:, None] * start + weight_end[:, None] * end)


def _as3d(points):
    """Points as an (M, 3) float array; planar (M, 2) input gets z = 0."""
    points = np.asarray(points, dtype=np.float64)
    return np.c_[points, np.zeros(len(points))] if points.shape[1] == 2 else points


def curve_curvature_vectors(centerline) -> np.ndarray:
    """Discrete curvature vectors kappa N at the nodes of a polyline.

    kappa N = 2 (t_i - t_{i-1}) / (l_{i-1} + l_i), zero at the two ends. Integrated over a node's Voronoi length it
    gives exactly the tangent change t_i - t_{i-1}, and it is exact for points on a circle (2 sin(psi/2) / l = 1/R).
    Used to cross-validate the tube measures.

    Parameters
    ----------
    centerline : array_like, shape (M, 2) or (M, 3)
        Polyline nodes [m].

    Returns
    -------
    np.ndarray, shape (M, 3)
        Curvature vectors [1/m].

    Examples
    --------
    >>> angle = np.linspace(0.0, np.pi, 7)
    >>> circle = np.c_[0.1 * np.cos(angle), 0.1 * np.sin(angle)]
    >>> np.linalg.norm(curve_curvature_vectors(circle), axis=1).round(6).tolist()
    [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    """
    points = _as3d(centerline)
    delta = np.diff(points, axis=0)
    seg_len = _rownorm(delta)
    tangents = delta / seg_len[:, None]
    kappa = np.zeros_like(points)
    kappa[1:-1] = 2.0 * (tangents[1:] - tangents[:-1]) / (seg_len[:-1] + seg_len[1:])[:, None]
    return kappa


def signed_planar_curvature_of_polyline(polyline) -> np.ndarray:
    """Signed curvature at each vertex of a polyline seen from +z: positive for a left (counter-clockwise) turn.

    Uses the same discrete curvature as :func:`curve_curvature_vectors`; zero at the two ends.

    Parameters
    ----------
    polyline : array_like, shape (M, 2) or (M, 3)
        Polyline vertices [m].

    Returns
    -------
    np.ndarray, shape (M,)
        Signed curvature [1/m].

    Examples
    --------
    >>> angle = np.linspace(0.0, np.pi, 7)
    >>> circle = np.c_[0.1 * np.cos(angle), 0.1 * np.sin(angle)]
    >>> (signed_planar_curvature_of_polyline(circle).round(6) + 0.0).tolist()
    [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    >>> signed_planar_curvature_of_polyline(circle[::-1]).round(6)[1:-1].tolist()
    [-10.0, -10.0, -10.0, -10.0, -10.0]
    """
    polyline = _as3d(polyline)
    kappa = curve_curvature_vectors(polyline)
    tangents = _unit(np.diff(polyline, axis=0))
    vertex_tangent = np.concatenate([tangents[:1], _unit(tangents[:-1] + tangents[1:]), tangents[-1:]])
    return vertex_tangent[:, 0] * kappa[:, 1] - vertex_tangent[:, 1] * kappa[:, 0]


def polyline_window_curvature(polyline, windows):
    """Mean curvature vector of a polyline over arclength windows, and its signed planar value.

    The mean over a window [lo, hi] (arclength from the first vertex) is (T(hi) - T(lo)) / (hi - lo), with T(s) the
    continuous unit tangent of :func:`_slerp_tangent`; the signed value is cross(T(mid), .) . z. TubeSurfaceDescriptor
    measures exactly this window mean on the rope (the K-moment of a ring strip integrates kappa N ds = dT and is
    divided by the window's current length, which equals hi - lo while the rope is unstretched), so
    ``polyline_window_curvature(route, desc.sample_window)`` gives target profiles on the same footing. Windows are
    clipped to the polyline; a window entirely beyond its end gets 0.

    Parameters
    ----------
    polyline : array_like, shape (M, 2) or (M, 3)
        Polyline vertices [m].
    windows : array_like, shape (K, 2)
        Arclength windows (lo, hi) [m].

    Returns
    -------
    kvec : np.ndarray, shape (K, 3)
        Mean curvature vector per window [1/m].
    signed : np.ndarray, shape (K,)
        Signed planar curvature per window [1/m] (positive for a left turn).

    Examples
    --------
    A counter-clockwise arc of radius 0.2 m and a 4 cm window on it:

    >>> angle = np.linspace(0.0, 0.8, 61) / 0.2
    >>> arc = np.c_[0.2 * np.sin(angle), 0.2 * (1.0 - np.cos(angle))]
    >>> kvec, signed = polyline_window_curvature(arc, np.array([[0.30, 0.34]]))
    >>> round(float(signed[0]), 3)
    4.993
    """
    polyline = _as3d(polyline)
    delta = np.diff(polyline, axis=0)
    seg_len = _rownorm(delta)
    keep = seg_len > 0  # a repeated point would give a 0/0 tangent and NaN targets
    tangents = delta[keep] / seg_len[keep, None]
    node_arc = np.r_[0.0, np.cumsum(seg_len[keep])]
    bounds = np.clip(np.asarray(windows, dtype=np.float64), 0.0, node_arc[-1])

    def tangent(arc):
        """Continuous unit tangent at the arclengths ``arc``."""
        return _slerp_tangent(tangents, *_cell_coords(node_arc, arc))

    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    kvec = np.where((span > 1e-12)[:, None], (tangent(hi) - tangent(lo)) / np.maximum(span, 1e-12)[:, None], 0.0)
    mid_tangent = tangent(0.5 * (lo + hi))
    return kvec, mid_tangent[:, 0] * kvec[:, 1] - mid_tangent[:, 1] * kvec[:, 0]
