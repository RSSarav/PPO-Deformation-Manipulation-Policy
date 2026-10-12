# A. Deformation math: surface area and curvature, and where each piece lives in the code

**Main code:** `rope_routing/deformation/surface_forms.py`. Line numbers refer to that file unless a section names another.

## 1. The idea in one paragraph
Any deformable object has a **boundary surface**: cloth is the sheet, a soft body is the skin of its volume mesh, and a rope is the tube around its centreline. Deformation is described by how that surface is **stretched** and how it is **bent**. In differential geometry these are the surface's two fundamental forms:

| Form | Measures | Quantities used here |
|---|---|---|
| **First fundamental form** (the metric) | Lengths, angles and **area** | Area stretch `J = dA / dA₀`, membrane strain `E` |
| **Second fundamental form** | How the surface normal turns, i.e. **curvature** | Mean curvature `H`, Gaussian curvature `K`, principal curvatures `k₁`, `k₂` |

By Bonnet's theorem, the two forms (subject to the Gauss–Codazzi equations) determine the surface up to a rigid motion, so the pair is a complete description of shape. Nothing in it is rope-specific: the same code measures any triangle mesh whose connectivity stays fixed. `tube_surface` is the **only** rope-specific adapter.

## 2. Continuum definitions
A material point `p` on the rest surface `S₀` is mapped by the deformation `φ` to `x = φ(p)` on the current surface `S`.

- **First form:** `g = (dφ)ᵀ(dφ)`, restricted to the tangent plane.
  - Membrane (Green) strain: `E = ½(g − g₀)`.
  - Area stretch: `J = √(det g / det g₀)`, so that `dA = J dA₀`.
  - Principal stretches: the singular values of `dφ`.
- **Second form:** `b = dx · dn`, where `n` is the outward normal.
  - Shape operator: `Sh = g⁻¹ b`.
  - `H = ½ tr Sh`.
  - `K = det Sh`.
  - `k₁,₂ = H ± √(H² − K)`.
- **Sign convention:** a sphere of radius `R` has `H = 1/R` and `K = 1/R²`.
- **Deformation pair:** stretching is `E` (or `J − 1`); bending is `b − b₀`. This is the split used by Koiter shell theory and by Discrete Shells (Grinspun et al. 2003).

## 3. Discrete estimators on a triangle mesh: `SurfaceMeshForms` (line 80)
These are precomputed once from the rest mesh (`__init__`, lines 137–191); each call is then vectorised numpy, about 1.5 ms for the rope.

| Quantity | Discrete formula | Code |
|---|---|---|
| Face metric (first form) | `F_f = D_s D_m⁻¹`, `g_f = F_fᵀF_f`. The current edge Gram matrix needs only squared edge lengths: *the metric is the edge lengths*. | `__call__`, lines 284–291 |
| Face strain and stretches | `E_f = ½(g_f − I)`; stretches are `√` of the eigenvalues of `g_f` | lines 292–298 |
| Vertex area | Mixed Voronoi area (Meyer et al. 2003): the Voronoi split for non-obtuse faces, `A/2` or `A/4` otherwise | `_compute`, lines 218–223 |
| Area stretch per vertex | `A_v / A_v⁰` | `area_ratio_v`, line 303 |
| Mean curvature `H` | Cotangent Laplacian: `(1/2A_v) Σⱼ (cot αᵢⱼ + cot βᵢⱼ)(xᵢ − xⱼ) = 2H n` | lines 225–243 |
| Gaussian curvature `K` | Angle defect: `(2π − Σθ) / A_v`, with `π − Σθ` at boundary vertices | lines 244–245 |
| Principal curvatures | `k₁,₂ = H ± √max(H² − K, 0)` | line 246 |
| `∫K n dA` (Gauss-image moment) | `½ Σⱼ θᵢⱼ · unit(xᵢ − xⱼ)`, with dihedral angles `θ`. Exact for the polyhedral surface. | lines 248–259 |
| Check | Discrete Gauss–Bonnet: `Σ defect = 2π χ` (an identity; the residual checks the mesh) | line 302 |

## 4. The rope as a surface: `tube_surface` / `TubeSurfaceDescriptor` (lines 640, 690)
- **Mesh:** a 12-sided tube of radius `r` around the simulated centreline (61 nodes), with mitred rings and hemispherical end caps. Ring radius `ρ` is chosen so the 12-gon has the circle's **perimeter** 2πr, so the polygonal tube has exactly the smooth tube's surface area 2πr·L for any bending (`ring_radius`, line 439).
  - A discrete elastic rod keeps rigid circular cross-sections, so this tube *is* the rod's simulated surface.
- **Material samples:** the 21 samples evenly spaced along the rope's rest length. Each sample owns the rings in its window, and its measures are area-weighted means over those rings (`group_stats`, line 331).
- **Tube identities**, exact on the mitred tube while it does not fold (derived in the class docstring):

| Identity | Consequence |
|---|---|
| Ring-strip area = `2πr ds` for any bending | Per-sample area ratio = **exact axial stretch** |
| `∫H` over a strip = `π ds` | Area-mean `H ≈ 1/2r`; carries no bending information |
| `∫K n` over a strip = `−π κN ds` | Curvature vector `κN = −2r · moment` |
| `∫|K|` over a strip = `4|κ| ds` | `K_abs = 2|κ|/(πr)` |

- **Folds:** a sharp hairpin can make the tube's prisms pass through each other. Each sample is checked for that (`valid`, `fold_max`, lines 987–1007). Folded samples take the smooth tube's closed-form values instead (`smooth_tube_forms`, line 895; substitution at lines 1016–1027), so the signal stays continuous.

## 5. What the RL uses: `rope_routing/env/*`

**Observation**, per sample (`DEF_FEATURES`, `surface_route_env.py:124`, built in `generic_surface_env.py:_make_obs`):

| Feature | Scaled as |
|---|---|
| Position relative to the gripper; velocity | as is |
| Area strain | `10(J − 1)` |
| Mean-curvature change | `10r(H − H_rest)` |
| Gaussian curvature mean, absolute mean, min and max | `r²K` |
| Curvature vector; signed planar curvature | `× 3 cm` |
| `valid` flag | 0 / 1 |

Two globals are added: total area strain, and the Willmore energy `ΣH²A` relative to rest.

**Reward** (`generic_surface_env.py`):
- **Coverage:** the target is itself a surface, with vertices and their Gaussian curvature `K*`. For every target vertex, find the nearest current vertex; take the mean distance (`coverage`, `generic_surface_env.py` line 237).
- **Curvature mismatch:** `mean |r²K* − r²K|` at those nearest points.
- **Stretch penalty:** from `J` (first form), `−50 Σ max(0, J − 1 − 0.02)²` (`surface_route_env.py`, `step`).

**Why correspondence-free:** nearest-point matching never asks *which* material point should go where. Relabelling or reversing the object leaves the reward unchanged (`test_coverage_is_correspondence_free`).

## 6. Older generic layer (still used for sampling and strain checks)
- `material_graph.py`: material samples along the rest object and the rest geodesic distance matrix `geodesic` (Berenson 2013).
- `measures.py`: edge Green strain `ε = ½(|x_ij|²/|X_ij|² − 1)`, used by the start-state driver's strain guard.
- `shape_distance.py`: ordered / Hungarian / Chamfer distances, used by the old SurfaceRouteEnv only.

## 7. Tests
- **`tests/test_surface_forms.py`** (35 tests):

  | What is checked | Tests |
  |---|---|
  | Sphere `H = 1/R`, `K = 1/R²` and Gauss–Bonnet | `test_icosphere_area_curvature_and_gauss_bonnet` |
  | Cotangent Laplacian = area gradient | `test_cotan_laplacian_is_the_area_gradient` |
  | Torus `K`, `H` | `test_torus_patch_per_vertex_K_and_H` |
  | Tube identities are exact | `test_tube_identity_is_exact_for_any_turn`, `test_bent_tube_area_is_exact_and_quads_are_planar` |
  | Rigid-motion invariance | `test_rigid_motion_invariance`, `test_rigid_motion_with_nonuniform_stretch` |
  | **Object-agnostic: the same code on cloth** | `test_object_agnostic_cloth_bending_and_stretch`, `test_cloth_random_deformation_and_rigid_motion` |
  | Folds | `test_hairpin_folds`, `test_180_degree_folds`, `test_validity_rule_is_sufficient_for_exactness` |
- **`tests/test_measures.py`:** generic strain, affine `F`, and circle bending `1/R`.

## References
- Meyer, Desbrun, Schröder, Barr (2003), *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*: Voronoi areas, cotangent `H`, angle-defect `K`.
- Grinspun et al. (2003), *Discrete Shells*.
- Rusinkiewicz (2004), *Estimating Curvatures and Their Derivatives on Triangle Meshes*.
- Bergou et al. (2008), *Discrete Elastic Rods* (the rope model in the simulator).
- Berenson (2013), *Manipulation of Deformable Objects Without Modeling and Simulating Deformation*: coverage as a correspondence-free goal, geodesic distance from constraints.
