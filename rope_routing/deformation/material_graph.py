"""Object-agnostic material-point graph (notes/A_deformation_math.md, section 6).

A deformable object is described by K material samples defined in its rest configuration. Each sample is a fixed
linear blend of simulator nodes, so a sample always refers to the same piece of material, whatever the deformation.
The graph also stores rest-neighbour edges between samples and the rest geodesic distance between every pair of
samples (shortest paths along the material, Berenson 2013).

Positions are in metres in the world frame.

All environments build the graph with :func:`from_polyline` (21 samples evenly spaced along the rope). The final
``GenericSurfaceEnv`` uses it for the per-sample positions and velocities of its observation and for the rest arclength
of each sample (``geodesic[:, 0]``); the edge-based measures of ``measures.py`` serve the legacy ``GraspRouteEnv`` and
the strain guard of the start-state driver.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass
class MaterialGraph:
    """Material samples of a deformable object, defined on its rest configuration.

    Attributes
    ----------
    rest_points : np.ndarray, shape (K, 3)
        Rest positions of the material samples [m].
    weights : np.ndarray, shape (K, N)
        Fixed blend weights: sample positions = ``weights @ simulator_nodes``.
    edges : np.ndarray of int, shape (E, 2)
        Pairs of rest-neighbour samples.
    geodesic : np.ndarray, shape (K, K)
        Rest geodesic distance between samples along the material [m].

    Examples
    --------
    >>> rest = np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)]
    >>> graph = from_polyline(rest, 3)
    >>> graph.num_samples, graph.weights.shape, graph.edges.tolist()
    (3, (3, 11), [[0, 1], [1, 2]])
    >>> graph.samples(2.0 * rest).round(3).tolist()
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    """

    rest_points: np.ndarray
    weights: np.ndarray
    edges: np.ndarray
    geodesic: np.ndarray

    @property
    def num_samples(self) -> int:
        """Number of material samples K."""
        return self.rest_points.shape[0]

    def samples(self, nodes: np.ndarray) -> np.ndarray:
        """Current positions of the material samples.

        Parameters
        ----------
        nodes : np.ndarray, shape (N, 3)
            Current simulator node positions [m].

        Returns
        -------
        np.ndarray, shape (K, 3)
            Current sample positions [m].
        """
        return self.weights @ nodes

    def neighbours(self) -> list[np.ndarray]:
        """Rest neighbours of every sample, from the edge list.

        Returns
        -------
        list of np.ndarray of int
            Element k holds the indices of the samples that share an edge with sample k.

        Examples
        --------
        >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
        >>> [members.tolist() for members in graph.neighbours()]
        [[1], [0, 2], [1]]
        """
        neighbour_lists = [[] for _ in range(self.num_samples)]
        for first, second in self.edges:
            neighbour_lists[first].append(second)
            neighbour_lists[second].append(first)
        return [np.array(members, dtype=int) for members in neighbour_lists]


def geodesic_distances(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Shortest-path distances between all points along the edges of a graph (Dijkstra).

    Parameters
    ----------
    points : np.ndarray, shape (K, 3)
        Point positions [m].
    edges : np.ndarray of int, shape (E, 2)
        Undirected edges; each edge is weighted by its Euclidean length.

    Returns
    -------
    np.ndarray, shape (K, K)
        Geodesic distances [m]; ``inf`` between disconnected points.

    Examples
    --------
    >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    >>> geodesic_distances(points, np.array([[0, 1], [1, 2]])).tolist()
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    """
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    count = points.shape[0]
    graph = csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(count, count))
    return shortest_path(graph, directed=False)


def from_polyline(rest_nodes: np.ndarray, num_samples: int) -> MaterialGraph:
    """Material graph of a one-dimensional object: samples evenly spaced in rest arclength, both ends included.

    Each sample interpolates linearly between the two simulator nodes that bracket its rest arclength; consecutive
    samples are connected by an edge.

    Parameters
    ----------
    rest_nodes : np.ndarray, shape (N, 3)
        Rest positions of the simulator nodes along the object [m].
    num_samples : int
        Number of samples K.

    Returns
    -------
    MaterialGraph
        The graph; ``geodesic[:, 0]`` is the rest arclength of each sample from node 0.

    Examples
    --------
    >>> graph = from_polyline(np.c_[np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], 3)
    >>> graph.rest_points.round(3).tolist()
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    >>> graph.geodesic.round(3).tolist()
    [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]]
    """
    segment_lengths = np.linalg.norm(np.diff(rest_nodes, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    sample_arclength = np.linspace(0.0, arclength[-1], num_samples)
    num_nodes = rest_nodes.shape[0]
    weights = np.zeros((num_samples, num_nodes))
    for sample, position in enumerate(sample_arclength):
        # Bracketing segment (the last segment for the far end) and the position within it.
        node = min(np.searchsorted(arclength, position, side="right") - 1, num_nodes - 2)
        fraction = (position - arclength[node]) / (arclength[node + 1] - arclength[node])
        weights[sample, node] = 1.0 - fraction
        weights[sample, node + 1] = fraction
    rest_points = weights @ rest_nodes
    edges = np.stack([np.arange(num_samples - 1), np.arange(1, num_samples)], axis=1)
    return MaterialGraph(rest_points=rest_points, weights=weights, edges=edges,
                         geodesic=geodesic_distances(rest_points, edges))
