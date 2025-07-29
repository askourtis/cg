from compas.datastructures import Mesh

from typing import Dict

import numpy as np


def labels_to_meshes(mesh: Mesh, labels: Dict[int, int]) -> Dict[int, Mesh]:
    """Convert labels to separate meshes.

    Arguments:
        mesh: The original mesh.
        labels: A dictionary mapping face indices to their labels.

    Returns:
        A dictionary mapping labels to submeshes."""

    meshes = {}
    for label in set(labels.values()):
        sub_faces = {f: mesh.face_vertices(f) for f in labels if labels[f] == label}
        sub_vertices = {v: [dx for dx in d.values()] for v, d in mesh.vertex.items()}
        submesh = Mesh.from_vertices_and_faces(sub_vertices, sub_faces)
        submesh.remove_unused_vertices()
        meshes[label] = submesh
    return meshes


def vertices_to_features(mesh: Mesh) -> Dict[int, np.ndarray]:
    """Map vertices to feature vectors.

    Arguments:
        mesh: The mesh to extract features from.

    Returns:
        A dictionary mapping vertex indices to feature vectors."""

    def vertex_avg_distance_from_all(mesh: Mesh, vertex: int) -> float:
        """Calculate the average distance of a vertex to all other vertices."""
        vertices = np.array(mesh.vertices_attributes('xyz'))
        distances = np.linalg.norm(vertices - vertices[mesh.vertex_index()[vertex]], axis=1)
        return np.mean(distances)

    def vertex_curvature(mesh: Mesh, vertex: int) -> float:
        """Calculate the curvature of a vertex based on its neighboring faces."""
        return mesh.vertex_curvature(vertex)

    r = {v: np.array([
        vertex_avg_distance_from_all(mesh, v),
        vertex_curvature(mesh, v)
    ]) for v in mesh.vertices()}

    r_max_per_dim = np.max(np.array(list(r.values())), axis=0)
    normalized = {v: r[v] / r_max_per_dim for v in r}

    return normalized
