"""
Utility Functions for 3D Mesh Processing

This module provides essential utility functions for mesh manipulation,
feature extraction, and geometric analysis used throughout the pattern
matching pipeline.

Functions:
    labels_to_meshes: Convert face labels to separate mesh objects
    vertices_to_features: Extract geometric features from mesh vertices

The feature extraction focuses on local geometric properties such as
curvature and spatial relationships that are useful for pattern matching.
"""

from compas.datastructures import Mesh

from typing import Dict

import numpy as np


def labels_to_meshes(mesh: Mesh, labels: Dict[int, int]) -> Dict[int, Mesh]:
    """Convert face labels to separate mesh objects.

    Takes a mesh with labeled faces and creates individual mesh objects
    for each label group. This is useful for processing clustered surface
    regions independently.

    Args:
        mesh: The original mesh containing all faces.
        labels: A dictionary mapping face indices to their cluster labels.
                Label 0 typically indicates unlabeled/background faces.

    Returns:
        A dictionary mapping labels to their corresponding submesh objects.
        Each submesh contains only the faces with that specific label.

    Note:
        The function automatically removes unused vertices from each submesh
        to ensure clean geometry.
    """

    meshes = {}
    for label in set(labels.values()):
        sub_faces = {f: mesh.face_vertices(f) for f in labels if labels[f] == label}
        sub_vertices = {v: [dx for dx in d.values()] for v, d in mesh.vertex.items()}
        submesh = Mesh.from_vertices_and_faces(sub_vertices, sub_faces)
        submesh.remove_unused_vertices()
        meshes[label] = submesh
    return meshes


def vertices_to_features(mesh: Mesh) -> Dict[int, np.ndarray]:
    """Extract geometric feature vectors from mesh vertices.

    Computes characteristic geometric properties for each vertex that can
    be used for pattern matching and surface analysis. Features are
    normalized to ensure scale-invariant comparison.

    Args:
        mesh: The mesh to extract features from.

    Returns:
        A dictionary mapping vertex indices to normalized feature vectors.
        Each feature vector contains:
        - Average distance to all other vertices (global shape context)
        - Local curvature estimate (local surface characteristics)

    Note:
        Features are normalized by their maximum values across all vertices
        to ensure values are in the range [0, 1] for consistent comparison.
    """

    # TODO: Use dependency injection for feature extraction functions
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
