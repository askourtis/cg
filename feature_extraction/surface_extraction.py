"""
Surface Extraction and Clustering Algorithms

This module implements algorithms for segmenting 3D meshes into distinct
surface regions based on geometric properties, primarily surface normal
similarity. The main algorithm uses region growing to cluster faces with
similar orientations.

Functions:
    area_expansion: Main interface for surface clustering with configurable parameters
"""

from typing import Dict

import numpy as np

from compas.datastructures import Mesh


def expand_face_surface(face: int, mesh: Mesh, labels: Dict[int, int], current_label: int, threshold: float) -> None:
    """Perform region growing to cluster one surface region based on normal similarity.

    Starting from a seed face, this function uses a breadth-first search approach
    to find all connected faces with similar surface normals. This implements
    the core region growing algorithm for surface segmentation.

    Args:
        face: The starting face index (seed for region growing).
        mesh: The mesh to operate on.
        labels: A dictionary mapping face indices to their labels.
               Modified in-place during clustering.
        current_label: The label to assign to faces in this cluster.
        threshold: The cosine similarity threshold for including adjacent faces.
                  Range: [0, 1] where 1 = identical normals, 0 = perpendicular.

    Note:
        This function modifies the labels dictionary in-place. Faces with
        label 0 are considered unlabeled and available for clustering.
    """
    stack = [face]

    while stack:
        current_face = stack.pop()
        if labels[current_face] != 0:
            continue

        labels[current_face] = current_label

        face_normal = mesh.face_normal(current_face)
        for halfedge in mesh.face_halfedges(current_face):
            adjacent_face = mesh.halfedge_face(reversed(halfedge))
            if adjacent_face is None or labels[adjacent_face] != 0:
                continue

            adjacent_face_normal = mesh.face_normal(adjacent_face)
            similarity = np.dot(face_normal, adjacent_face_normal)

            if similarity >= threshold:
                stack.append(adjacent_face)

def filter_out_surfaces_based_on_number_of_faces(labels: Dict[int, int], min_faces: int, max_faces: int) -> None:
    """Remove surface clusters that don't meet size criteria.

    Filters out clusters that are too small (noise) or too large (over-segmentation)
    by resetting their labels to 0 (unlabeled). This helps ensure that only
    meaningful surface regions are retained for pattern matching.

    Args:
        labels: A dictionary mapping face indices to their labels.
               Modified in-place during filtering.
        min_faces: The minimum number of faces required for a surface to be retained.
                  Clusters smaller than this are considered noise.
        max_faces: The maximum number of faces allowed for a surface to be retained.
                  Set to 0 to disable maximum size filtering.

    Note:
        This function modifies the labels dictionary in-place, setting filtered
        cluster faces back to label 0 (unlabeled).
    """
    face_counts = {label: 0 for label in set(labels.values())}

    for label in labels.values():
        face_counts[label] += 1

    for label in labels.values():
        if label == 0:
            continue

        f_count = face_counts.get(label, 0)
        if f_count >= min_faces:
            if max_faces == 0 or f_count <= max_faces:
                continue

        for f, l in labels.items():
            if l == label:
                labels[f] = 0


def area_expansion(mesh: Mesh, *, threshold: float = 0.9, min_faces: int = 4, max_faces: int = 0) -> Dict[int, int]:
    """Cluster mesh surfaces using iterative region growing based on normal similarity.

    This is the main interface for surface clustering. It repeatedly finds unlabeled
    faces and grows surface regions from them until all faces are either clustered
    or determined to be noise.

    Args:
        mesh: The mesh to cluster into surface regions.
        threshold: The cosine similarity threshold for clustering (0.0-1.0).
                  Higher values create more distinct/separated clusters.
                  Typical range: 0.8-0.99
        min_faces: The minimum number of faces required for a surface to be retained.
                  Smaller clusters are filtered out as noise.
        max_faces: The maximum number of faces allowed for a surface to be retained.
                  Set to 0 to disable maximum size filtering. Useful to prevent
                  over-clustering of large flat surfaces.

    Returns:
        A dictionary mapping face indices to their cluster labels.
        Label 0 indicates unlabeled faces (noise or filtered out).
        Labels 1, 2, 3, ... indicate distinct surface clusters.

    Example:
        >>> labels = area_expansion(mesh, threshold=0.95, min_faces=10, max_faces=300)
        >>> num_clusters = len(set(labels.values())) - 1  # -1 to exclude label 0
    """
    labels = {k: 0 for k in mesh.face}

    current_label = 1
    while True:
        # Find the first unlabelled face
        face = next((face for face, label in labels.items() if label == 0), None)

        if face is None:
            break

        expand_face_surface(face, mesh, labels, current_label, threshold)

        current_label += 1

    filter_out_surfaces_based_on_number_of_faces(labels, min_faces, max_faces)
    return labels