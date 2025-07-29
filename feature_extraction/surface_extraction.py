from typing import Dict

import numpy as np

from compas.datastructures import Mesh


def expand_face_surface(face: int, mesh: Mesh, labels: Dict[int, int], current_label: int, threshold: float) -> None:
    """Given a starting face, cluster one surface region of the mesh, based on surface normal similarity.

    Arguments:
        face: The starting face index.
        mesh: The mesh to operate on.
        labels: A dictionary mapping face indices to their labels.
        current_label: The label to assign to the clustered faces.
        threshold: The cosine similarity threshold for clustering.
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
    """Filter out small surfaces based on the number of faces.

    Arguments:
        labels: A dictionary mapping face indices to their labels.
        min_faces: The minimum number of faces required for a surface to be retained.
        max_faces: The maximum number of faces allowed for a surface to be retained.
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
    """An iterative process to cluster the surface of a mesh based on surface normal similarity.
    Arguments:
        mesh: The mesh to operate on.
        threshold: The cosine similarity threshold for clustering.
        min_faces: The minimum number of faces required for a surface to be retained.
    Returns:
        A dictionary mapping face indices to their labels, where 0 indicates unlabelled faces.
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