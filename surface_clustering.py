from typing import Dict

import numpy as np

from compas.datastructures import Mesh



def surface_cluster(face: int, mesh: Mesh, labels: Dict[int, int], current_label: int, curvature_tol: float) -> None:
    """Given a starting face, cluster one surface region of the mesh, based on curvature."""
    stack = [face]

    while stack:
        current_face = stack.pop()
        if labels[current_face] != 0:
            continue

        labels[current_face] = current_label

        for vertex in mesh.face_vertices(current_face):
            if abs(mesh.vertex_curvature(vertex)) < curvature_tol:
                stack.extend(mesh.vertex_faces(vertex))

        mesh.face.pop(current_face, None)  # Ensure the face is removed from the mesh if needed
        mesh.remove_unused_vertices()


def surface_clustering(mesh: Mesh, curvature_tol: float = 0.1) -> np.ndarray:
    """An iterative process to cluster the surface of a mesh."""
    labels = {k: 0 for k in mesh.face}

    current_label = 1
    while True:
        # Find the first unlabelled face
        face = next((face for face, label in labels.items() if label == 0), None)

        if face is None:
            break

        surface_cluster(face, mesh, labels, current_label, curvature_tol)

        current_label += 1

    return labels
