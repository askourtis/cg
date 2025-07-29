from compas.datastructures import Mesh

from utilities import vertices_to_features

import numpy as np
from sklearn.neighbors import NearestNeighbors


MEAN_DISTANCE_RANDOM_LINE_SEGMENT = 2/(15 * np.sqrt(2)) + np.sqrt(2)/3

class NearestNeighborMatcher:
    """A matcher that uses nearest neighbors to compare mesh features."""

    def __init__(self, *, reduce=np.max):
        self.feature_array = np.array([])
        self.reduce = reduce

    def fit(self, mesh: Mesh) -> None:
        """Fit the matcher to a mesh.

        Arguments:
            mesh: The mesh to fit the matcher to."""
        mesh = mesh
        features = vertices_to_features(mesh)
        self.feature_array = np.array(list(features.values()))

    def predict(self, mesh: Mesh) -> float:
        """Predict similarity between two meshes based on their features.

        Arguments:
            mesh: The mesh to compare against the fitted mesh."""
        features = vertices_to_features(mesh)
        features_np = np.array(list(features.values()))

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.feature_array)
        distances, _ = nn.kneighbors(features_np)
        similarity12 = 1 - self.reduce(distances) / MEAN_DISTANCE_RANDOM_LINE_SEGMENT

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(features_np)
        distances, _ = nn.kneighbors(self.feature_array)
        similarity21 = 1 - self.reduce(distances) / MEAN_DISTANCE_RANDOM_LINE_SEGMENT

        return min(similarity12, similarity21)
