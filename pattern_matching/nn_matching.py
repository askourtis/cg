"""
Nearest Neighbor Pattern Matching for 3D Meshes

This module implements a pattern matching system that uses nearest neighbor
algorithms to compare geometric features between 3D mesh surfaces. The approach
extracts characteristic features from vertices and uses bidirectional matching
to compute similarity scores.

Classes:
    NearestNeighborMatcher: Main class for training and predicting mesh similarity

The matching algorithm computes bidirectional similarity to ensure robust
pattern detection regardless of mesh orientation or vertex ordering.
"""

from compas.datastructures import Mesh

from utilities import vertices_to_features

import numpy as np
from sklearn.neighbors import NearestNeighbors


# Theoretical mean distance for random points in a unit square
# Used as normalization baseline for similarity computation
# Formula: 2/(15*sqrt(2)) + sqrt(2)/3 ≈ 0.565
MEAN_DISTANCE_RANDOM_LINE_SEGMENT = 2/(15 * np.sqrt(2)) + np.sqrt(2)/3

class NearestNeighborMatcher:
    """A pattern matcher that uses nearest neighbors to compare mesh geometric features.

    This class implements a bidirectional nearest neighbor approach for comparing
    3D mesh surfaces. It extracts geometric features from vertices and uses
    k-nearest neighbors to find correspondences between meshes.

    The similarity computation is bidirectional (A->B and B->A) to ensure
    robust matching regardless of mesh complexity differences.

    Attributes:
        feature_array (np.ndarray): Stored feature vectors from the fitted mesh.
        reduce (callable): Function to reduce distance arrays (default: np.max).

    Example:
        >>> matcher = NearestNeighborMatcher()
        >>> matcher.fit(template_mesh)
        >>> similarity = matcher.predict(test_mesh)  # Not normalized
    """

    def __init__(self, *, reduce=np.max):
        self.feature_array = np.array([])
        self.reduce = reduce

    def fit(self, mesh: Mesh) -> None:
        """Train the matcher on a template mesh by extracting its features.

        This method processes the template mesh and stores its feature vectors
        for later comparison with other meshes. The features capture geometric
        properties that are characteristic of the mesh's shape.

        Args:
            mesh: The template mesh to learn features from.
                 This becomes the reference pattern for matching.

        Note:
            Call this method once with your template/pattern mesh before
            using predict() to compare other meshes against it.
        """
        mesh = mesh
        features = vertices_to_features(mesh)
        self.feature_array = np.array(list(features.values()))

    def predict(self, mesh: Mesh) -> float:
        """Compute similarity between a test mesh and the fitted template mesh.

        Uses bidirectional nearest neighbor matching to compute a similarity score.
        The algorithm finds nearest neighbors in both directions (template->test
        and test->template) and returns the minimum similarity to ensure conservative
        matching.

        Args:
            mesh: The test mesh to compare against the fitted template.

        Returns:
            A similarity score where:
            - Values > 0.85 typically indicate good matches

        Note:
            The similarity is computed as 1 - (normalized_distance / baseline)
            where baseline is the expected distance for random patterns.
        """
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
