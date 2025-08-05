"""
3D Mesh Pattern Matching System

This script demonstrates a complete pipeline for finding similar geometric patterns
within 3D meshes. It uses surface clustering to segment meshes into regions and
nearest neighbor matching to identify similar surface patterns.

The workflow includes:
1. Loading scene and part meshes
2. Clustering the scene mesh into surface regions based on normal similarity
3. Training a pattern matcher on the part mesh features
4. Finding matching regions in the scene mesh
5. Visualizing the results with color-coded highlighting

Usage:
    python pattern_matching.py

Requirements:
    - STL files in resources/ directory (scene.stl, part.stl)
    - All dependencies listed in requirements.txt
"""

# %% Imports + Definitions
from compas_viewer import Viewer
from compas.colors import Color
from compas.datastructures import Mesh

from feature_extraction import surface_extraction as sf
from pattern_matching.nn_matching import NearestNeighborMatcher
from utilities import labels_to_meshes

# %% Load the scene mesh and part mesh
scene_mesh = Mesh.from_stl('resources/scene.stl')
part_mesh = Mesh.from_stl('resources/part.stl')

# %% Surface clustering
scene_labels = sf.area_expansion(scene_mesh, threshold=0.95, min_faces=10, max_faces=300)
submeshes = labels_to_meshes(scene_mesh, scene_labels)

if False:  # visualize the clustering result
    fcol = {k: Color.from_number(l/max(scene_labels.values())) if l != 0 else Color.black() for k, l in scene_labels.items()}
    viewer = Viewer()
    viewer.scene.clear()
    obj1 = viewer.scene.add(scene_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
    viewer.show()

# %% Pattern matching using nearest neighbors
nnm = NearestNeighborMatcher()
nnm.fit(part_mesh)

predictions = {label: nnm.predict(submesh) for label, submesh in submeshes.items() if label != 0}

threshold = 0.85
matched_labels = [label for label, score in predictions.items() if score >= threshold]

# %% Visualize the pattern matching results
fcol = {f: Color.red() for f in scene_mesh.faces() if scene_labels[f] in matched_labels}
viewer = Viewer()
viewer.scene.clear()
obj1 = viewer.scene.add(scene_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
viewer.show()
