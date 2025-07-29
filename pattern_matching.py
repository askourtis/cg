# %% Imports + Definitions
from compas_viewer import Viewer
from compas.colors import Color
from compas.datastructures import Mesh

from feature_extraction import surface_extraction as sf
from pattern_matching.nn_matching import NearestNeighborMatcher
from utilities import labels_to_meshes

# %% load the scene mesh and part mesh
scene_mesh = Mesh.from_stl('scene.stl')
part_mesh = Mesh.from_stl('part.stl')

# %% Surface clustering
scene_labels = sf.area_expansion(scene_mesh, threshold=0.95, min_faces=10, max_faces=300)
submeshes = labels_to_meshes(scene_mesh, scene_labels)

if False:  # visualize the clustering result
    fcol = {k: Color.from_number(l/max(scene_labels.values())) if l != 0 else Color.black() for k, l in scene_labels.items()}
    viewer = Viewer()
    viewer.scene.clear()
    obj1 = viewer.scene.add(scene_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
    viewer.show()

# %% Map part mesh to feature vector
nnm = NearestNeighborMatcher()
nnm.fit(part_mesh)

predicitions = {label: nnm.predict(submesh) for label, submesh in submeshes.items() if label != 0}

threshold = 0.85
matched_labels = [label for label, score in predicitions.items() if score >= threshold]

# %% Visualize the clustering result
fcol = {f: Color.red() for f in scene_mesh.faces() if scene_labels[f] in matched_labels}
viewer = Viewer()
viewer.scene.clear()
obj1 = viewer.scene.add(scene_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
viewer.show()
