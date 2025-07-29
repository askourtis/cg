# %% Imports + Definitions
from compas_viewer import Viewer
from compas.colors import Color
from compas.datastructures import Mesh

from feature_extraction.surface_extraction import surface_extraction_area_expansion

# %% load the scene mesh and part mesh
scene_mesh = Mesh.from_stl('scene.stl')
part_mesh = Mesh.from_stl('part.stl')

# %% Surface clustering
labels = surface_extraction_area_expansion(scene_mesh, threshold=0.95, min_faces=4)
fcol = {k: Color.from_number(l/max(labels.values())) if l != 0 else Color.black() for k, l in labels.items()}


# %% Map part mesh to feature vector



# %% Visualize the clustering result
viewer = Viewer()
viewer.scene.clear()
#render mesh with vertex colors, faces, verticies, edges
obj1 = viewer.scene.add(scene_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
viewer.show()
# %%
