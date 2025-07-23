# %% Imports
from random import random

from compas.colors import Color
from compas.geometry import Box, Sphere, Cylinder
from compas.geometry import Frame
from compas.geometry import Translation
from compas_viewer import Viewer
from compas.files import OBJ, STL
from compas.datastructures import Mesh


import numpy as np
from sklearn.manifold import Isomap
from skimage.feature import hog
import matplotlib.pyplot as plt

from surface_clustering import surface_clustering

# %% load the scene mesh and part mesh
scene_mesh = Mesh.from_stl('model.stl')
part_mesh = Mesh.from_stl('part.stl')

# %% Surface clustering
labels = surface_clustering(part_mesh.copy(), 1e-3)
fcol = {k: Color.from_i(l/10) for k, l in labels.items()}

# %% Visualize the clustering result
viewer = Viewer()
viewer.scene.clear()
#render mesh with vertex colors, faces, verticies, edges
obj1 = viewer.scene.add(part_mesh, facecolor=fcol, show_faces=True, show_vertices=True, show_edges=True)
viewer.show()
# %%
