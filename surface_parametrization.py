# %% 0. Imports
import numpy as np
%matplotlib ipympl
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from compas.datastructures import Mesh
from compas.geometry import (
    subtract_vectors,
    cross_vectors,
    dot_vectors,
    length_vector
)
from compas.files import OBJ, STL
import compas
from compas_viewer import Viewer

# %% 1. Load mesh
mesh = Mesh.from_stl("part.stl")  # built-in mesh (triangular)

# Index vertices
vertex_index = {v: i for i, v in enumerate(mesh.vertices())}
index_vertex = {i: v for v, i in vertex_index.items()}
n = len(vertex_index)

# %% 2. Cotangent Laplacian
L = lil_matrix((n, n))

for face in mesh.faces():
    v0, v1, v2 = mesh.face_vertices(face)
    i0 = vertex_index[v0]
    i1 = vertex_index[v1]
    i2 = vertex_index[v2]

    p0 = mesh.vertex_coordinates(v0)
    p1 = mesh.vertex_coordinates(v1)
    p2 = mesh.vertex_coordinates(v2)

    # Angles: cotangent weights
    def cotangent(a, b, c):
        u = subtract_vectors(b, a)
        v = subtract_vectors(c, a)
        cross = length_vector(cross_vectors(u, v))
        dot = dot_vectors(u, v)
        return dot / cross if cross != 0 else 0

    cot0 = cotangent(p1, p0, p2)
    cot1 = cotangent(p2, p1, p0)
    cot2 = cotangent(p0, p2, p1)

    for (i, j, cot) in [(i1, i2, cot0), (i2, i0, cot1), (i0, i1, cot2)]:
        L[i, j] -= cot
        L[j, i] -= cot
        L[i, i] += cot
        L[j, j] += cot

# %% 3. Boundary conditions
boundary = mesh.vertices_on_boundary()[:-1]
b_indices = [vertex_index[v] for v in boundary]
interior = [v for v in mesh.vertices() if v not in boundary]
i_indices = [vertex_index[v] for v in interior]

# Map boundary to unit circle
m = len(b_indices)
boundary_pos = np.zeros((n, 2))
for k, i in enumerate(b_indices):
    theta = 2 * np.pi * k / m
    boundary_pos[i] = [np.cos(theta), np.sin(theta)]

# %% 4. Solve Laplace system
L = L.tocsr()
Lii = L[i_indices, :][:, i_indices]
Lib = L[i_indices, :][:, b_indices]

bx = boundary_pos[b_indices, 0]
by = boundary_pos[b_indices, 1]

rhs_x = -Lib @ bx
rhs_y = -Lib @ by

ux = spsolve(Lii, rhs_x)
uy = spsolve(Lii, rhs_y)

# Combine all positions
uv = np.zeros((n, 2))
uv[b_indices] = boundary_pos[b_indices]
for k, i in enumerate(i_indices):
    uv[i] = [ux[k], uy[k]]

# %% 5. Visualize flattened mesh
plt.figure(figsize=(6, 6))
for face in mesh.faces():
    idx = [vertex_index[v] for v in mesh.face_vertices(face)]
    pts = np.array([uv[i] for i in idx + [idx[0]]])
    plt.plot(pts[:, 0], pts[:, 1], 'k-')
plt.scatter(bx, by, c='red', s=10, label='Boundary Conditions')
plt.scatter(uv[i_indices, 0], uv[i_indices, 1], c='blue', s=10, label='Interior Vertices')

plt.title("Harmonic Parameterization (Scipy + COMPAS)")
plt.axis('equal')
plt.show()
# %% 3D visualization via compas_viewer
viewer = Viewer()
viewer.scene.clear()
viewer.scene.add(mesh, show_faces=True, show_vertices=True, show_edges=True)
viewer.show()

# %%
