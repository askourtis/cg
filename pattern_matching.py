# IDEA:
# 1. Find sharp edges
# 2. Find loops of sharp edges to bound a ridge
# 3. Carve out the ridge
# 4. Compute measure and curvature of the mesh (UV coordinates)
# 5. Run KNN mean distance on UV coordinates and produce a similarity measure
# 6. If similarity is above a threshold then label the vetices inside the boundary
# 7. Visualize the mesh with vertex colors based on labels


# %% 0. Imports + Definitions
from typing import Callable, Dict
import heapq

from compas_viewer import Viewer
from compas.colors import Color
from compas.datastructures import Mesh

import matplotlib.pyplot as plt
%matplotlib ipympl


def geodesic_distance(mesh: Mesh, vertex_a: int, vertex_b: int) -> float:
    """Calculate the geodesic distance between two vertices in the mesh using Dijkstra's algorithm."""
    queue = [(0.0, vertex_a)]
    visited = set()
    distances = {vertex_a: 0.0}

    while queue:
        dist, current = heapq.heappop(queue)
        if current == vertex_b:
            return dist
        if current in visited:
            continue
        visited.add(current)
        for neighbor in mesh.vertex_neighbors(current):
            edge_length = mesh.edge_length((current, neighbor))
            new_dist = dist + edge_length
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))
    return float('inf')


def measure_vertex(mesh: Mesh, vertex: int) -> float:
    """Measure a vertex in the mesh."""
    return sum(geodesic_distance(mesh, vertex, other) for other in mesh.vertex) / len(mesh.vertex)

def map_vertex_to_measure(mesh: Mesh) -> Dict[int, float]:
    """Map each vertex of the mesh to its measure."""
    return {vertex: measure_vertex(mesh, vertex) for vertex in mesh.vertices()}




# %% 2. load mesh and compute vertex measures + curvature
mesh = Mesh.from_stl("part1.stl")

measure_map = map_vertex_to_measure(mesh)
curvature_map = {k: mesh.vertex_curvature(k) for k in mesh.vertices()}

normalized_measure_map = {
    vertex: (measure - min(measure_map.values())) / (max(measure_map.values()) - min(measure_map.values()))
    for vertex, measure in measure_map.items()
}

# %% 3. Visualize the mesh with vertex colors based on measures
viewer = Viewer()
viewer.scene.clear()
viewer.scene.add(mesh, use_vertexcolors=True,  vertexcolor={vertex: Color.from_hsv(0.6 * measure, 1.0, 1.0) for vertex, measure in normalized_measure_map.items()})
viewer.show()


# %% 3. Visualize curvature vs measure
plt.figure(figsize=(12, 6))

plt.title("Vertex Curvature vs Measure")
XY = [ (normalized_measure_map[v], curvature_map[v]) for v in mesh.vertices() ]
plt.scatter(*zip(*XY), c='b', label='Measure')
plt.xlabel("Measure (Normalized)")
plt.ylabel("Curvature")
plt.grid()

plt.tight_layout()
plt.show()
# %%
