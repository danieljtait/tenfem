# Copyright 2020 Daniel J. Tait
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from .triangle_mesh import TriangleMesh
from scipy.spatial import Delaunay
import numpy as np
import tensorflow as tf
from functools import lru_cache
from .matlab_trimesh_generator import MatlabTrimeshGenerator


def square(nx, ny, dtype=tf.float32):
    """ Create a square mesh in R2.

    Args:
        nx: Python integer, the number of nodes in the 'x' direction.
        ny: Python integer, the number of nodes in the 'y' direction.
        dtype: An optional `tf.DType` object giving the data-type of the mesh nodes.
          Default `tf.float32`.

    Returns:
        tri_mesh: A `TriangleMesh` object represent a mesh of the
          `nx` times `ny` unit square.
    """
    xx, yy = np.meshgrid(*(np.linspace(0., 1., n) for n in [nx, ny]))

    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    tri = Delaunay(nodes)

    tri_mesh = TriangleMesh(tri.points, tri.simplices, tri.convex_hull, dtype=dtype)
    tri_mesh.add_matplotlib_tri()

    return tri_mesh


@lru_cache(maxsize=8)
def circle(hmax: float, dtype: tf.DType = tf.float32) -> TriangleMesh:
    """ Create a `TriangleMesh` of a circular domain.

    Args:
        hmax: Average maximum edge length of triangles in the mesh.
        dtype: The datatype of the mesh nodes. Default `float32`

    Return:
        mesh: A `TriangleMesh` object of a circular domain.
    """
    thetas = np.linspace(0., 2*np.pi)
    verts = np.array([np.sin(thetas), np.cos(thetas)]).T.tolist()
    mesh = MatlabTrimeshGenerator.from_verts(verts, hmax, dtype=dtype)
    mesh.add_matplotlib_tri()
    return mesh
