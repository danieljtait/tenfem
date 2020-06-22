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
""" Layer to do interpolation on a triangular mesh. """
import tensorflow as tf
import tenfem
from tenfem.layers import BaseFEMLayer


def barycentric(p, a, b, c):
    """
    Find the barycentric coordinate of the point `p` with respect to the triangle
    `[a, b, c]`.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = tf.reduce_sum(v0 * v0, axis=-1)  # dot product along batches
    d01 = tf.reduce_sum(v0 * v1, axis=-1)
    d11 = tf.reduce_sum(v1 * v1, axis=-1)
    d20 = tf.reduce_sum(v2 * v0, axis=-1)
    d21 = tf.reduce_sum(v2 * v1, axis=-1)

    denom = d00 * d11 - d01 * d01
    r = (d11 * d20 - d01 * d21) / denom
    s = (d00 * d21 - d01 * d20) / denom
    t = 1. - r - s
    return r, s, t


class TriangleMeshInterpolator(BaseFEMLayer):
    """ Layer to do interpolation on a triangular mesh.

    ToDo(dan):
        remove dependency on trifinder

    """
    def __init__(self,
                 tri_finder=None,
                 name='trimesh_interpolate',
                 *args, **kwargs):
        super(TriangleMeshInterpolator, self).__init__(name=name, *args, **kwargs)
        self.tri_finder = tri_finder

    def call(self, inputs):
        if self.tri_finder is None:
            raise NotImplementedError('Currently a tri_finder must be supplied.')
        else:
            tri_finder = self.tri_finder

        points, u, mesh_tensor_repr = inputs
        mesh = tenfem.mesh.utils.mesh_from_tensor_repr(mesh_tensor_repr,
                                                       self.reference_element)
        nodal_vals = tf.gather(u[..., 0], mesh.elements)

        # find the element indices for points
        tris = tf.numpy_function(lambda x: tri_finder(x[..., 0], x[..., 1]),
                                 [points, ],
                                 Tout=tf.int32)

        not_in_mesh = tris < 0  # store a reference to those points not in the mesh

        tri_indices = tf.where(not_in_mesh, tf.zeros_like(tris), tris)
        elements = tf.gather(mesh.elements[..., :3], tri_indices)

        element_nodes = tf.gather(mesh.nodes, elements)

        r, s, t = barycentric(points,
                              element_nodes[..., 0, :],
                              element_nodes[..., 1, :],
                              element_nodes[..., 2, :])

        shape_fn, _ = self.reference_element.shape_function(r, s)

        element_nodal_vals = tf.gather(nodal_vals, tri_indices)

        interp_vals = tf.reduce_sum(element_nodal_vals * shape_fn, axis=-1)

        # mask back the bad tris
        interp_vals = tf.where(not_in_mesh, tf.zeros_like(interp_vals), interp_vals)
        return interp_vals[..., tf.newaxis]
