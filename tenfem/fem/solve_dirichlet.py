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
""" Solve a FEM problem with Dirichlet boundary conditions. """
from typing import Union, Tuple
import tensorflow as tf


def dirichlet_form_linear_system(lhs_matrix: tf.Tensor,
                                 rhs_vector: tf.Tensor,
                                 node_types: tf.Tensor,
                                 boundary_vals: Union[None, tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Forms the linear system Ax = b of the Dirichlet problem.

    Forms the matrix A and vector b in the dirichlet problem A @ u = b
    where the problem is reduced by collecting only the free interior terms.

    This function expects a matrix, this is because the return is not
    a fixed shape and so we can't carry out a straight forward batching
    of this function.

    Args:
        lhs_matrix: A float `Tensor`, the left-hand side of the system of equations.
          Must be a rank 2 square tensor.
        rhs_vector: A float `Tensor`, the right-hand side of the system of equations.
        node_types: An integer `Tensor` giving the node types of the mesh, should be a
          rank 1 tensor of length equation to lhs_matrix.shape[0].
        boundary_vals: Either the fixed values of the system of equations, or else None
          in which case it is equivalent to a vector of zeros. Default is `None`.

    Returns:
        lhs_matrix: `tf.Tensor`
        rhs_tensor: `tf.Tensor`.
    """
    int_nodes = tf.where(node_types == 0)[:, 0]
    bnd_nodes = tf.where(node_types == 1)[:, 0]

    matrix_int = tf.gather(tf.gather(lhs_matrix, int_nodes, axis=-1), int_nodes, axis=-2)

    vec_int = tf.gather(rhs_vector, int_nodes, axis=-2)
    if boundary_vals is not None:
        matrix_int_bnd = tf.gather(tf.gather(lhs_matrix, bnd_nodes, axis=-1), int_nodes, axis=-2)
        vec_int = vec_int - tf.matmul(matrix_int_bnd, boundary_vals)

    return matrix_int, vec_int


def solve_dirichlet_form_linear_system(bilinear_form: tf.Tensor,
                                       load: tf.Tensor,
                                       node_types: str,
                                       boundary_vals: Union[tf.Tensor, None],
                                       method: str = 'lu') -> tf.Tensor:
    """ Solves the linear FEM problem with Dirichlet boundary conditions.

    Args:
        bilinear_form: A rank 2 matrix of shape `[n_nodes, n_nodes]` representing
          the bilinear form of the problem.
        load: A vector of shape `[n_nodes, 1]` giving the value of the load vector
          at the mesh nodes.
        node_types: An integer tensor giving the node types in the mesh.
          `node_types == 0` gives the interior nodes, `node_types == 1`
          gives the boundary nodes. The remaining are invalid nodes introduced for
          padding.
        boundary_vals: A float `Tensor` giving the value of the solution on
          the boundary nodes.
          Default: `None` then the problem is treated as being equal to zero
          on the boundary.
        method: A python string identifying the method used to solve the linear
          system.

    Returns:
        solution: A float `Tensor` of shape `[n_nodes, 1]` giving the nodal
          coefficient vector of the FEM solution to the Dirichlet problem.

    """
    stiffness_interior, load_interior = dirichlet_form_linear_system(
        bilinear_form,
        load,
        node_types,
        boundary_vals)

    if method == 'lu':
        lu, p = tf.linalg.lu(stiffness_interior)
        uo = tf.linalg.lu_solve(lu, p, load_interior)
    else:
        raise NotImplementedError('Unrecognised method {}'.format(method))

    n_nodes = tf.shape(bilinear_form)[-1]

    indices = tf.where(node_types == 0)
    bnd_indices = tf.where(node_types == 1)

    u = tf.scatter_nd(indices, uo[:, 0], shape=[n_nodes])
    if boundary_vals is not None:
        u = tf.tensor_scatter_nd_update(u, bnd_indices, boundary_vals[:, 0])
    return u[..., tf.newaxis]
