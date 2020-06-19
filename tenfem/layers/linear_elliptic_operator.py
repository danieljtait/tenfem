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
""" Layer to form and solve linear elliptic operators. """
from typing import Callable, Union
import tensorflow as tf
import tenfem
from tenfem.layers import BaseFEMLayer, SolveDirichletProblem


class LinearEllipticOperator(BaseFEMLayer):
    """ tf.keras Layer for solving linear elliptic second order PDES. """
    def __init__(self,
                 diffusion_coefficient: Callable,
                 source: Callable,
                 transport_vector_field: Union[Callable, None] = None,
                 name: str ='linear_elliptic_opeartor',
                 boundary_condition: str = 'dirichlet',
                 reference_element=None,
                 boundary_values=None):
        """ Creates a LinearEllipticOperator instance.

        Args:
            diffusion_coefficient: A callable giving the diffusion
              of the elliptic operator. Will be queried at the mesh
              quadrature nodes, and reshaped to
              `[-1, n_elements, element_dim]`.
            source: A callable giving the source term of the elliptic
              operator equations. Will be called at the mesh quadrature
              nodes, and must be able to be reshape to
              `[-1, n_elements, element_dim]`.
            transport_vector_field: An optional callable returing the
              values of the transport vector field at the mesh quadrature
              nodes and so should return a 2-vector at each point.
              Default: None, in which case there is no transport term.
            name: A python string giving the name of this layer, defaults to
              'linear_elliptic_operator'.
            boundary_condition: A python string giving the type of boundary
              condition, should be one of `dirichlet`, `neumann` or `mixed.

        Raises:
            NotImplementedError: `If boundary_condition != 'dirichlet'`
              currently only Dirichlet boundary conditions are implemented.
            ValueError: `If boundary_condition == 'dirichlet'` but no function
              is supplied to provide the values on the boundary.
        """
        super(LinearEllipticOperator, self).__init__(name=name,
                                                     reference_element=reference_element)

        self._diffusion_coefficient = diffusion_coefficient
        self._source = source
        self._transport_vector_field = transport_vector_field
        self._boundary_condition = boundary_condition

        if self._boundary_condition == 'dirichlet':
            try:
                self._boundary_values = boundary_values
                self._solve_layer = SolveDirichletProblem(
                    boundary_values,
                    reference_element=self.reference_element)
            except KeyError:
                raise ValueError(''.join((
                    'If this model uses Dirichlet boundary conditions then ',
                    'the values of the solution on the boundary must be supplied',
                    'or else a callable to evaluate these solutions')))
        else:
            raise NotImplementedError('Only Dirichlet boundary conditions currently implemented')

    @property
    def solve_layer(self):
        """ Layer used to solve the problem. """
        return self._solve_layer

    @property
    def diffusion_coefficient(self):
        """ Diffusion coefficient of the elliptic operator. """
        return self._diffusion_coefficient

    @property
    def source(self):
        """ Source term of the elliptic operator. """
        return self._source

    def call(self, mesh_tensor_repr):
        mesh = tenfem.mesh.utils.mesh_from_tensor_repr(mesh_tensor_repr,
                                                       self.reference_element)
        element = self.reference_element

        # shape [mesh.n_elements, element_dim, spatial_dim]
        mesh_quadrature_nodes = element.get_quadrature_nodes(mesh)

        element_dim = tf.shape(mesh_quadrature_nodes)[-2]
        spatial_dim = tf.shape(mesh.nodes)[-1]

        # evaluate the diffusion coefficient at the quadrature nodes
        flat_mesh_quadrature_nodes = tf.reshape(mesh_quadrature_nodes, [-1, spatial_dim])
        diffusion_coeff_vals = tf.reshape(
            self.diffusion_coefficient(flat_mesh_quadrature_nodes),
            [-1, mesh.n_elements, element_dim])

        local_stiffness_mat = tenfem.fem.assemble_local_stiffness_matrix(
            diffusion_coeff_vals, mesh, element)
        local_stiffness_mat = tf.reshape(local_stiffness_mat,
                                         [-1, mesh.n_elements, element_dim, element_dim])

        batch_size = tf.shape(local_stiffness_mat)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        global_stiffness_mat = tenfem.fem.scatter_matrix_to_global(
            local_stiffness_mat, elements, mesh.n_nodes)

        source_vals = tf.reshape(
            self.source(flat_mesh_quadrature_nodes),
            [-1, mesh.n_elements, element_dim])

        local_load_vector = tenfem.fem.assemble_local_load_vector(
            source_vals, mesh, element)
        local_load_vector = tf.reshape(local_load_vector,
                                       [-1, mesh.n_elements, element_dim])

        batch_size = tf.shape(local_load_vector)[0]
        elements = tf.tile(mesh.elements[tf.newaxis, ...], [batch_size, 1, 1])

        load_vector = tenfem.fem.scatter_vector_to_global(
            local_load_vector, elements, mesh.n_nodes)

        return self.solve_layer((global_stiffness_mat, load_vector, mesh_tensor_repr))
