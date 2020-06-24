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
""" Base class for reference elements. """
import abc
import six
import tensorflow as tf

__all__ = ['BaseReferenceElement', ]


@six.add_metaclass(abc.ABCMeta)
class BaseReferenceElement(tf.Module):
    """ Base class for reference elements. """
    def __init__(self,
                 name: str = 'base_reference_element'):
        """ Create a BaseReferenceElement instance

        Args:
            name: A python string giving the name of the base element,
              Default: `base_reference_element`.

        """
        super(BaseReferenceElement, self).__init__(name=name)

    @property
    def element_dim(self):
        """ Number of nodes needed to define an element. """
        return self._element_dim

    @property
    def embedding_dim(self):
        """ Dimension of the space in which this element is embedded. """
        return self._embedding_dim

    @property
    def dtype(self):
        """ Data-type of the element. """
        return self._dtype

    @property
    def shape_function(self):
        """ Shape functions on the element. """
        return self._shape_fn

    @property
    def quadrature_order(self):
        """ The order of Gaussian quadrature performed on the reference element. """
        return self._quadrature_order

    @property
    def n_shape_functions(self):
        """ Number of shape functions on the element. """
        return self._n_shape_functions
