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
""" Base layer for FEM related operations. """
import abc
import six
import tensorflow as tf
import tenfem


__all__ = ['BaseFEMLayer', ]


@six.add_metaclass(abc.ABCMeta)
class BaseFEMLayer(tf.keras.layers.Layer):
    """ Base layer for FEM operations. """
    def __init__(self,
                 name: str = 'base_fem_layer'):
        """ Create a BaseFEMLayer instance

        Args:
            mesh: A finite element mesh.
            name: A python string giving the name of the layer op.
              Default `base_fem_layer`.

        """
        super(BaseFEMLayer, self).__init__(name=name)
