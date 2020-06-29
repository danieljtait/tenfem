# Copyright 2020 Daniel J. Tait.
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
""" Layers to provide tapered sub-meshes. """
import tensorflow as tf
from tenfem.layers import MeshProvider


class TaperedSubmeshProvider(MeshProvider):
    """ Provides a submesh using mini-patch tapering. """
    def __init__(self,
                 mesh,
                 reference_element,
                 threshold,
                 name='tapered_submesh_provider'):
        """ Create a TaperedSubmeshProvider instance. """
        super(TaperedSubmeshProvider, self).__init__(mesh,
                                                     reference_element,
                                                     name=name)
        self.threshold = tf.constant(threshold)

    def _build_tapered_neighbors(self):
