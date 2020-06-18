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

from tenfem.layers import BaseFEMLayer


class LinearEllipticOperator(BaseFEMLayer):
    """ tf.keras Layer for solving linear elliptic second order PDES. """
    def __init__(self,
                 diffusion_coefficient: Callable,
                 source: Callable,
                 transport_vector_field: Union[Callable, None] = None,
                 name='linear_elliptic_opeartor',
                 *args, **kwargs):
        super(LinearEllipticOperator, self).__init__(name=name, *args, **kwargs)

        self._diffusion_coefficient = diffusion_coefficient
        self._source = source
        self._transport_vector_field = transport_vector_field
