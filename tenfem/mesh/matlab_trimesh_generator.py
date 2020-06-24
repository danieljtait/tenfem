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
""" Use matlab functions to provide meshes. """
import tensorflow as tf
import numpy as np
from tenfem.mesh import TriangleMesh


# string of the Matlab function
# to triangulate a region.
meshfrompolyverts = (
    'function [p, e, t] = meshfrompolyverts(X, hmax)\n'
    '  pg = polyshape(X(:, 1), X(:, 2))\n',
    '  tr = triangulation(pg);\n',
    '  model = createpde();\n',
    '  geom = geometryFromMesh(model, tr.Points\', tr.ConnectivityList\');\n',
    '  femmesh = generateMesh(model, \'Hmax\', hmax, \'GeometricOrder\', \'linear\');\n',
    '  [p, e, t] = meshToPet(femmesh);\n',
)


class MatlabTrimeshGenerator:

    @staticmethod
    def from_verts(verts, hmax, dtype=tf.float32):
        """ Creates a mesh of the polygon defined by verts using the matlab engine

        Parameters
        ----------
        verts : 2d list
            list of vertices [[x1, y1], ..., [xn, yn]]

        hmax : float
            Maximum (approx.) edge size for generating the mesh

        Returns
        -------
        trimesh : TriangularMesh
            Returns the triangular mesh created from the polygon.
        """
        try:
            import os
            import matlab.engine
        except ImportError:
            msg = ''.join(("You must have a working copy of Matlab and installed the MATLAB Engine.\n",
                           "Follow the instructions at: \n\n",
                           "\thttps://mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html"))
            raise ImportError(msg)

        # start the matlab engine
        eng = matlab.engine.start_matlab()

        # create the file defining the matlab function to be used
        fname = 'meshfrompolyverts.m'
        path = os.path.join(os.getcwd(), fname)

        with open(path, 'w') as f:
            f.write(''.join(meshfrompolyverts))

        p, e, t = (np.asarray(item)
                   for item in eng.meshfrompolyverts(matlab.double(verts),
                                                     hmax,
                                                     nargout=3))
        eng.quit()

        # additional postprocessing on d to convert from matlab indexing to
        # python indexing
        t[:3, :] -= 1
        e[:2, :] -= 1

        t = np.asarray(t, dtype=np.int32)
        e = np.asarray(e, dtype=np.int32)

        return TriangleMesh(p.T, t[:3].T, e[:2].T)
