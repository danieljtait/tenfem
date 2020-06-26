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


@lru_cache(maxsize=5)
def star(hmax: float, rot_deg: float = 30, dtype: tf.DType = tf.float32):
    """ Create a `TriangleMesh` of a star-shaped domain.

    Args:
        hmax: Average maximum edge ledge of triangles in the mesh.
        dtype: The datatype of the mesh nodes. Default `float32`

        Return:
        mesh: A `TriangleMesh` object of a star shaped domain.
    """
    import re
    from matplotlib.path import Path
    from matplotlib import transforms

    def svg_parse(path):
        """ svg_parse code from https://matplotlib.org/3.1.1/gallery/showcase/firefox.html """
        commands = {'M': (Path.MOVETO,),
                    'L': (Path.LINETO,),
                    'Q': (Path.CURVE3,) * 2,
                    'C': (Path.CURVE4,) * 3,
                    'Z': (Path.CLOSEPOLY,)}
        path_re = re.compile(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]+)', re.IGNORECASE)
        float_re = re.compile(r'(?:[\s,]*)([+-]?\d+(?:\.\d+)?)')
        vertices = []
        codes = []
        last = (0, 0)
        for cmd, values in path_re.findall(path):
            points = [float(v) for v in float_re.findall(values)]
            points = np.array(points).reshape((len(points) // 2, 2))
            if cmd.islower():
                points += last
            cmd = cmd.capitalize()
            last = points[-1]
            codes.extend(commands[cmd])
            vertices.extend(points.tolist())
        return codes, vertices

    star_svg = ["m 156.01339,146.73213",
                "c -1.54964,4.55501 -25.88435,1.13997 -29.81541,3.9142",
                "c -3.93106,2.77422 -8.86744,26.84646 -13.67839,26.78024",
                "c -4.81094,-0.0662 -9.08288,-24.26521 -12.936086,-27.14659",
                "c -3.85321,-2.88138 -28.272694,-0.13743 -29.696376,-4.73337",
                "c -1.423682,-4.59594 20.270822,-16.1367 21.820466,-20.69171",
                "c  1.549644,-4.55502 -8.60604,-26.9314 -4.674981,-29.705626",
                "c 3.931059,-2.774227 21.610937,14.292186 26.421877,14.358406",
                "c 4.81095,0.0662 22.95387,-16.50709 26.80708,-13.625714",
                "c 3.85321,2.881376 -6.91453,24.969754 -5.49084,29.565694",
                "c 1.42368,4.59594 22.79231,16.72945 21.24266,21.28447 z"]
    star_svg = ' '.join(star_svg)

    codes, verts = svg_parse(star_svg)
    verts = np.array(verts)
    path = Path(verts, codes)

    segments = path.to_polygons()[0]
    segments -= np.mean(segments, axis=0)
    segments /= np.max(np.std(segments, axis=0))

    transform = transforms.Affine2D()
    transform = transform.rotate(np.deg2rad(rot_deg))
    transform.transform(segments)

    segments = transform.transform(segments)

    mesh = MatlabTrimeshGenerator.from_verts(segments.tolist(), hmax, dtype=dtype)
    mesh.add_matplotlib_tri()

    return mesh
