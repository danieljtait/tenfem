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
""" Install tenfem """
import os, sys

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name='tenfem',
    version='0.0.1',
    author='Daniel J. Tait',
    author_email='tait.djk@gmail.com',
    description=('Tensorflow implementation of the Finite Element Method', ),
    long_description=read('README.md'),
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=['tensorflow>=2.1', 'tensorflow_probability'],
    keywords='probability bayesian finite-element-method machine learning '
)
