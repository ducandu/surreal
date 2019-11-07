# Copyright 2019 ducandu GmbH, All Rights Reserved
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
# ==============================================================================

import os

from setuptools import setup, find_packages

# Read __version__ avoiding imports that might be in install_requires.
version_vars = {}
with open(os.path.join(os.path.dirname(__file__), 'surreal', 'version.py')) as fp:
    exec(fp.read(), version_vars)

install_requires = [
    'numpy',
    'opencv-python',
    'packaging',
    'pygame',
    'pyyaml',
    'pytest',
    'requests',
    'scipy',
    'tensorflow',
    'tensorflow_probability'
]

setup_requires = []

extras_require = {
    'atari': ['swig', 'box2d'],
    'tf-gpu': ['tensorflow-gpu', 'tensorflow_probability'],
    # To use openAI Gym Envs (e.g. Atari).
    # for Win atari: pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    'gym': ['gym', 'atari-py'],
}

setup(
    name='surreal',
    version=version_vars['__version__'],
    description='A Library for Rapid RL-Algo Development, Testing, and Deployment',
    long_description="""
Surreal is a library allowing rapid (within one(!) day) development and implementation of published reinforcement
learning algos. It comes with many state-of-the-art benchmark algos that can be run out of the box against
popular envs (e.g. openAI gym) as well as UnrealEngine4 games.
The main development focus is on environment/game controlled execution (e.g. from within UE4), 
mixing different algos, multi-agent learning (adversarial and cooperative), distributed execution (cloud),
model-based RL, hierarchical RL, and intrinsic motivation.
""",
    url='https://deepgames.ai',
    author='ducandu GmbH',
    author_email='sven.mika@ducandu.com',
    license='Apache 2.0',
    packages=[package for package in find_packages() if package.startswith('surreal')],
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False
)
