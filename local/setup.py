# Copyright 2020 Unibg Seclab (https://seclab.unibg.it)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup


setup(
    name="mondrian",
    version="0.1.0",
    description="Single-threaded mondrian",
    install_requires=[
        "numpy",
        "pandas==1.1.3",
        "scipy",
        "treelib==1.6.1",
        "scikit-learn",
        "minisom"
    ],
    url="https://github.com/unibg-seclab/spark-mondrian",
    author="UniBG Seclab",
    author_email="seclab@unibg.it",
    license="Apache",
    packages=[
        "mondrian",
    ],
    keywords="anonymization k-anonymity mondrian",
)
