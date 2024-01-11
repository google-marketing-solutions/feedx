# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for feedx package."""

import os

import setuptools

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_readme():
  try:
    readme = open(
        os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8"
    ).read()
  except OSError:
    readme = ""
  return readme


def _get_version():
  with open(os.path.join(CURRENT_DIR, "feedx", "__init__.py")) as fp:
    for line in fp:
      if line.startswith("__version__") and "=" in line:
        version = line[line.find("=") + 1 :].strip(" '\"\n")
        if version:
          return version
    raise ValueError("`__version__` not defined in `feedx/__init__.py`")


def _parse_requirements(path):
  with open(os.path.join(CURRENT_DIR, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith("#"))
    ]


VERSION = _get_version()
README = _get_readme()
INSTALL_REQUIREMENTS = _parse_requirements(
    os.path.join(CURRENT_DIR, "requirements.txt")
)
TEST_REQUIREMENTS = _parse_requirements(
    os.path.join(CURRENT_DIR, "requirements_tests.txt")
)

setuptools.setup(
    name="feedx",
    version=VERSION,
    python_requires=">=3.10",
    description="Package for running A/B experiments on shopping feeds.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Google gTech Ads Data Science",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    url="https://github.com/google-marketing-solutions/feedx",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.10",
    ],
)
