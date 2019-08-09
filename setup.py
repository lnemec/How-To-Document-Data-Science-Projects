# -*- coding: utf-8 -*-
# This file is part of How-To-Document-Data-Science-Projects.
#
#    How-To-Document-Data-Science-Projects is distributed in the
#    hope that it will be useful, but WITHOUT ANY WARRANTY;
#    without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE.
"""Setting up the How-To-Document-Data-Science-Projects module.

See: README.rst
"""

import sys

from setuptools import setup, find_packages
from example_mnist import __version__, __author__
from os import path

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')

here = path.abspath(path.dirname(__file__))

# Get long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='How-To-Document-Data-Science-Projects',
      version=__version__,
      author=__author__,
      author_email='Lydia.Nemec@gmail.com',
      maintainer='Dr. Lydia Nemec <Lydia.Nemec@gmail.com>',
      description='Example of how-to document a Data-Science project using Microsoft Azure Services.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='',
      url='https://github.com/lnemec/How-To-Document-Data-Science-Projects/edit/master',
      packages=find_packages(exclude=['doc']),
      requires=['tensorflow', 'numpy']
      )
