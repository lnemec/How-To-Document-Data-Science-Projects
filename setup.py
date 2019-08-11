# -*- coding: utf-8 -*-
# This file is part of How-To-Document-Data-Science-Projects.
#
#    How-To-Document-Data-Science-Projects is distributed in the
#    hope that it will be useful, but WITHOUT ANY WARRANTY;
#    without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE.
"""
Setting up the How-To-Document-Data-Science-Projects module.

Example of how-to document a Data-Science project using
Microsoft Azure Services.

See: README.rst
"""

import sys
from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc
from example_mnist import __version__, __author__
from os import path

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')

project_name='How-To-Document-Data-Science-Projects'
here = path.abspath(path.dirname(__file__))

# Get long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

cmdclass = {'build_sphinx': BuildDoc}

setup(name=project_name,
      version=__version__,
      author=__author__,
      maintainer= __author__,
      description='Example of how-to document a Data-Science project using Microsoft Azure Services.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache License Version 2.0',
      url='https://github.com/lnemec/How-To-Document-Data-Science-Projects/edit/master',
      packages=find_packages(exclude=['doc']),
      package_data = { '': ['*.txt', '*.rst'] },
      requires=['tensorflow', 'numpy'],
      cmdclass=cmdclass,
      command_options={'build_sphinx': {
            'project': ('setup.py', project_name),
            'version': ('setup.py', __version__),
            'source_dir': ('setup.py', 'doc/source'),
            'build_dir': ('setup.py', 'doc/build')} }
      )
