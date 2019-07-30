#! /usr/bin/env python
from setuptools import setup, find_packages

with open('./README.md') as f:
    long_description = f.read()

setup(name='typecluster',
      version='0.1',
      author='Stephen Plaza',
      description='Tool for cluster neurons',
      long_description=long_description,
      author_email='plazas@janelia.hhmi.org',
      url='https://github.com/janelia-flyem/typecluster',
      packages=find_packages(),
      )

