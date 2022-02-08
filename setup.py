#! /usr/bin/env python
from setuptools import setup, find_packages

with open('./README.md') as f:
    long_description = f.read()

setup(name='CBLAST',
      version='0.10',
      author='Stephen Plaza',
      description='Tool for cluster neurons using connectivity',
      long_description=long_description,
      author_email='plazas@janelia.hhmi.org',
      url='https://github.com/janelia-flyem/cblast',
      packages=find_packages(),
      )

