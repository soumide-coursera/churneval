# -*- coding: utf-8 -*-
"""
@author: Soumi De
"""

import setuptools


with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='churneval',
    author='Soumi De',
    author_email='soumi.de@res.christuniversity.in',
    license='LICENSE.txt',
    description='churneval is a python package for evaluating churn models',
    version='1.1',
    python_requires=">=3.5",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['churneval'],
    install_requires=['churneval','pandas','sklearn','matplotlib'],
)