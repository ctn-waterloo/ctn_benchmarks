#!/usr/bin/env python
import imp
import io
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup


setup(
    name="ctn_benchmark",
    author="Terry Stewart",
    author_email="tcstewar@uwaterloo.ca",
    packages=find_packages(),
    scripts=[],
    license="See LICENSE",
)
