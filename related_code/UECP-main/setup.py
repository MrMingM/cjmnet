# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from os.path import dirname, realpath
from setuptools import setup, find_packages
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='UECP',
    version=__version__,
    packages=find_packages(),
    license='Academic Software License',
    author='Kang Yang, Tianci Bu, Peng Wang, Deying Li, Wen Jie, Yongcai Wang',
    description='Official UECP implementation for uncertainty-enhanced collaborative perception',
    long_description=open("README.md").read(),
    install_requires=[],
)
