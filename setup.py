""" Distribution configuration for TMD
"""
from setuptools import setup
from setuptools import find_packages
import imp


config = {
    'description': 'TMD: a python package for the topological analysis of neurons',
    'author': 'Lida Kanari',
    'url': 'https://github.com/BlueBrain/TMD',
    'author_email': 'lida.kanari@epfl.ch',
    'install_requires': [
        'h5py>=2.8.0',
        'scipy>=0.13.3',
        'numpy>=1.8.0',
        'scikit-learn>=0.19.1',
        'munkres>=1.0.12',
        'cached-property>=1.5.1',
        'morphio>=2.7.1'
    ],
    'setup_requires':['setuptools_scm'],
    'extras_require': {
                       'viewer': ['matplotlib>=3.2.0',],
                       },
    'packages': find_packages(),
    'license': 'LGPL',
    'scripts': [],
    'name': 'tmd',
    'include_package_data': True,
    'use_scm_version': True,
    'python_requires': '>=3.6',
}

setup(**config)
