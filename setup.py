""" Distribution configuration for TMD
"""
from setuptools import setup
from setuptools import find_packages

exec(open('tmd/version.py').read())

config = {
    'description': 'TMD: a python package for the topological analysis of neurons',
    'author': 'Lida Kanari',
    'url': 'https://github.com/BlueBrain/TMD',
    'author_email': 'lida.kanari@epfl.ch',
    'install_requires': [
        'h5py>=2.8.0',
        'enum34>=1.0.4',
        'scipy>=0.13.3',
        'numpy>=1.8.0'
    ],
    'extras_require': {
                       'viewer': ['matplotlib>=1.3.1',],
                       },
    'packages': find_packages(),
    'license': 'LGPL',
    'scripts': [],
    'name': 'tmd',
    'include_package_data': True,
    'version': VERSION,  # pylint: disable=undefined-variable
}

setup(**config)
