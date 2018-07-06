""" Distribution configuration for TMD
"""
from setuptools import setup
from setuptools import find_packages

exec(open('tmd/version.py').read())

config = {
    'description': 'TMD: a python package for the topological analysis of neurons',
    'author': 'Lida Kanari',
    'url': 'https://bbpcode.epfl.ch/molecularsystems/TMD',
    'author_email': 'lida.kanari@epfl.ch',
    'install_requires': [
        'matplotlib>=1.3.1',
        'h5py>=2.8.0',
        'enum34>=1.0.4',
        'scipy>=0.13.3',
        'numpy>=1.8.0'
    ],
    'extras_require': {},
    'packages': find_packages(),
    'license': 'BSD',
    'scripts': [],
    'name': 'tmd',
    'include_package_data': True,
    'version': VERSION,  # pylint: disable=undefined-variable
}

setup(**config)
