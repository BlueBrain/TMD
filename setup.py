"""Setup for the TMD package."""
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "h5py>=2.9.0",
    "scipy>=1.4.0",
    "numpy>=1.18.0",
    "scikit-learn>=0.22.0",
    "munkres>=1.0.12",
    "cached-property>=1.5.1",
    "morphio>=3.3.4,<4",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-click",
    "docutils<0.21",
]

test_reqs = [
    "mock>=3",
    "pytest>=6",
    "pytest-cov>=3",
    "pytest-html>=2",
]

setup(
    name="TMD",
    author="Blue Brain Project, EPFL",
    description="A python package for the topological analysis of neurons.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://TMD.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/TMD/issues",
        "Source": "https://github.com/BlueBrain/TMD",
    },
    license="GNU Lesser General Public License v3.0",
    packages=find_namespace_packages(include=["tmd*"]),
    python_requires=">=3.8",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
        "viewer": ["matplotlib>=3.2.0"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
