# TMD

[![Run all tox jobs using Python3](https://github.com/BlueBrain/TMD/actions/workflows/run-tox.yml/badge.svg)](https://github.com/BlueBrain/TMD/actions/workflows/run-tox.yml)
[![license](https://img.shields.io/pypi/l/tmd.svg)](https://github.com/BlueBrain/TMD/blob/master/LICENSE-LGPL.txt)
[![codecov.io](https://codecov.io/github/BlueBrain/TMD/coverage.svg?branch=master)](https://codecov.io/github/BlueBrain/TMD?branch=master)
[![Documentation Status](https://readthedocs.org/projects/tmd/badge/?version=latest)](http://tmd.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10678146.svg)](https://doi.org/10.5281/zenodo.10678146)

A python package for the topological analysis of neurons.

The TMD performs the topological analysis of neuronal morphologies and extracts the persistence
barcodes of trees.

This Python module includes:

* Basic loading of neuronal morphologies in swc and h5 file format.
* Extraction of the topological descriptors of tree morphologies.
* Visualization of neuronal trees and neurons.
* Plotting persistence diagrams, barcodes and images.


## Installation

This package should be installed using pip:

```bash
pip install TMD
```

For installation of optional viewers:

```bash
pip install TMD[viewer]
```


## Usage

```python
# Import the TMD toolkit in IPython
import tmd

# Load a neuron
neuron = tmd.io.load_neuron('input_path_to_file/input_file.swc')

# Extract the tmd of a neurite, i.e., neuronal tree
pd = tmd.methods.get_persistence_diagram(neuron.neurites[0])
```

## Citation

If you use this software or method for your research, we kindly ask you to cite the following publication associated to this repository:

**A Topological Representation of Branching Neuronal Morphologies**

_Cite this article as:_

Kanari, L., Dłotko P., Scolamiero M., et al., A Topological Representation of Branching Neuronal Morphologies, Neuroinformatics 16, nᵒ 1 (2018): 3‑13. https://doi.org/10.1007/s12021-017-9341-1.


## Related publications

**Comprehensive Morpho-Electrotonic Analysis Shows 2 Distinct Classes of L2 and L3 Pyramidal Neurons in Human Temporal Cortex, Cerebral Cortex**

_Cite this article as:_

Deitcher Y., Eyal G., Kanari L., et al., Comprehensive Morpho-Electrotonic Analysis Shows 2 Distinct Classes of L2 and L3 Pyramidal Neurons in Human Temporal Cortex, Cerebral Cortex, Volume 27, Issue 11, November 2017, Pages 5398–5414, https://doi.org/10.1093/cercor/bhx226

**Objective Morphological Classification of Neocortical Pyramidal Cells**:

_Cite this article as:_

Lida Kanari, Srikanth Ramaswamy, Ying Shi, Sebastien Morand, Julie Meystre, Rodrigo Perin, Marwan Abdellah, Yun Wang, Kathryn Hess, Henry Markram, Objective Morphological Classification of Neocortical Pyramidal Cells, Cerebral Cortex, Volume 29, Issue 4, April 2019, Pages 1719-1735, https://doi.org/10.1093/cercor/bhy339

Developed in Blue Brain Project.

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2021-2022 Blue Brain Project/EPFL
