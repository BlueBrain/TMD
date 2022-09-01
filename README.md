# TMD

A python package for the topological analysis of neurons.

The TMD performs the topological analysis of neuronal morphologies and extracts the persistence
barcodes of trees.

This Python module includes:

* Basic loading of neuronal morphologies in swc and h5 file format.
* Extraction of the topological descriptors of tree morphologies.
* Visualization of neuronal trees and neurons.
* Plotting persistence diagrams, barcodes and images.

Publication:

A Topological Representation of Branching Neuronal Morphologies

_Cite this article as:_
    Kanari, L., Dłotko, P., Scolamiero, M. et al. Neuroinform (2018) 16:3.
    DOI: <https://doi.org/10.1007/s12021-017-9341-1>

Related publications:

Comprehensive Morpho-Electrotonic Analysis Shows 2 Distinct Classes of L2 and L3 Pyramidal Neurons
in Human Temporal Cortex.

_Cite this article as:_
   Deitcher Y., Eyal G., Kanari L., et al. Cerebral Cortex (2017) 27:11
   DOI: <https://doi.org/10.1093/cercor/bhx226>

Objective Classification of Neocortical Pyramidal Cells
    DOI: <http://dx.doi.org/10.1101/349977>

Developed in Blue Brain Project


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


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2021-2022 Blue Brain Project/EPFL
