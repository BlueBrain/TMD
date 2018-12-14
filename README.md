TMD: Topological Morphology Descriptor
========================================

The TMD performs topological analysis of neuronal morphologies and extracts the persistence of trees.

Details
---------
Author: Lida Kanari

Contributors: Pawel Dlotko, Benoit Coste

Publication: A Topological Representation of Branching Neuronal Morphologies

_Cite this article as:_
    Kanari, L., Dłotko, P., Scolamiero, M. et al. Neuroinform (2018) 16:3.
    DOI: <https://doi.org/10.1007/s12021-017-9341-1>

Sotware description
---------------------

This Python module includes: 

* Basic loading of neuronal morphologies in swc and h5 file format. 
* Extraction of the topological descriptors of tree morphologies.
* Visualization of neuronal trees and neurons.
* Ploting persistence diagrams, barcodes and images.

Supported OS
--------------

Ubuntu : 12.0, 14.04, 16.04

macOS: Sierra 10.13.3

Required Dependencies
---------------------

Python : 2.7+
numpy : 1.8.1+
scipy : 0.13.3+

Optional Dependencies
----------------------
h5py : 2.8.0+ (optional)
matplotlib : 1.3.1+ (required for viewer mode)

Instalation instructions
--------------------------------

```bash
virtualenv test_tmd
source ./test_tmd/bin/activate
git clone https://github.com/BlueBrain/TMD
pip install ./TMD
```

For installation of viewers (only works in Python2)

```bash
pip install ./TMD[viewer]
```


