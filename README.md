TMD: Topological Morphology Descriptor
========================================

The TMD performs the topological analysis of neuronal morphologies and extracts the persistence barcodes of trees.

©Blue Brain Project/EPFL 2005 – 2019. All rights reserved

Details
---------
Author: Lida Kanari

Contributors: Pawel Dlotko, Benoit Coste

Publication: 


A Topological Representation of Branching Neuronal Morphologies

_Cite this article as:_
    Kanari, L., Dłotko, P., Scolamiero, M. et al. Neuroinform (2018) 16:3.
    DOI: <https://doi.org/10.1007/s12021-017-9341-1>
    
Related publications:


Comprehensive Morpho-Electrotonic Analysis Shows 2 Distinct Classes of L2 and L3 Pyramidal Neurons in Human Temporal Cortex.

_Cite this article as:_
   Deitcher Y., Eyal G., Kanari L., et al. Cerebral Cortex (2017) 27:11
   DOI: <https://doi.org/10.1093/cercor/bhx226>
   
Objective Classification of Neocortical Pyramidal Cells 
    DOI: <http://dx.doi.org/10.1101/349977>
    
Developed in Blue Brain Project

Funding
---------
The Blue Brain Project receives Swiss government funding from ETH Board of the ETH Domain (ETH = Swiss Federal Institutes of Technology) and receives support as a research center by the École polytechnique fédérale de Lausanne (EPFL).

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

numpy : 1.8.1+,
scipy : 0.13.3+,
enum34 : 1.0.4+,
scikit-learn : 0.19.1+,
munkres: 1.0.12+

Optional Dependencies
----------------------
h5py : 2.8.0+ (optional),
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


