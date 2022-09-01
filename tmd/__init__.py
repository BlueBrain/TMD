"""TMD package.

A python package for the topological analysis of neurons.
"""
import pkg_resources

from tmd import utils
from tmd.io import io
from tmd.Neuron import Neuron
from tmd.Population import Population
from tmd.Soma import Soma
from tmd.Topology import analysis
from tmd.Topology import methods
from tmd.Topology import statistics
from tmd.Tree import Tree

__version__ = pkg_resources.get_distribution("TMD").version
