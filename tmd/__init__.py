"""TMD package.

A python package for the topological analysis of neurons.
"""
import pkg_resources

from tmd import utils  # noqa
from tmd.io import io  # noqa
from tmd.Neuron import Neuron  # noqa
from tmd.Population import Population  # noqa
from tmd.Soma import Soma  # noqa
from tmd.Topology import analysis  # noqa
from tmd.Topology import methods  # noqa
from tmd.Topology import statistics  # noqa
from tmd.Tree import Tree  # noqa

__version__ = pkg_resources.get_distribution("TMD").version
