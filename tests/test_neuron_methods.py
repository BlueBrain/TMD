"""Test tmd.Neuron"""
import os

import numpy as np

from tmd.io import io
from tmd.Neuron import Neuron
from tmd.Neuron import methods

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, "data")

# Filenames for testing
sample_file = os.path.join(DATA_PATH, "sample.swc")
