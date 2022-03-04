'''Test tmd.Neuron'''
import numpy as np
from tmd.Neuron import Neuron
from tmd.io import io
from tmd.Neuron import methods
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, 'data')

# Filenames for testing
sample_file = os.path.join(DATA_PATH, 'sample.swc')
