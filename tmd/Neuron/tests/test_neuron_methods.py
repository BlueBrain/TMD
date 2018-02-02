'''Test tmd.Neuron'''
from nose import tools as nt
import numpy as np
from tmd.Neuron import Neuron
from tmd.io import io
from tmd.Neuron import methods
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')

# Filenames for testing
sample_file = os.path.join(DATA_PATH, 'sample.swc')

neu1 = io.load_neuron(sample_file)

def test_size():
    nt.ok_(neu1.size() == 2)
    nt.ok_(neu1.size(neurite_type='axon') == 1)

def test_section_lengths():
    neu1.get_section_lengths()
    neu1.get_section_lengths(neurite_type='basal')

