'''Test tmd.utils'''
from nose import tools as nt
from tmd import utils

def test_term_dict():
    nt.ok_(utils.term_dict == {'x': 0,
                               'y': 1,
                               'z': 2})

def test_tree_type():
    nt.assert_dict_equal(utils.tree_type,
                         {1: 'soma',
                          2: 'axon',
                          3: 'basal',
                          4: 'apical'})
