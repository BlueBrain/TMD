'''Test tmd.Tree'''
from nose import tools as nt
import numpy as np
from tmd.Tree import Tree
from tmd.io import io
from tmd.Tree import methods
import os

_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../test_data')

# Filenames for testing
sample_file = os.path.join(DATA_PATH, 'sample.swc')
sample_file1 = os.path.join(DATA_PATH, '/valid/sample.swc')

neu1 = io.load_neuron(sample_file)
neu2 = io.load_neuron(sample_file1)
tree0 = neu1.neurites[0]
tree1 = neu1.neurites[1]
tree2 = neu2.basal[0]

x1 = np.array([0.,  1.,  2.,  3., 4.])
y1 = np.array([0.,  2.,  3.,  4., 5.])
z1 = np.array([0.,  3.,  4.,  5., 6.])
d1 = np.array([2.,  4.,  6.,  8., 10.])
t1 = np.array([2,   2,   2,   2,  2])
p1 = np.array([-1,  0,   1,   2,  1])

tree = Tree.Tree(x=x1, y=y1, z=z1, d=d1, t=t1, p=p1)

def test_rd():
    nt.ok_(methods._rd([0,0], [0,1]) == 1.)
    nt.ok_(methods._rd([0,0,0], [0,0,1]) == 1.)
    nt.ok_(methods._rd([1,2,0], [0,2,1]) == np.sqrt(2.))

# def test_rd_w():
#    nt.ok_(methods._rd_w([0,0], [0,1], w=[0., 1.]) == 1.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 2.], normed=False) == 2.)
#    nt.ok_(methods._rd_w([0,0], [1,1], w=[0., 0.], normed=False) == 0.)
#    nt.ok_(methods._rd_w([1,2,0], [0,2,1], normed=False) == methods._rd([1,2,0], [0,2,1]))

def test_size():
    nt.ok_(tree0.size() == 31.)
    nt.ok_(tree1.size() == 21.)

def test_get_type():
    nt.ok_(tree0.get_type() == 2)
    nt.ok_(tree1.get_type() == 3)

def test_get_trunk():
    nt.ok_(tree0.get_trunk() == 0)
    nt.ok_(tree1.get_trunk() == 0)

def test_get_bounding_box():
    nt.ok_(np.allclose(tree0.get_bounding_box(), 
                       np.array([[-5.,  0., -5.],
                                 [ 5., 10.,  5.]])))
    nt.ok_(np.allclose(tree1.get_bounding_box(), 
                       np.array([[-5.,  0., 0.],
                                 [ 5., 10., 0.]])))

def test_get_segments():
    seg0 = tree0.get_segments()
    seg1 = tree1.get_segments()
    seg  = tree.get_segments()
    nt.ok_(len(seg0) == 30)
    nt.ok_(len(seg1) == 20)
    nt.ok_(np.allclose(seg, [np.array([[ 0.,  0.,  0.],
                                       [ 1.,  2.,  3.]]),
                             np.array([[ 1.,  2.,  3.],
                                       [ 2.,  3.,  4.]]),
                             np.array([[ 2.,  3.,  4.],
                                       [ 3.,  4.,  5.]]),
                             np.array([[ 1.,  2.,  3.],
                                       [ 4.,  5.,  6.]])]))

def test_get_segment_lengths():
    sl = tree.get_segment_lengths()
    nt.ok_(np.allclose(sl, np.array([3.74165739, 1.73205081, 1.73205081, 5.19615242])))

def test_get_segment_radial_dist():
    rds = tree.get_segment_radial_distances()
    nt.ok_(np.allclose(rds, np.array([3.74165739, 5.38516481, 7.07106781, 8.77496439])))

def test_get_point_radial_dist():
    prds = tree.get_point_radial_distances()
    nt.ok_(np.allclose(prds, np.array([ 0., 3.74165739, 5.38516481, 7.07106781, 8.77496439])))

# def test_get_point_weighted_radial_dist():
#    prds = tree.get_point_weighted_radial_distances(w=(1, 1, 1), normed=False)
#    nt.ok_(np.allclose(prds, np.array([ 0., 3.74165739, 5.38516481, 7.07106781, 8.77496439])))

def test_get_point_path_dist():
    pds = tree.get_point_path_distances()
    nt.ok_(np.allclose(pds, np.array([ 0., 3.74165739, 5.47370819, 7.205759, 8.93780981])))

def test_get_sections():
    secs = tree.get_sections()
    nt.ok_(np.allclose(secs, np.array([[0, 1, 1], [1, 3, 4]])))

def test_get_sections_2():
    secs = tree.get_sections_2()
    nt.ok_(np.allclose(secs, np.array([[0, 1, 1], [1, 3, 4]])))

def test_get_section_number():
    nt.ok_(tree.get_section_number() == 3)
    nt.ok_(tree0.get_section_number() == 5)
    nt.ok_(tree1.get_section_number() == 3)

def test_get_section_lengths():
    nt.ok_(np.allclose(tree2.get_section_lengths(), np.array([10.00, 4.00, 4.00])))

def test_get_section_radial_distances():
    nt.ok_(np.allclose(tree.get_section_radial_distances(), np.array([3.74165739, 7.07106781, 8.77496439])))
    nt.ok_(np.allclose(tree.get_section_radial_distances(initial=True), np.array([0. , 3.74165739, 3.74165739])))

def test_get_section_path_distances():
    nt.ok_(np.allclose(tree.get_section_path_distances(), np.array([3.74165739,  3.46410162, 12.12435565])))

def test_get_bif_term():
    nt.ok_(np.allclose(tree.get_bif_term(), np.array([ 1.,  2.,  1.,  0.,  0.])))

def test_get_bifurcations():
    nt.ok_(np.allclose(tree.get_bifurcations(), np.array([1])))

def test_get_terminations():
    nt.ok_(np.allclose(tree.get_terminations(), np.array([3, 4])))

def test_get_children():
    nt.ok_(np.allclose(tree.get_children(), np.array([1])))
    nt.ok_(np.allclose(tree.get_children(sec_id=1), np.array([2, 4])))
    nt.ok_(np.allclose(tree.get_children(sec_id=2), np.array([3])))
    nt.ok_(np.allclose(tree.get_children(sec_id=3), np.array([])))

def test_get_way_to_root():
    nt.ok_(np.allclose(tree.get_way_to_root(), np.array([-1])))

def test_get_way_to_section_end():
    nt.ok_(np.allclose(tree.get_way_to_section_end(), np.array([0, 1])))

def test_get_way_to_section_start():
    nt.ok_(np.allclose(tree.get_way_to_section_start(), np.array([0, -1])))
