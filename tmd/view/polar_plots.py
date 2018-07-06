import numpy as np
from matplotlib import pylab as plt

def get_histogram_polar_coordinates(neuron, neurite_type='basal', N=25):
    '''
    '''
    def seg_angle(seg):
        mean_x = np.mean([seg[0][0], seg[1][0]])
        mean_y = np.mean([seg[0][1], seg[1][1]])

        return np.arctan2(mean_y, mean_x)

    segs = []
    for tr in getattr(neuron, neurite_type):
     segs = segs + tr.get_segments()

    angles = np.array([seg_angle(s) for s in segs])
    lens = []

    for tr in getattr(neuron, neurite_type):
        lens = lens + tr.get_segment_lengths().tolist()

    angles = np.array(angles)
    lens = np.array(lens)

    step = 2*np.pi/N
    ranges = [[i*step -np.pi, (i+1)*step-np.pi] for i in xrange(N)]

    results = []

    for r in ranges:
        results.append(r + [np.sum(lens[np.where((angles > r[0]) & (angles < r[1]))[0]])])

    return results


def plot_polar_coordinates(input_data, output_name=None, output_path=None, output_format='png'):
    '''
    '''
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    theta = np.array(input_data)[:,0]
    radii = np.array(input_data)[:,2] / np.max(input_data)
    width = 2*np.pi/len(input_data)
    bars = ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.8)

    if output_path is not None:
        plt.savefig(output_path + '/Polar_' + output_name, format=output_format)
