'''
Python module that contains the functions
about reading swc files.
'''
import re
import numpy as np


# Definition of swc data container
SWC_DCT = {'index': 0,
           'type': 1,
           'x': 2,
           'y': 3,
           'z': 4,
           'radius': 5,
           'parent': 6}


def read_swc(input_file, line_delimiter='\n'):
    '''Function to properly load a swc file
       that contains a list of sections,
       into a 'Data' format that contains all
       info extracting comments.
    '''
    # Read all data from file.
    with open(input_file, 'r') as f:

        read_data = f.read()

    # Split data per lines
    split_data = read_data.split(line_delimiter)

    # Clean data from comments and empty lines
    split_data = [a for a in split_data if '#' not in a]
    split_data = [a for a in split_data if a != '']

    return np.array(split_data)


def swc_to_data(data_swc):
    '''Transform swc to data to be used in make_tree.
    '''
    expected_data = re.compile(r'^\s*([-+]?\d*\.\d+|[-+]?\d+)'
                               r'\s*([-+]?\d*\.\d+|[-+]?\d+)\s'
                               r'*([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*$')

    data = []

    for dpoint in data_swc:

        if expected_data.match(dpoint.replace('\r', '')):

            segment_point = np.array(expected_data.match(dpoint.replace('\r', '')).groups(),
                                     dtype=np.float)

            # make the radius diameter
            segment_point[SWC_DCT['radius']] = 2. * segment_point[SWC_DCT['radius']]

            data.append(segment_point)

    return np.array(data)


def swc_data_to_lists(data):
    """
    Transforms data as loaded from read_swc
    into a set of 'meaningful' lists:

    x: list of floats
        x-coordinates

    y: list of floats
        y-coordinates

    z: list of floats
        z-coordinates

    d: list of floats
        diameters

    t: list of ints
        tree type

    p: list of ints
        parent id

    ch: dictionary
        children id(s)
    """
    length = len(data)

    # Here we define the expected structure of the data.
    # If this structure is not followed, the data will fail
    # to load and the method will be terminated, with an error message.

    expected_data = re.compile(r'^\s*([-+]?\d*\.\d+|[-+]?\d+)'
                               r'\s*([-+]?\d*\.\d+|[-+]?\d+)\s'
                               r'*([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*'
                               r'([-+]?\d*\.\d+|[-+]?\d+)\s*$')

    # Definition of swc data from SWC_DCT function

    x = np.zeros(length, dtype=float)
    y = np.zeros(length, dtype=float)
    z = np.zeros(length, dtype=float)
    d = np.zeros(length, dtype=float)
    t = np.zeros(length, dtype=int)
    p = np.zeros(length, dtype=int)
    ch = {}

    first_line_data = expected_data.match(data[0].replace('\r', ''))

    total_offset = int(first_line_data.groups()[0])

    for enline in range(length):

        segment_point = expected_data.match(data[enline].replace('\r', '')).groups()

        x[enline] = float(segment_point[SWC_DCT['x']])
        y[enline] = float(segment_point[SWC_DCT['y']])
        z[enline] = float(segment_point[SWC_DCT['z']])
        # swc contains radii, and here it is transformed into diameter.
        d[enline] = 2 * float(segment_point[SWC_DCT['radius']])
        t[enline] = int(segment_point[SWC_DCT['type']])
        if enline != 0:
            p[enline] = int(segment_point[SWC_DCT['parent']]) - total_offset
        else:
            p[enline] = int(segment_point[SWC_DCT['parent']])

        if int(segment_point[SWC_DCT['index']]) - enline != total_offset:
            raise Exception("Aborting process, with non-sequential ids error.\
                             Fix to proceed.")

    for enline in range(length):

        ch[enline] = list(np.where(p == enline)[0])

    return x, y, z, d, t, p, ch
