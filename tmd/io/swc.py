"""Python module that contains the functions about reading swc files."""

# Copyright (C) 2022  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re

import numpy as np

from tmd.utils import TmdError

# Definition of swc data container
SWC_DCT = {"index": 0, "type": 1, "x": 2, "y": 3, "z": 4, "radius": 5, "parent": 6}


def read_swc(input_file, line_delimiter="\n"):
    """Load a swc file containing a list of sections, into a 'Data' format."""
    # Read all data from file.
    with open(input_file, "r", encoding="utf-8") as f:
        read_data = f.read()

    # Split data per lines
    split_data = read_data.split(line_delimiter)

    # Clean data from comments and empty lines
    split_data = [a for a in split_data if "#" not in a]
    split_data = [a for a in split_data if a != ""]

    return np.array(split_data)


def swc_to_data(data_swc):
    """Transform swc to data to be used in make_tree."""
    expected_data = re.compile(
        r"^\s*([-+]?\d*\.\d+|[-+]?\d+)"
        r"\s*([-+]?\d*\.\d+|[-+]?\d+)\s"
        r"*([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*$"
    )

    data = []

    for dpoint in data_swc:
        if expected_data.match(dpoint.replace("\r", "")):
            segment_point = np.array(
                expected_data.match(dpoint.replace("\r", "")).groups(), dtype=float
            )

            # make the radius diameter
            segment_point[SWC_DCT["radius"]] = 2.0 * segment_point[SWC_DCT["radius"]]

            data.append(segment_point)

    return np.array(data)


def swc_data_to_lists(data):
    """Transforms data as loaded from read_swc into a set of 'meaningful' lists.

    The lists are the following:

    * x: list of floats
        x-coordinates
    * y: list of floats
        y-coordinates
    * z: list of floats
        z-coordinates
    * d: list of floats
        diameters
    * t: list of ints
        tree type
    * p: list of ints
        parent id
    * ch: dictionary
        children id(s)
    """
    length = len(data)

    # Here we define the expected structure of the data.
    # If this structure is not followed, the data will fail
    # to load and the method will be terminated, with an error message.

    expected_data = re.compile(
        r"^\s*([-+]?\d*\.\d+|[-+]?\d+)"
        r"\s*([-+]?\d*\.\d+|[-+]?\d+)\s"
        r"*([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*"
        r"([-+]?\d*\.\d+|[-+]?\d+)\s*$"
    )

    # Definition of swc data from SWC_DCT function

    x = np.zeros(length, dtype=float)
    y = np.zeros(length, dtype=float)
    z = np.zeros(length, dtype=float)
    d = np.zeros(length, dtype=float)
    t = np.zeros(length, dtype=int)
    p = np.zeros(length, dtype=int)
    ch = {}

    first_line_data = expected_data.match(data[0].replace("\r", ""))

    total_offset = int(first_line_data.groups()[0])

    for enline in range(length):
        segment_point = expected_data.match(data[enline].replace("\r", "")).groups()

        x[enline] = float(segment_point[SWC_DCT["x"]])
        y[enline] = float(segment_point[SWC_DCT["y"]])
        z[enline] = float(segment_point[SWC_DCT["z"]])
        # swc contains radii, and here it is transformed into diameter.
        d[enline] = 2 * float(segment_point[SWC_DCT["radius"]])
        t[enline] = int(segment_point[SWC_DCT["type"]])
        if enline != 0:
            p[enline] = int(segment_point[SWC_DCT["parent"]]) - total_offset
        else:
            p[enline] = int(segment_point[SWC_DCT["parent"]])

        if int(segment_point[SWC_DCT["index"]]) - enline != total_offset:
            raise TmdError(
                "Aborting process, with non-sequential ids error.\
                             Fix to proceed."
            )

    for enline in range(length):
        ch[enline] = list(np.where(p == enline)[0])

    return x, y, z, d, t, p, ch
