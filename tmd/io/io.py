"""Python module that contains the functions about reading and writing files."""

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

import os
import warnings
from pathlib import Path

import numpy as _np
from scipy import sparse as sp
from scipy.sparse import csgraph as cs

from tmd.io.conversion import convert_morphio_soma
from tmd.io.conversion import convert_morphio_trees
from tmd.io.h5 import read_h5
from tmd.io.swc import SWC_DCT
from tmd.io.swc import read_swc
from tmd.io.swc import swc_to_data
from tmd.Neuron import Neuron
from tmd.Population import Population
from tmd.Soma import Soma
from tmd.Tree import Tree
from tmd.utils import SOMA_TYPE
from tmd.utils import TREE_TYPE_DICT
from tmd.utils import TmdError


class LoadNeuronError(TmdError):
    """Captures the exception of failing to load a single neuron."""


def make_tree(data):
    """Make tree structure from loaded data."""
    tr_data = _np.transpose(data)

    parents = [
        _np.where(tr_data[0] == i)[0][0] if len(_np.where(tr_data[0] == i)[0]) > 0 else -1
        for i in tr_data[6]
    ]

    return Tree.Tree(
        x=tr_data[SWC_DCT["x"]],
        y=tr_data[SWC_DCT["y"]],
        z=tr_data[SWC_DCT["z"]],
        d=tr_data[SWC_DCT["radius"]],
        t=tr_data[SWC_DCT["type"]],
        p=parents,
    )


def redefine_types(user_types=None):
    """Return tree types depending on the customized types selected by the user.

    Args:
        user_types (dictionary or None):

    Returns:
        final_types (dict): tree types for the construction of Neuron.
    """
    final_tree_types = TREE_TYPE_DICT.copy()
    if user_types is not None:
        final_tree_types.update(user_types)
    return final_tree_types


def load_neuron(
    input_file, line_delimiter="\n", soma_type=None, user_tree_types=None, remove_duplicates=True
):
    """I/O method to load an swc or h5 file into a Neuron object."""
    tree_types = redefine_types(user_tree_types)

    # Definition of swc types from type_dict function
    if soma_type is None:
        soma_index = SOMA_TYPE
    else:
        soma_index = soma_type

    # Make neuron with correct filename and load data
    ext = os.path.splitext(input_file)[-1].lower()
    if ext == ".swc":
        data = swc_to_data(read_swc(input_file=input_file, line_delimiter=line_delimiter))
        neuron = Neuron.Neuron(name=str(input_file).replace(".swc", ""))

    elif ext == ".h5":
        data = read_h5(input_file=input_file, remove_duplicates=remove_duplicates)
        neuron = Neuron.Neuron(name=str(input_file).replace(".h5", ""))

    else:
        raise LoadNeuronError(
            f"{input_file} is not a valid h5 or swc file. If asc set use_morphio to True."
        )

    # Check for duplicated IDs
    IDs, counts = _np.unique(data[:, 0], return_counts=True)
    if (counts != 1).any():
        warnings.warn(f"The following IDs are duplicated: {IDs[counts > 1]}")

    data_T = _np.transpose(data)

    try:
        soma_ids = _np.where(data_T[1] == soma_index)[0]
    except IndexError as exc:
        raise LoadNeuronError("Soma points not in the expected format") from exc

    # Extract soma information from swc
    soma = Soma.Soma(
        x=data_T[SWC_DCT["x"]][soma_ids],
        y=data_T[SWC_DCT["y"]][soma_ids],
        z=data_T[SWC_DCT["z"]][soma_ids],
        d=data_T[SWC_DCT["radius"]][soma_ids],
    )

    # Save soma in Neuron
    neuron.set_soma(soma)
    p = _np.array(data_T[6], dtype=int) - _np.transpose(data)[0][0]
    # return p, soma_ids
    try:
        dA = sp.csr_matrix(
            (_np.ones(len(p) - len(soma_ids)), (range(len(soma_ids), len(p)), p[len(soma_ids) :])),
            shape=(len(p), len(p)),
        )
    except Exception as exc:
        raise LoadNeuronError("Cannot create connectivity, nodes not connected correctly.") from exc

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids) :, len(soma_ids) :])

    # Extract trees
    for i in range(comp[0]):
        tree = make_tree(data[_np.where(comp[1] == i)[0] + len(soma_ids)])
        neuron.append_tree(tree, tree_types)

    return neuron


def load_neuron_from_morphio(path_or_obj, user_tree_types=None):
    """Create Neuron object from morphio object or from path loaded via morphio.

    Supported file formats: h5, swc, asc.

    Args:
        path_or_obj (Union[str, morphio.Morphology]):
            Filepath or morphio object

    Returns:
        neuron (Neuron): tmd Neuron object
    """
    from morphio import Morphology  # pylint: disable=import-outside-toplevel

    tree_types = redefine_types(user_tree_types)

    if isinstance(path_or_obj, (str, Path)):
        obj = Morphology(path_or_obj)
        filename = path_or_obj
    else:
        obj = path_or_obj
        # MorphIO does not support naming of objects yet.
        filename = ""

    neuron = Neuron.Neuron()
    neuron.name = filename
    neuron.set_soma(convert_morphio_soma(obj.soma))
    for tree in convert_morphio_trees(obj):
        neuron.append_tree(tree, tree_types)

    return neuron


def load_population(neurons, user_tree_types=None, name=None, use_morphio=False):
    """Load all data of recognised format (swc, h5) into a Population object.

    Takes as input a directory or a list of files to load.
    """
    if isinstance(neurons, (list, tuple)):
        files = neurons
        name = name if name is not None else "Population"
    elif os.path.isdir(neurons):  # Assumes given input is a directory
        files = [os.path.join(neurons, neuron_dir) for neuron_dir in os.listdir(neurons)]
        name = name if name is not None else os.path.basename(neurons)
    elif os.path.isfile(neurons):  # Assumes given input is a file
        files = [neurons]
        name = name if name is not None else os.path.basename(neurons)
    else:
        raise TypeError(
            "The format of the given neurons is not supported. "
            "Expected an iterable of files, or a directory, or a single morphology file. "
            f"Got: {neurons}"
        )

    pop = Population.Population(name=name)

    for filename in files:
        try:
            ext = os.path.splitext(filename)[-1][1:].lower()
            if not use_morphio:
                assert ext in ("h5", "swc")
                pop.append_neuron(load_neuron(filename, user_tree_types=user_tree_types))
            else:
                assert ext in ("h5", "swc", "asc")
                pop.append_neuron(
                    load_neuron_from_morphio(filename, user_tree_types=user_tree_types)
                )

        except AssertionError as exc:
            raise Warning(
                "{filename} is not a valid h5, swc or asc file. If asc set use_morphio to True."
            ) from exc
        except LoadNeuronError:
            print(f"File failed to load: {filename}")

    return pop
