"""Test tmd.utils."""
from tmd import utils


def test_term_dict():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert utils.term_dict == {"x": 0, "y": 1, "z": 2}


def test_tree_type():
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert utils.TREE_TYPE_DICT == {1: "soma", 2: "axon", 3: "basal_dendrite", 4: "apical_dendrite"}
