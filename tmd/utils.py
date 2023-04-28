"""Contains all the commonly used functions and data useful for multiple tmd modules."""

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

term_dict = {"x": 0, "y": 1, "z": 2}

TREE_TYPE_DICT = {1: "soma", 2: "axon", 3: "basal_dendrite", 4: "apical_dendrite"}

SOMA_TYPE = 1


class TmdError(Exception):
    """Specific exception raised by TMD."""
