"""TMD class : Population."""

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

import warnings

from tmd.Neuron import Neuron


class Population:
    """A Population object is a container for Neurons.

    Args:
        name (str): The name of the Population.
        neurons (list[tmd.Neuron.Neuron.Neuron]): A list of neurons to include in the Population.
    """

    def __init__(self, name="Pop", neurons=None):
        """Creates an empty Population object."""
        self.neurons = []
        self.name = name

        if neurons:
            for neuron in neurons:
                self.append_neuron(neuron)

    @property
    def axon(self):
        """Get axon."""
        return [a for n in self.neurons for a in n.axon]

    @property
    def apical(self):
        """Get apical dendrite."""
        warnings.warn(
            "The 'apical' property is deprecated, please use 'apical_dendrite' instead",
            DeprecationWarning,
        )
        return self.apical_dendrite

    @property
    def apical_dendrite(self):
        """Get apical dendrite."""
        return [a for n in self.neurons for a in n.apical_dendrite]

    @property
    def basal(self):
        """Get basal dendrite."""
        warnings.warn(
            "The 'basal' property is deprecated, please use 'basal_dendrite' instead",
            DeprecationWarning,
        )
        return self.basal_dendrite

    @property
    def basal_dendrite(self):
        """Get basal dendrite."""
        return [a for n in self.neurons for a in n.basal_dendrite]

    @property
    def undefined(self):
        """I dont know."""
        return [a for n in self.neurons for a in n.undefined]

    @property
    def neurites(self):
        """Get neurites."""
        return self.apical_dendrite + self.axon + self.basal_dendrite + self.undefined

    @property
    def dendrites(self):
        """Get dendrites."""
        return self.apical_dendrite + self.basal_dendrite

    def append_neuron(self, new_neuron):
        """Append a Neuron object to the Population."""
        if isinstance(new_neuron, Neuron.Neuron):
            self.neurons.append(new_neuron)

    def get_by_name(self, name):
        """Get the neurons whose name is equal to the one given."""
        return [n for n in self.neurons if n.name == name]
