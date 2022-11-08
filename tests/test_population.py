"""Test tmd.population."""
from tmd.Population.Population import Population


def test_constructor(neuron):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    pop = Population(neurons=[neuron])
    assert len(pop.neurons) == 1


def test_population(population, neuron):
    # noqa: D103 ; pylint: disable=missing-function-docstring
    assert len(population.neurons) == 5

    population.append_neuron(neuron)
    assert len(population.neurons) == 6

    assert len(population.get_by_name("Neuron")) == 1

    population.append_neuron(neuron)
    assert len(population.neurons) == 7
    assert len(population.get_by_name("Neuron")) == 2
