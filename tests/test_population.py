"""Test tmd.population"""
from tmd.Population.Population import Population


def test_constructor(neuron):
    pop = Population(neurons=[neuron])
    assert len(pop.neurons) == 1


def test_population(population, neuron):
    assert len(population.neurons) == 5

    population.append_neuron(neuron)
    assert len(population.neurons) == 6
