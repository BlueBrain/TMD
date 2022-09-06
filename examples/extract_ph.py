"""Example to extract the persistence diagram from a neuronal tree."""

# Step 1: Import the tmd module
import tmd
from tmd.view import plot
from tmd.view import view

# Step 2: Load your morphology
filename = "../tests/data/valid/C010398B-P2.CNG.swc"
neu = tmd.io.load_neuron(filename)

# Step 3: Extract the ph diagram of a tree
tree = neu.neurites[0]
ph = tmd.methods.get_persistence_diagram(tree)

# Step 4: Extract the ph diagram of a neuron's trees
ph_neu = tmd.methods.get_ph_neuron(neu)

# Step 5: Extract the ph diagram of a neuron's trees,
# depending on the neurite_type
ph_apical = tmd.methods.get_ph_neuron(neu, neurite_type="apical_dendrite")
ph_axon = tmd.methods.get_ph_neuron(neu, neurite_type="axon")
ph_basal = tmd.methods.get_ph_neuron(neu, neurite_type="basal_dendrite")

# Step 6: Plot the extracted topological data with three different ways

# Visualize the neuron
view.neuron(neu)

# Visualize a selected neurite type or multiple of them
view.neuron(neu, neurite_type=["apical_dendrite"])

# Visualize the persistence diagram
plot.diagram(ph_apical)

# Visualize the persistence barcode
plot.barcode(ph_apical)

# Visualize the persistence image
plot.persistence_image(ph_apical)
