import tmd
from view import polar_plots

filename = './'

pop = tmd.io.load_population(filename)

# Get the data for a single cell
res = polar_plots.get_histogram_polar_coordinates(pop.neurons[0], neurite_type='basal', N=30)

# Plot the data for a single cell
polar_plots.plot_polar_coordinates(res)

# Extract the polar plots of all neurons in the population and save images
# in a selected directory
for n in pop.neurons:
    res = polar_plots.get_histogram_polar_coordinates(n, neurite_type='basal', N=30)
    polar_plots.plot_polar_coordinates(res, output_path='./PolarPlots/', output_name=n.name.split('/')[-1])
    plt.close()
