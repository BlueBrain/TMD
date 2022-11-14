"""Example for distance computation."""
import numpy as np

import tmd

# from matplotlib import cm

# import tmd.view as view


def compute_distances(directory1, directory2):
    """Compute distances."""
    pop1 = tmd.io.load_population(directory1)
    pop2 = tmd.io.load_population(directory2)

    phs1 = [tmd.methods.get_ph_neuron(n, neurite_type="basal_dendrite") for n in pop1.neurons]
    phs2 = [tmd.methods.get_ph_neuron(n, neurite_type="basal_dendrite") for n in pop2.neurons]

    # Normalize the limits
    xlim, ylim = tmd.analysis.get_limits(phs1 + phs2)

    # Create average images for populations
    # imgs1 = [tmd.analysis.get_persistence_image_data(p, xlim=xlim, ylim=ylim) for p in phs1]
    IMG1 = tmd.analysis.get_average_persistence_image(phs1, xlim=xlim, ylim=ylim)
    # imgs2 = [tmd.analysis.get_persistence_image_data(p, xlim=xlim, ylim=ylim) for p in phs2]
    IMG2 = tmd.analysis.get_average_persistence_image(phs2, xlim=xlim, ylim=ylim)

    # You can plot the images if you want to create pretty figures
    # average_figure1 = view.common.plot_img_basic(
    #     IMG1, title="", xlim=xlim, ylim=ylim, cmap=cm.jet
    # )
    # average_figure2 = view.common.plot_img_basic(
    #     IMG2, title="", xlim=xlim, ylim=ylim, cmap=cm.jet
    # )

    # Create the difference between the two images
    # Subtracts IMG2 from IMG1 so anything red IMG1 has more of it and anything blue IMG2 has more
    # of it - or that's how it is supposed to be :)
    DIMG = tmd.analysis.get_image_diff_data(IMG1, IMG2)

    # Plot the difference between them
    # diff_image = view.common.plot_img_basic(
    #     DIMG, vmin=-1.0, vmax=1.0
    # )  # vmin, vmax important to see changes

    # Quantify the absolute distance between the two averages
    dist = np.sum(np.abs(DIMG))

    return dist
