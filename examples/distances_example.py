import tmd
import view

pop1 = tmd.io.load_population(directory1)
pop2 = tmd.io.load_population(directory2)
phs1 = [tmd.methods.get_ph_neuron(n, neurite_type='axon') for n in pop1.neurons]
phs2 = [tmd.methods.get_ph_neuron(n, neurite_type='axon') for n in pop2.neurons]

# Normalize the limits
xlims, ylims = define_limits(phs1 + phs2)
# Create average images for populations
imgs1 = [persistence_image(p, xlims=xlims, ylims=ylims) for p in phs1]
IMG1 = average_ph_image(imgs1)
imgs2 = [persistence_image(p, xlims=xlims, ylims=ylims) for p in phs2]
IMG2 = average_ph_image(imgs2)

# You can plot the images if you want to create pretty figures
average_figure1 = plot_imgs(IMG1, title='', xlims=xlims, ylims=ylims, cmap=cm.jet)
average_figure2 = plot_imgs(IMG2, title='', xlims=xlims, ylims=ylims, cmap=cm.jet)

# Create the diffence between the two images
DIMG = img_diff(IMG1, IMG2) # subtracts IMG2 from IMG1 so anything red IMG1 has more of it and anything blue IMG2 has more of it - or that's how it is supposed to be :)
# Plot the difference between them
diff_image = plot_imgs(DIMG, vmin=-1.0, vmax=1.0) # vmin, vmax important to see changes
# Quantify the absolute distance between the two averages
dist = np.sum(np.abs(DIMG))
