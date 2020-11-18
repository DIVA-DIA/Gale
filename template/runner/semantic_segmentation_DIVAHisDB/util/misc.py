import numpy as np


def make_colour_legend_image(img_name, colour_encoding):
    import matplotlib.pyplot as plt

    labels = sorted(colour_encoding.keys())
    colors = [tuple(np.array(colour_encoding[k])/255) for k in labels]
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", c) for c in colors]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(img_name, dpi=1000, bbox_inches=bbox)