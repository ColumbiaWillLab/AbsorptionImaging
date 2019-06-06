import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from config import config


def plot(fig, shot):
    fig.clf()
    norm = (-0.1, 1.0)
    color_norm = colors.Normalize(*norm)

    x, y = shot.meshgrid
    params = shot.twoD_gaussian.best_values
    h, v = shot.best_fit_lines
    hfit, vfit = shot.oneD_gaussians

    ratio = [1, 9]
    gs = gridspec.GridSpec(2, 2, width_ratios=ratio, height_ratios=ratio)

    image = fig.add_subplot(gs[1, 1])
    image.imshow(shot.absorption, cmap="cividis", norm=color_norm)

    image.contour(
        shot.twoD_gaussian.eval(x=x, y=y).reshape(shot.shape),
        levels=shot.contour_levels,
        cmap="magma",
        linewidths=1,
        norm=color_norm,
    )

    image.axhline(params["y0"], linewidth=0.3)
    image.axvline(params["x0"], linewidth=0.3)

    h_x = np.arange(h.shape[0])
    h_y = hfit.eval(x=h_x)
    v_y = np.arange(v.shape[0])
    v_x = vfit.eval(x=v_y)

    hor = fig.add_subplot(gs[0, 1])
    hor.plot(h_x, h, "ko", markersize=0.2)
    hor.plot(h_x, h_y, "r", linewidth=0.5)
    hor.set_ylim(*norm)
    hor.get_xaxis().set_visible(False)

    ver = fig.add_subplot(gs[1, 0])
    ver.plot(v, v_y, "ko", markersize=0.2)
    ver.plot(v_x, v_y, "r", linewidth=0.5)
    ver.set_xlim(*norm)
    ver.invert_xaxis()
    ver.get_yaxis().set_visible(False)

