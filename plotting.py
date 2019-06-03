import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec


def plot(fig, shot, best, atom_num):
    pixelsize = 3.75e-3  # 3.75 um, reported in mm.
    # fig.add_subplot(111).imshow(shot.transmission)
    color_norm = colors.Normalize(-0.1, 1.0)

    # fig = plt.figure(1)
    wr = [0.9, 8, 1.1]
    hr = [1, 9]
    gs = gridspec.GridSpec(2, 3, width_ratios=wr, height_ratios=hr)
    # font = {"size": 16}
    # plt.rc("font", **font)

    # convert best-fit parameters to text
    title = "Shot " + str(0)
    A = str(np.round(best[0], 2))
    x_0 = str(np.round(pixelsize * best[1], 3))
    y_0 = str(np.round(pixelsize * best[2], 3))
    w_x = str(np.round(pixelsize * best[3], 3))
    w_y = str(np.round(pixelsize * best[4], 3))
    theta = str(np.round(best[5], 2))
    z_0 = str(np.round(best[6], 2))

    text1 = "A = " + A
    text2 = "x_0 = " + x_0
    text3 = "y_0 = " + y_0
    text4 = "sigma_x = " + w_x
    text5 = "sigma_y = " + w_y
    text6 = "theta = " + theta + " rad"
    text7 = "N = " + str(np.round(atom_num / 1000000.0, 2)) + " million"

    # # best-fit parameters: display
    ax5 = fig.add_subplot(gs[5])
    ax5.axis("off")
    ax5.text(0, 0.9, title, fontsize=24)
    ax5.text(0, 0.7, text1)
    ax5.text(0, 0.6, text2)
    ax5.text(0, 0.5, text3)
    ax5.text(0, 0.4, text4)
    ax5.text(0, 0.3, text5)
    ax5.text(0, 0.2, text7)

    # # horizontal and vertical 1D fits

    # ax1 = fig.add_subplot(gs[1])
    # ax1.plot(x_axis, 1 - horizontal, "ko", markersize=2)
    # ax1.plot(x_axis, 1 - fit_h, "r", linewidth=1)
    # ax1.xlim(0, width)
    # ax1.ylim(norm_min, norm_max)
    # ax1.gca().axes.get_xaxis().set_visible(False)

    # ax3 = fig.aax3subplot(gs[3])
    # ax3.plot(1 - vertical, y_axis, "ko", markersize=2)
    # ax3.plot(1 - fit_v, y_axis, "r", linewidth=1)
    # ax3.plot()
    # ax3.xlim(norm_max, norm_min)
    # ax3.ylim(0, height)
    # ax3.gca().axes.get_yaxis().set_visible(False)

    # # transmission plot with axis lines and zoom box
    ax4 = fig.add_subplot(gs[4])
    ax4.imshow(
        1 - shot.transmission, cmap="gray", norm=color_norm
    )  # , cmap=c, norm=norm, extent=pixels)
    # ax4.plot(pixelsize * x_hor, pixelsize * y_hor, color="g", linewidth=0.5)
    # ax4.plot(pixelsize * x_ver, pixelsize * y_ver, color="g", linewidth=0.5)

    # plt.xlim(pixels[0], pixels[1])
    # plt.ylim(pixels[3], pixels[2])  # y-axis is upside down!

    # # save best-fit parameters and image to files
    # save_path = ".."
    # now = time.strftime("%Y%m%d-%H%M%S")

    # pic_path = save_path + "/Analysis Results/" + now + ".png"
    # txt_path = save_path + "/Analysis Results/diary.txt"

    # print("Saving image and writing to diary")
    # diary = open(txt_path, "a+")
    # diary_text = (now, np.round(best, 2), np.round(int_error, 2))
    # diary.write("Time: %s. Fit: %s. Error: %s. \n" % diary_text)
    # diary.close()

    # plt.ion()
    # plt.pause(0.01)
    # plt.draw()
    # plt.savefig(pic_path, dpi=150)
