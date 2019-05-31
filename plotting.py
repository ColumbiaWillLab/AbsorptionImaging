import matplotlib.gridspec as gridspec


def plot(fig, shot):
    fig.add_subplot(111).imshow(shot.transmission)
    # norm = plt.Normalize(norm_min, norm_max)

    # fig = plt.figure(1)
    # wr = [0.9, 8, 1.1]
    # hr = [1, 9]
    # gs = gridspec.GridSpec(2, 3, width_ratios=wr, height_ratios=hr)
    # font = {"size": 16}
    # plt.rc("font", **font)

    # # convert best-fit parameters to text
    # title = "Shot " + str(shotnum)
    # A = str(np.round(best[0], 2))
    # x_0 = str(np.round(pixelsize * best[1], 3))
    # y_0 = str(np.round(pixelsize * best[2], 3))
    # w_x = str(np.round(pixelsize * best[3], 3))
    # w_y = str(np.round(pixelsize * best[4], 3))
    # theta = str(np.round(best[5], 2))
    # z_0 = str(np.round(best[6], 2))

    # text1 = "A = " + A
    # text2 = "x_0 = " + x_0
    # text3 = "y_0 = " + y_0
    # text4 = "sigma_x = " + w_x
    # text5 = "sigma_y = " + w_y
    # # text6 = 'theta = '+ theta + ' rad'
    # text7 = "N = " + str(np.round(atom_num / 1000000.0, 2)) + " million"

    # # best-fit parameters: display
    # ax5 = plt.subplot(gs[5])
    # plt.axis("off")
    # plt.text(0, 0.9, title, fontsize=24)
    # plt.text(0, 0.7, text1)
    # plt.text(0, 0.6, text2)
    # plt.text(0, 0.5, text3)
    # plt.text(0, 0.4, text4)
    # plt.text(0, 0.3, text5)
    # plt.text(0, 0.2, text7)

    # # horizontal and vertical 1D fits

    # ax1 = plt.subplot(gs[1])
    # plt.plot(x_axis, 1 - horizontal, "ko", markersize=2)
    # plt.plot(x_axis, 1 - fit_h, "r", linewidth=1)
    # plt.xlim(0, width)
    # plt.ylim(norm_min, norm_max)
    # plt.gca().axes.get_xaxis().set_visible(False)

    # ax3 = plt.subplot(gs[3])
    # plt.plot(1 - vertical, y_axis, "ko", markersize=2)
    # plt.plot(1 - fit_v, y_axis, "r", linewidth=1)
    # plt.plot()
    # plt.xlim(norm_max, norm_min)
    # plt.ylim(0, height)
    # plt.gca().axes.get_yaxis().set_visible(False)

    # # transmission plot with axis lines and zoom box
    # ax4 = plt.subplot(gs[4])
    # plt.imshow(1 - transmission, cmap=c, norm=norm, extent=pixels)
    # plt.plot(pixelsize * x_hor, pixelsize * y_hor, color="g", linewidth=0.5)
    # plt.plot(pixelsize * x_ver, pixelsize * y_ver, color="g", linewidth=0.5)

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
