""" Shot Image Analysis Instructions
STEP 0. Make sure that mode == 'automatic' down below.
STEP 1. Run the script using Imaging Code>python Imaging.py.
A large white display labeled Figure 1 should pop up on the left monitor.
STEP 2. Run a single iteration in Cicero; the display should turn black.
After a few seconds the window will read "(Not Responding)" when clicked.
STEP 3. When the window is not responding, drag it to the right monitor
and keep it there fore the duration of data-taking.
STEP 4. Run as many repeated iterations in Cicero as your heart desires.
The display will show the background-subtracted absorption image from the
latest shot, and will update live with each new shot. Do not set the MOT
loading time below 1s, as the program is not fast enough to keep up.
STEP 5. Occasionally the 1D Gaussian fit will throw an error and crash the
program. Start over at step 1. (I am working on fixing this!)
STEP 6. After the last shot, hit the red X on the top right of the display.
It will say that python.exe is not responding; close the program.
Raw data and each live-updated image are saved in their respective folders.
"""
import time
import os
import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import Helper as hp
import shots
import fitting.gaussian as fitting

np.set_printoptions(suppress=True)  # suppress scientific notation
start = time.clock()

# helpful variables to play with
shotnum = 0
mode = "automatic"  # auto or manual
kernel = 3  # for gaussian smoothing
n = 1  # number of fit iterations
mag = 3  # magnification

pixelsize = 3.75e-3  # 3.75 um, reported in mm.
lam = 589.158e-9  # resonant wavelength
delta = 2 * math.pi * 1  # beam detuning in MHz
Gamma = 2 * math.pi * 9.7946  # D2 linewidth in MHz

# de-enhanging factor
if mode == "automatic":
    f = 5
elif mode == "manual":
    f = 3
norm_min = -0.1  # absolute color scale min
norm_max = 1.0  # absolute color scale max
c = "gray"  # colormap for plotting
plt.figure(figsize=(20, 13))  # dimensions of final plot
plt.show(block=False)  # continue computation

stop = time.clock()
print("Initializing took " + str(round(stop - start, 2)) + " seconds")
print(" ")

##################################################
######### 1: READ FILES AND CREATE DATA ##########
##################################################

# main data analysis loop; runs until ctrl+c or ctrl+. interrupts
while True:
    initial = time.clock()
    print("SHOT " + str(shotnum))
    print(" ")

    print("1: READ FILES AND CREATE DATA")
    print("-------------------------------")

    count = 0
    despacito2 = []
    path_to_watch = "."
    before = dict([(a, None) for a in os.listdir(path_to_watch)])
    print("Watching for new files...")

    try:  # main file monitoring loop - counts to 3
        while True:
            after = dict([(a, None) for a in os.listdir(path_to_watch)])
            added = [a for a in after if not a in before]
            removed = [a for a in before if not a in after]

            if added:  # append filename to despacito2; move counter up
                print("Added: ", ", ".join(added))
                despacito2.extend(added)
                for filename in added:
                    count += 1
                if count >= 3:
                    raise hp.MyExcept  # found 3 images; exit the loop
            if removed:  # move counter down or keep it at 0
                print("Removed: ", ", ".join(removed))
                if despacito2 != []:  # files were deleted mid-shot
                    for filename in removed:
                        del despacito2[-1]
                        count -= 1
                else:  # some other file was deleted: do nothing
                    pass

            before = after
    except hp.MyExcept:
        start = time.clock()
    despacito2.sort()  # make sure the images are in order

    shot = shots.Shot(despacito2)
    width, height = shot.width, shot.height
    x, y = shot.meshgrid
    pixels = shot.pixels
    transmission = shot.transmission

    # # show transmission plot - debugging purposes only
    # plt.close()
    # plt.figure(1)
    # plt.imshow(transmission, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.draw() # display now; keep computing

    stop = time.clock()
    print("Writing data took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 2: GAUSSIAN FITTING (2D IMAGE) #########
    ##################################################

    print("2: GAUSSIAN FITTING (2D IMAGE)")
    print("-------------------------------")
    start = time.clock()
    final_error, best, zoomed, int_error = fitting.two_D_gaussian(mode, f, shot, n)
    stop = time.clock()
    print("2D fitting took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######## 3: GAUSSIAN FITTING (1D SLICES) #########
    ##################################################

    print("3: GAUSSIAN FITTING (1D SLICES)")
    print("-------------------------------")
    start = time.clock()
    fit_h, fit_v, param_h, param_v, x_hor, y_hor, x_ver, y_ver, x_axis, y_axis, horizontal, vertical = fitting.one_D_gaussian(
        shot, best
    )
    stop = time.clock()
    print("1D fitting took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ########## 4: PHYSICAL DENSITY ANALYSIS ##########
    ##################################################

    print("4: PHYSICAL DENSITY ANALYSIS")
    print("-------------------------------")
    start = time.clock()

    # sodium and camera parameters
    sigma_0 = (3.0 / (2.0 * math.pi)) * (lam) ** 2  # cross-section
    sigma = sigma_0 / (1 + (delta / (Gamma / 2)) ** 2)  # off resonance
    area = (pixelsize * 1e-3 * mag) ** 2  # pixel area in SI units

    density = -np.log(transmission)
    if mode == "manual":
        density = -np.log(zoomed)
    atom_num = (area / sigma) * np.sum(density)
    print("Atom number: " + str(np.round(atom_num / 1e6, 2)) + " million")

    stop = time.clock()
    print("Doing physics took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 5: GRAPHS, PLOTS, AND PICTURES #########
    ##################################################

    print("5: GRAPHS, PLOTS, AND PICTURES")
    print("-------------------------------")
    start = time.clock()

    # # preliminary plots: 3 images and transmission
    # plt.figure(1)
    # plt.imshow(data, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.figure(2)
    # plt.imshow(beam, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.figure(3)
    # plt.imshow(dark, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.figure(4)
    # plt.imshow(data - dark, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.figure(5)
    # plt.imshow(beam - dark, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.figure(6)
    # plt.imshow(transmission, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.show()

    # # coarse and fine fits, and relative errors
    # plt.figure(1)
    # plt.imshow(final_error, cmap = c, extent = pixels)
    # plt.colorbar()
    # plt.show()

    stop = time.clock()
    print("Plotting graphs took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 6: SAVE AND DISPLAY FINAL PLOT #########
    ##################################################

    # create figure: 2x3 grid (boxes labeled 0-5) containing:
    # - best-fit parameter outputs (5)
    # - horizontal (1) and vertical (3) plots
    # - transmission plot, in color (4)
    # - empty boxes: 0, 2

    print("6: SAVE AND DISPLAY FINAL PLOT")
    print("-------------------------------")
    start = time.clock()

    print("Painting Gaussian fits in oil")
    norm = plt.Normalize(norm_min, norm_max)

    fig = plt.figure(1)
    wr = [0.9, 8, 1.1]
    hr = [1, 9]
    gs = gridspec.GridSpec(2, 3, width_ratios=wr, height_ratios=hr)
    font = {"size": 16}
    plt.rc("font", **font)

    # convert best-fit parameters to text
    title = "Shot " + str(shotnum)
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
    # text6 = 'theta = '+ theta + ' rad'
    text7 = "N = " + str(np.round(atom_num / 1000000.0, 2)) + " million"

    # best-fit parameters: display
    ax5 = plt.subplot(gs[5])
    plt.axis("off")
    plt.text(0, 0.9, title, fontsize=24)
    plt.text(0, 0.7, text1)
    plt.text(0, 0.6, text2)
    plt.text(0, 0.5, text3)
    plt.text(0, 0.4, text4)
    plt.text(0, 0.3, text5)
    plt.text(0, 0.2, text7)

    # horizontal and vertical 1D fits

    ax1 = plt.subplot(gs[1])
    plt.plot(x_axis, 1 - horizontal, "ko", markersize=2)
    plt.plot(x_axis, 1 - fit_h, "r", linewidth=1)
    plt.xlim(0, width)
    plt.ylim(norm_min, norm_max)
    plt.gca().axes.get_xaxis().set_visible(False)

    ax3 = plt.subplot(gs[3])
    plt.plot(1 - vertical, y_axis, "ko", markersize=2)
    plt.plot(1 - fit_v, y_axis, "r", linewidth=1)
    plt.plot()
    plt.xlim(norm_max, norm_min)
    plt.ylim(0, height)
    plt.gca().axes.get_yaxis().set_visible(False)

    # transmission plot with axis lines and zoom box
    ax4 = plt.subplot(gs[4])
    plt.imshow(1 - transmission, cmap=c, norm=norm, extent=pixels)
    plt.plot(pixelsize * x_hor, pixelsize * y_hor, color="g", linewidth=0.5)
    plt.plot(pixelsize * x_ver, pixelsize * y_ver, color="g", linewidth=0.5)

    plt.xlim(pixels[0], pixels[1])
    plt.ylim(pixels[3], pixels[2])  # y-axis is upside down!

    # save best-fit parameters and image to files
    save_path = ".."
    now = time.strftime("%Y%m%d-%H%M%S")

    pic_path = save_path + "/Analysis Results/" + now + ".png"
    txt_path = save_path + "/Analysis Results/diary.txt"

    print("Saving image and writing to diary")
    diary = open(txt_path, "a+")
    diary_text = (now, np.round(best, 2), np.round(int_error, 2))
    diary.write("Time: %s. Fit: %s. Error: %s. \n" % diary_text)
    diary.close()

    plt.ion()
    plt.pause(0.01)
    plt.draw()
    plt.savefig(pic_path, dpi=150)

    stop = time.clock()
    final = time.clock()
    print("Saving results took " + str(round(stop - start, 2)) + " seconds")
    print("Total runtime: " + str(round(final - initial, 2)) + " seconds")
    print("Ready for the next shot!")
    print(" ")

    shotnum += 1
