### HOW TO RUN THIS PROGRAM ###
"""
STEP 0. Make sure that mode == 'automatic' down below.
STEP 1. Run the script using \Imaging Code>python Imaging.py.
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

##################################################
######## 0: CLASSES, FUNCTIONS, LIBRARIES ########
##################################################

print("0: SET UP LIBRARIES/FUNCTIONS")
print("------------------------------")

import time
start = time.clock()

import os
import shutil
import imageio
import cv2

import math
import numpy as np
np.set_printoptions(suppress=True) # suppress scientific notation

from scipy.optimize import curve_fit
from lmfit import minimize, Parameters

import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.gridspec as gridspec

# all function definitions and classes
import Helper as hp

# mode = 'manual' or 'automatic'
mode = 'automatic'

stop = time.clock()
print("Initializing took " + str(round(stop - start, 2)) + " seconds")
print(" ")

##################################################
######### 1: READ FILES AND CREATE DATA ##########
##################################################

shot = 1
plt.figure(figsize = (20,13))
plt.show(block=False)

# main data analysis loop; runs until ctrl+c or ctrl+. interrupts
while True:
    initial = time.clock()
    print("SHOT " + str(shot))
    print(" ")

    print("1: READ FILES AND CREATE DATA")
    print("-------------------------------")
    
    count = 0
    despacito2 = []
    path_to_watch = "."
    before = dict([(f, None) for f in os.listdir(path_to_watch)])
    print("Watching for new files...")
    
    try: # main file monitoring loop - counts to 3
        while True:
            after = dict ([(f, None) for f in os.listdir (path_to_watch)])
            added = [f for f in after if not f in before]
            removed = [f for f in before if not f in after]
            
            if added: # append filename to despacito2; move counter up
                print "Added: ", ", ".join(added)
                despacito2.extend(added)
                for filename in added:
                    count += 1
                if count >= 3:
                    raise hp.MyExcept # found 3 images; exit the loop
            if removed: # move counter down or keep it at 0
                print "Removed: ", ", ".join(removed)
                if despacito2 != []: # files were deleted mid-shot
                    for filename in removed:
                        del despacito2[-1]
                        count -= 1
                else: # some other file was deleted: do nothing
                    pass
            
            before = after
    except hp.MyExcept:
        start = time.clock()
    despacito2.sort() # make sure the images are in order
    
    # read images into large arrays of pixel values    
    print("Writing image data into arrays")
    data = imageio.imread(despacito2[0])
    beam = imageio.imread(despacito2[1])
    dark = imageio.imread(despacito2[2])
    width = len(data[0])
    height = len(data)
    
    # save raw images in a new folder
    garbage_path = '../Raw Data/'
    now = time.strftime("%Y%m%d-%H%M%S")
    pic_num = 1
    
    for meme in despacito2:
        name = "Raw_%s_%s.bmp" % (now, str(pic_num))
        os.rename(meme, name)
        shutil.copy2(name, garbage_path)
        pic_num += 1
        os.remove(name)

    # create a meshgrid: each pixel is (3.75 um) x (3.75 um); images
    # have resolution (964 p) x (1292 p) --> (3.615 mm) x (4.845 mm).
    pixelsize = 3.75e-3 # 3.75 um, reported in mm.
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    (x,y) = np.meshgrid(x,y)
    pixels = [0, pixelsize * width, pixelsize * height, 0]

    # create fake data for laser & atom sample; do background subtraction
    kernel = 1
    noise = 1   
    # data, beam, dark = fake_data(laser, atoms, noise)
    transmission = hp.subtraction(data, beam, dark, kernel)

    """
    # show transmission plot - debugging purposes only
    plt.close()
    plt.figure(1)
    plt.imshow(transmission, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.draw() # display now; keep computing
    """

    stop = time.clock()
    print("Writing data took " + str(round(stop - start, 2)) + " seconds")

    print(" ")

    ##################################################
    ######### 2: GAUSSIAN FITTING (2D IMAGE) #########
    ##################################################

    print("2: GAUSSIAN FITTING (2D IMAGE)")
    print("-------------------------------")
    start = time.clock()
    
    print("Mode: " + mode)
    # gaussian parameters: p = [A, x0, y0, sigma_x, sigma_y, theta, z0]

    # compute parameters automatically
    if mode == 'automatic':
        f = 5
        
        # coarsen the image; create a coarse meshgrid for plotting
        coarse = hp.de_enhance(transmission, f)
        x_c = np.linspace(f, len(coarse[0])*f, len(coarse[0]))
        y_c = np.linspace(f, len(coarse) * f, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)

        # take an "intelligent" guess and run the coarse fit
        (y0, x0, peak) = hp.peak_find(coarse, f) # guess an initial center point
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        guess = [amp, x0, y0, 200, 200, 0, z0]
        coarse_fit, best = hp.iterfit(hp.residual,guess,x_c,y_c,width,height,coarse,1)

        # compute the relative error from the coarse fit
        error = (coarse - coarse_fit) / coarse
        area = (width * height) / (f**2)
        int_error = (np.sum((error)**2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))
        
    # guess parameters based on user input
    elif mode == 'manual':
        f = 3
        
        # allow the user to select a region of interest
        r = cv2.selectROI(transmission)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # zoom in, coarsen, and create a coarse meshgrid
        zoomed = hp.zoom_in(transmission, r)
        coarse = hp.de_enhance(zoomed, f)
        x_c = np.linspace(f, len(coarse[0])*f, len(coarse[0]))
        y_c = np.linspace(f, len(coarse) * f, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)
        
        # take an intelligent guess at fit parameters
        (y0, x0, peak) = hp.peak_find(coarse, f)
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        sigma_x = 0.5*(r[2]/f)
        sigma_y = 0.5*(r[3]/f)
        guess = [amp, x0, y0, sigma_x, sigma_y, 0, z0]
        
        # run the zoomed-in fit and compute its relative error
        fine_fit, best = hp.iterfit(hp.residual,guess,x_c,y_c,width,height,coarse,1)
        best[1] = best[1] + r[0]
        best[2] = best[2] + r[1]
        error = (coarse - fine_fit) / coarse
        area = (r[2] * r[3]) / (f**2)
        int_error = (np.sum((error)**2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))
    
    # generate final-fit transmission data; compute relative error
    params0 = hp.list2params(best)
    fit_data = 1 - hp.gaussian(params0, x,y)
    final_error = (transmission - fit_data) / transmission

    stop = time.clock()
    print("2D fitting took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######## 3: GAUSSIAN FITTING (1D SLICES) #########
    ##################################################

    print("3: GAUSSIAN FITTING (1D SLICES)")
    print("-------------------------------")
    start = time.clock()

    # define the best-fit axes
    x_val = np.linspace(-2*width, 2*width, 4*width)
    y_val = np.linspace(-2*height, 2*height, 4*height)
    (x_hor, y_hor, x_ver, y_ver) = hp.lines(x_val, best)

    # collect (Gaussian) data along these axes
    print("Collecting 1D data")
    (x_axis, horizontal) = hp.collect_data(transmission, x_hor, y_hor, 'x')
    (y_axis, vertical) = hp.collect_data(transmission, x_ver, y_ver, 'y')

    # perform a 1D Gaussian fit on each data set:
    # for the 1D fits, take the guess [A, x0/y0, sigma_x/sigma_y, z0]
    guess_h = np.array([best[0], best[1], best[3], best[6]])
    guess_v = np.array([best[0], best[2], best[4], best[6]])

    # perform the horizontal and vertical 1D fits
    fit_h, param_h = hp.fit_1d(hp.residual_1d, guess_h, x_axis, horizontal)
    fit_v, param_v = hp.fit_1d(hp.residual_1d, guess_v, y_axis, vertical)

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
    lam = 589.158e-9 # resonant wavelength
    delta = 0 # detuning :)
    sigma_0 = (3.0/(2.0*math.pi)) * (lam)**2 # cross-section
    sigma = sigma_0 / (1 + delta**2) # cross-section off resonance
    area = (pixelsize * 1e-3)**2 # pixel area in SI units
    
    density = -np.log(transmission)
    if mode == 'manual':
        density = -np.log(zoomed)
    atom_num = (area/sigma) * np.sum(density)
    print("Atom number: " + str(np.round(atom_num/1e6, 2)) + " million")

    stop = time.clock()
    print("Doing physics took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 5: GRAPHS, PLOTS, AND PICTURES #########
    ##################################################

    print("5: GRAPHS, PLOTS, AND PICTURES")
    print("-------------------------------")
    start = time.clock()

    # preliminary plots: 3 images and transmission
    """
    plt.figure(1)
    plt.imshow(data, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(2)
    plt.imshow(beam, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(3)
    plt.imshow(dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(4)
    plt.imshow(data - dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(5)
    plt.imshow(beam - dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(6)
    plt.imshow(transmission, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.show()
    """

    # coarse and fine fits, and relative errors
    """
    plt.figure(1)
    plt.imshow(final_error, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.show()
    """
    
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
    norm_min = -0.01
    norm_max = 0.99
    norm = plt.Normalize(norm_min, norm_max)
    
    fig = plt.figure(1)
    wr = [0.9, 8, 1.1]
    hr = [1, 9]
    gs = gridspec.GridSpec(2, 3, width_ratios = wr, height_ratios = hr)
    font = {'size'   : 16}
    plt.rc('font', **font)

    # convert best-fit parameters to text
    title = 'Shot ' + str(shot)
    A = str(np.round(best[0], 2))
    x_0 = str(np.round(pixelsize * best[1], 3))
    y_0 = str(np.round(pixelsize * best[2], 3))
    w_x = str(np.round(2 * pixelsize * best[3], 3))
    w_y = str(np.round(2 * pixelsize * best[4], 3))
    theta = str(np.round(best[5], 2))
    z_0 = str(np.round(best[6], 2))

    text1 = 'A = ' + A
    text2 = 'x_0 = ' + x_0
    text3 = 'y_0 = ' + y_0
    text4 = 'w_x = '+ w_x
    text5 = 'w_y = '+ w_y
    # text6 = 'theta = '+ theta + ' rad'
    text7 = 'N = ' + str(np.round(atom_num/1000000.0, 2)) + ' million'

    # best-fit parameters: display
    ax5 = plt.subplot(gs[5])
    plt.axis('off')
    plt.text(0, 0.9, title, fontsize = 24)
    plt.text(0, 0.7, text1)
    plt.text(0, 0.6, text2)
    plt.text(0, 0.5, text3)
    plt.text(0, 0.4, text4)
    plt.text(0, 0.3, text5)
    plt.text(0, 0.2, text7)

    # horizontal and vertical 1D fits
    ax1 = plt.subplot(gs[1])
    plt.plot(x_axis, 1 - horizontal, 'ko', markersize = 2)
    plt.plot(x_axis, 1 - fit_h, 'r', linewidth = 1)
    plt.xlim(0, width)
    plt.ylim(norm_min, norm_max)
    plt.gca().axes.get_xaxis().set_visible(False)

    ax3 = plt.subplot(gs[3])
    plt.plot(1 - vertical, y_axis, 'ko', markersize = 2)
    plt.plot(1 - fit_v, y_axis, 'r', linewidth = 1)
    plt.xlim(norm_max, norm_min)
    plt.ylim(height, 0)
    plt.gca().axes.get_yaxis().set_visible(False)

    # transmission plot with axis lines and zoom box
    ax4 = plt.subplot(gs[4])
    plt.imshow(1 - transmission, cmap='gray', norm=norm, extent=pixels)
    plt.plot(pixelsize*x_hor, pixelsize*y_hor, color = 'g', linewidth = 0.5)
    plt.plot(pixelsize*x_ver, pixelsize*y_ver, color = 'g', linewidth = 0.5)

    plt.xlim(pixels[0], pixels[1])
    plt.ylim(pixels[3], pixels[2]) # y-axis is upside down!

    # save best-fit parameters and image to files
    save_path = '..'
    now = time.strftime("%Y%m%d-%H%M%S")

    pic_path = save_path + '/Analysis Results/'+ now + '.png'
    txt_path = save_path + '/Analysis Results/diary.txt'

    print("Saving image and writing to diary")
    diary = open(txt_path, "a+")
    diary_text = (now, np.round(best, 2), np.round(int_error, 2))
    diary.write("Time: %s. Fit: %s. Error: %s. \n" % diary_text)
    diary.close()

    plt.ion()
    plt.pause(.01)
    plt.draw()
    plt.savefig(pic_path, dpi = 150)

    stop = time.clock()
    final = time.clock()
    print("Saving results took " + str(round(stop - start, 2)) + " seconds")
    print("Total runtime: " + str(round(final - initial, 2)) + " seconds")
    print("Ready for the next shot!")
    print(" ")

    shot += 1