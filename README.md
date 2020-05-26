# AbsorptionImaging v0.3.2b

## Table of contents
* [General info](#general-info)
* [Features](#features)
* [Setup](#setup)

## General info
Configurable absorption imaging pipeline for ultracold atom experiments. Computes and fits physical properties from raw camera output.

Developed from original CLI to provide:
- Concurrent processing of images and display
- Image logging (up to 15, configurable)
- Configurability for atomic species
- User-friendly GUI for ease of usability
	
## Features
### Image processing utilities
1. Thermal cloud fitting with multiple options
2. Temperature fitting from time-of-flight image sequence
3. Atom number optimization
4. Three ROI comparisons

### Universal Camera compatibility
This package is compatible with any imaging hardware, automatically processes any three images labelled at end with "filename-1", "filename-2", "filename-3" that is saved in the same directory.

Current version only processes .bmp files.

For optimal performance, use with Mako cameras and camera trigger saver program:
https://github.com/ColumbiaWillLab/CameraTriggerSaver

## Directory Structure
The structure of the code follows the implementation of typical absorption image processing scripts with a Model–view–presenter (MVP) data structure using the tkinter package. The tkinter package provides standard python GUI widgets in a (reasonably) easy manner to implement.

For the beginner, this allows us to compartmentalize the functionality of the code in the name of organization, and prevents the well-intentioned grad student from unwittingly changing things that they should not.

### Purpose of the various folders:

* models/
    + an interface defining the data to be displayed or otherwise acted upon in the user interface, e.g. contains fitting functions etc.
* presenters/
    + Interface for retrieving data from model to be processed and presented in view and vice versa.
* tests/
    + Contains tests information that will be conducted during the compilation process.
* utils/
    + Contains various useful functions that can be called.
* views/
    + Sets up your GUI.
* workers/
    + script that begins processing sequence once it detects the three images saved in the directory.

In addition to the file, on processing the first image, the package moves the original set of three images into a separate folder:
../Raw Data/
and the absorption image shot into:
../Data Analysis/

###File Structure
```bash
.
├── README.md
├── __main__.py
├── __pycache__
│   └── config.cpython-35.pyc
├── config.ini
├── config.py
├── models
│   ├── __init__.py
│   ├── sequences.py
│   └── shots.py
├── presenters
│   ├── __init__.py
│   ├── application.py
│   ├── logs.py
│   ├── sequences.py
│   └── shots.py
├── pytest.ini
├── requirements.txt
├── tests
│   ├── __init__.py
│   ├── config.ini
│   ├── data
│   │   ├── old
│   │   │   ├── tricky
│   │   │   │   ├── Raw_20190522-105317_1.bmp
│   │   │   │   ├── Raw_20190522-105317_2.bmp
│   │   │   │   ├── Raw_20190522-105317_3.bmp
│   │   │   │   ├── Raw_20190522-130927_1.bmp
│   │   │   │   ├── Raw_20190522-130927_2.bmp
│   │   │   │   └── Raw_20190522-130927_3.bmp
│   │   │   ├── typical
│   │   │   │   ├── Raw_20190520-151535_1.bmp
│   │   │   │   ├── Raw_20190520-151535_2.bmp
│   │   │   │   └── Raw_20190520-151535_3.bmp
│   │   │   └── underflow
│   │   │       ├── Raw_20190524-195614_1.bmp
│   │   │       ├── Raw_20190524-195614_2.bmp
│   │   │       └── Raw_20190524-195614_3.bmp
│   │   └── saturated
│   │       ├── 2019-07-26T162222-1.bmp
│   │       ├── 2019-07-26T162222-2.bmp
│   │       └── 2019-07-26T162222-3.bmp
│   └── test_shots.py
├── utils
│   ├── __init__.py
│   ├── fitting.py
│   ├── geometry.py
│   └── threading.py
├── views
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-35.pyc
│   │   ├── application.cpython-35.pyc
│   │   ├── logs.cpython-35.pyc
│   │   └── plots.cpython-35.pyc
│   ├── application.py
│   ├── components.py
│   ├── logs.py
│   ├── plots.py
│   ├── sequences.py
│   └── shots.py
└── workers
    ├── __init__.py
    └── file_watcher.py
```	

## Setup
To run the app without compliling, run "__main__".py from main folder.

To download the latest compiled release, https://github.com/ColumbiaWillLab/AbsorptionImaging/releases

For compiling into a windows .exe package, push package to release. Wait 5 minutes for processing and testing and it should be automatically uploaded.
Using cmd:
```
git tag v0.x.x
git push origin v0.x.x
```

## Logging.csv
Every time a shot or sequence is processed, relevant variables are saved in logging.csv, found in the "../Raw Data/" folder.

The index for the variables saved under the header are as follows:
["filename", "magnification", "atom number", "fitted shot", "tof_sequence", "time_sequence", "average_T (uK)"]

1. filename 
    returns shot.name / timestamp of the three shots
    - str() single image by default
    - list() for a processed sequence

2. magnification
    returns config.magnification

3. atom number
    returns the processed shot.atom_number
    - float() for a single show
    - array(float()) for tof sequence

4. fitted shot
    returns True if shot was fitted

5. tof_sequence
    returns True for processed tof_sequence

6. time_sequence
    returns list of user-input tof values

7. average_T (uK)
    returns fitted averaged temperature in uK for tof sequence

8. "threeroi"
    returns [] if threeroi is not enabled
    returns three roi array() if enabled

9. a_b_ratio
    returns the calculated atom ratio between ROI A and ROI B if threeroi is enabled

10. Comments
    returns input string in "Settings" tab.
