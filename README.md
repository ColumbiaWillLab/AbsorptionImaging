# AbsorptionImaging v0.3.2c

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

## Logging.hdf5
For documentation and recording of analysis results, we use the hdf5 format. For more information on the file format, you can refer to https://www.hdfgroup.org/.

Related documentation on the Python package can be found https://www.h5py.org/.

Every time a shot or sequence is processed, relevant variables are saved in logging.hdf5, found in the "../Raw Data/" folder. Accessing of the data can be done with the hdf5viewer, installed separately or via Python.

The file structure for the HDF5 format is as follows:

logging.hdf5
├── HH:MM:SS
├── HH:MM:SS
├── tof_sequence
│   ├── HH:MM:SS
│   └── HH:MM:SS
└── atomnum_sequence
    ├── HH:MM:SS
    └── HH:MM:SS

Each processed object is indexed by the 24hr timestamp (HH:MM:SS) it has been processed at. Each timestamp object contains the full dataset (atom,beam,dark), and can be identified via its variables.

It contains a snapshot of its associated config settings, filename, and as much relevant experimental parameters.

For time of flight (tof_sequence) and atom number optimization (atomnum_sequence), they are saved as a subgroup. Datasets are not saved as part of the sequence, but the used shots and their associated filenames can be found as a list variable.
