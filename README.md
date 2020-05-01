# AbsorptionImaging v3.0.1a

## Table of contents
* [General info](#general-info)
* [Features](#features)
* [Setup](#setup)

## General info
Configurable absorption imaging pipeline for ultracold atom experiments. Computes and fits physical properties from raw camera output.

Motivations for transitioning from a python script to a GUI were its user-friendly interface and low barrier to entry for usability.
	
## Features
- Concurrent processing of images and display
- Image logging (up to 15, configurable)
- Configurability for atomic species

### Image Processing utilities
1. 1D_summed fitting 
2. Temperature fitting from time-of-flight image sequence
3. Atom number optimization
4. Three ROI comparisons (under construction)
	
## Setup
To run the app without compliling, run __main__.py from folder.

To compile into a windows .exe file, push package to release.
