# AbsorptionImaging v3.0.1a

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
4. Three ROI comparisons (under construction)
	
## Setup
To run the app without compliling, run __main__.py from main folder.

To compile into a windows .exe package, push package to release. Wait 5 minutes for processing and testing.
Using cmd:
```json
{
git tag v0.x.x
git push origin v0.x.x
}
```
