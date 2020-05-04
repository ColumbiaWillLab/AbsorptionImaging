# AbsorptionImaging v3.0.2

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

.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html
	
## Setup
To run the app without compliling, run "__main__".py from main folder.

To download the latest compiled release, https://github.com/ColumbiaWillLab/AbsorptionImaging/releases

For compiling into a windows .exe package, push package to release. Wait 5 minutes for processing and testing and it should be automatically uploaded.
Using cmd:
```
git tag v0.x.x
git push origin v0.x.x
```
