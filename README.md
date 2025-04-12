# Computational Astrodynamics Implementations
This repository contains my solutions to one of the problem sets from a master's course in Computational Astrodynamics. It includes two Jupyter notebooks.

### Notebook 1: Ground Track Visualization

This notebook focuses on plotting 24-hour ground tracks for five satellites with specified initial conditions at *t = 0*. The satellites included are:

- ISS  
- A Sentinel satellite  
- A Molniya satellite  
- Two geosynchronous satellites (one of which is geostationary)

The first step involved completing the necessary functions in astrodynamicslibrary.py to produce the required plots. In the second part of the exercise, proper initial conditions for the geostationary satellite *(a = 42164.0 km, e = 0.0001, i = 0°)* were determined so that it remains fixed over the longitude of Thessaloniki.

### Notebook 2: Earth-to-Mars Porkchop Plot

This notebook generates a porkchop plot for an interplanetary transfer from Earth to Mars. 

To produce the plot, Lambert’s problem was solved for a range of departure and arrival dates, allowing for the visualization of optimal transfer windows.
