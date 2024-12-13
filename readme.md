<p align="center">
  <img src="atpase_loc.png">
</p>

# Mitochondrial morphology provides a mechanism for energy buffering at synapses
[DOI:10.1038/s41598-019-54159-1](https://doi.org/10.1038/s41598-019-54159-1)

Welcome to the repository of the mitochondrial morphology project. This repository contains all the final code used for this publication, for the simulations and images in the paper. Other materials as the electron tomogram used to generate the meshes, and the meshes themselves can be found [here](https://doi.org/10.17881/lcsb.20190507.01).

## Folders

For this publication, we performed four computational experiments. In the folder `mcell`  you can find the code for the spatial MCell simulations, each experiment has its directory. The names of the experiments are the same as in the publication. Three different spatial configurations of the proteins were explored they correspond to the directories `cristae`, `ibm`, and `both`. In the folder  `odes_and_figures` is the code used to run the space-independent ordinary differential equation (ODE) approach, as before each experiment has its own directory. These scripts are written in Python 3. Depending on the code you want to run the required software, please refer to the Installation section for details.

## Installation
Spatiotemporal simulations were performed with [MCell](https://mcell.org/) (version 3.4). The corresponding ODEs were integrated with [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html) (version 0.88), and the following Python libraries were also used for analysis and to produce the figures: [Numpy](https://numpy.org/) (version 1.18.1) and [Matplotlib](https://matplotlib.org/) (version 3.2.0).
