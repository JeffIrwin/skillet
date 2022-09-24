#!/bin/bash

# VTK filename to be visualized
f=./res/teapot.vtu

#f=./res/ico64.vtu
#f=./res/ico.vtu
#
#f=./scratch/rbc-sinx.vtu
#
## Legacy doesn't work?
#f=./scratch/teapot.vtk
#f=./scratch/teapot-ascii.vtk
#f=./scratch/cube.vtk
#
## polydata with texture coords
#f=./scratch/fran_cut.vtk
#
#f=./scratch/a.vtu

#=======================================================================

cargo run "$f"

