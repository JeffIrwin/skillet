#!/bin/bash

# VTK filename to be visualized
f=./res/teapot.vtu

## A vector result can be created in ParaView with Calculator -> coords.  Then
## a tensor result can be created with Python calculator -> gradient(MyVector).
##
## Ref:
##     https://vtk.org/Wiki/Python_Calculator#A_more_complex_example
##
#f=./res/ico-tensor.vtu

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

