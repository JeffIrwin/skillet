
*It's not rust, it's a layer of seasoning!*

# Skillet

Skillet is a lightweight, deep-fried, rust application for interactive scientific visualization with [vtkio](https://github.com/elrnv/vtkio) and [glium](https://github.com/glium/glium).

Compare skillet with [ParaView](https://www.paraview.org/):

| Skillet                     | ParaView               |
| -----------                 | -----------            |
| ![](doc/skillet-teapot.png) | ![](doc/pv-teapot.png) |

# Features

## Data arrays types

| Data array      | Skillet support?   |
| -----------     | -----------        |
| Point data      |  ✔               |
| Cell data       |  ✔               |
| Scalars         |  ✔               |
| Vectors         |  ✔               |
| Tensors         |  ✔               |
| Generic         |  ✔               |

## Cell types

| Cell            | Skillet support?   |
| -----------     | -----------        |
| Triangle        |  ✔               |
| Quad            |  ✔               |
| Tetra           |  ✔               |
| Hexahedron      |  ✔               |
| Wedge           |  ✔               |
| Pyramid         |  ✔               |
| Vertex cells    |  ❌               |
| Line cells      |  ❌               |
| Triangle strip  |  ❌               |
| Polygon         |  ❌               |
| Pixel           |  ❌               |
| Voxel           |  ❌               |
| Quadratic cells |  ❌               |

## File formats

| File (extension)            | Skillet support?   |
| -----------                 | -----------        |
| Unstructured grid (`.vtu`)  |  ✔               |
| Image data        (`.vti`)  |  ❌               |
| Poly data         (`.vtp`)  |  ❌               |
| Rectilinear grid  (`.vtr`)  |  ❌               |
| Structured grid   (`.vts`)  |  ❌               |
| Parallel files    (`.*pv*`) |  ❌               |
| Legacy files      (`.vtk`)  |  ❌               |
| Multiple piece data         |  ❌               |

## Operating systems

| OS            | Skillet support?   |
| -----------   | -----------        |
| Windows       |  ✔               |
| Ubuntu        |  ❌               |

