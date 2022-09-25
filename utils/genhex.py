
# Make an unstructured grid of hex cells in legacy VTK format.  This can be
# opened in ParaView to save as XML format (.vtu).

npx = 4
npy = 5
npz = 6

ncx = npx - 1
ncy = npy - 1
ncz = npz - 1

nxy = npx * npy
np  = npx * npy * npz
nc  = ncx * ncy * ncz

# num points per cell
nppc = 8

csize = (nppc + 1) * nc

f = open("./scratch/hex.vtk", "w")

print("# vtk DataFile Version 2.0", file = f)
print("Really cool data", file = f)
print("ASCII", file = f)
print("DATASET UNSTRUCTURED_GRID", file = f)

print(f"POINTS {np} float", file = f)
for k in range(0, npz):
    for j in range(0, npy):
        for i in range(0, npx):
            print(f"{i} {j} {k}", file = f)

print(f"CELLS {nc} {csize}", file = f)
for k in range(0, ncz):
    for j in range(0, ncy):
        for i in range(0, ncx):

            # Ref:  http://www.princeton.edu/~efeibush/viscourse/vtk.pdf
            i0 = (k+0) * nxy + (j+0) * npx + (i+0)
            i1 = (k+0) * nxy + (j+0) * npx + (i+1)
            i2 = (k+0) * nxy + (j+1) * npx + (i+1)
            i3 = (k+0) * nxy + (j+1) * npx + (i+0)
            i4 = (k+1) * nxy + (j+0) * npx + (i+0)
            i5 = (k+1) * nxy + (j+0) * npx + (i+1)
            i6 = (k+1) * nxy + (j+1) * npx + (i+1)
            i7 = (k+1) * nxy + (j+1) * npx + (i+0)

            print(f"{nppc} {i0} {i1} {i2} {i3} {i4} {i5} {i6} {i7}", file = f)

print(f"CELL_TYPES {nc}", file = f)
for i in range(0, nc):
    print("12", file = f)

