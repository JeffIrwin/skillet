
# Utility script for generating an HSV colormap
#
# Run:
#
#     python3 genhsv.py
#
import sys

cmax = 255

for i in range(0, cmax + 1):
    r = 0
    g = i
    b = cmax
    print(f"{r}u8, {g}u8, {b}u8, {cmax}u8,")

for i in range(0, cmax + 1):
    r = 0
    g = cmax
    b = cmax - i
    print(f"{r}u8, {g}u8, {b}u8, {cmax}u8,")

for i in range(0, cmax + 1):
    r = i
    g = cmax
    b = 0
    print(f"{r}u8, {g}u8, {b}u8, {cmax}u8,")

for i in range(0, cmax + 1):
    r = cmax
    g = cmax - i
    b = 0
    print(f"{r}u8, {g}u8, {b}u8, {cmax}u8,")

