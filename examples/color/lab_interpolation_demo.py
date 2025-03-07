"""
=========================
LAB Interpolation Example
=========================

This example demonstrates how to use LAB interpolation in a LinearSegmentedColormap
and the new accessibility context manager for visualizing data with color vision 
deficiency considerations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Import directly from our module
from lib.matplotlib.pyplot_accessible_context.pyplot_accessible_context import accessible_context


# Create some sample data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# RGB interpolation (default)
colors_rgb = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
cmap_rgb = LinearSegmentedColormap.from_list("RGB", colors_rgb)
im1 = axs[0, 0].imshow(Z, cmap=cmap_rgb)
axs[0, 0].set_title("RGB Interpolation (Default)")
fig.colorbar(im1, ax=axs[0, 0])

# LAB interpolation
cmap_lab = LinearSegmentedColormap.from_list(
    "LAB", colors_rgb, interpolation_space="lab"
)
im2 = axs[0, 1].imshow(Z, cmap=cmap_lab)
axs[0, 1].set_title("LAB Interpolation")
fig.colorbar(im2, ax=axs[0, 1])

# Demonstrate accessibility context
# First, simulate how the colormaps appear to someone with protanopia
with accessible_context('protanopia'):
    im3 = axs[1, 0].imshow(Z, cmap=cmap_rgb)
    axs[1, 0].set_title("RGB Interpolation with Protanopia")
    fig.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].imshow(Z, cmap=cmap_lab)
    axs[1, 1].set_title("LAB Interpolation with Protanopia")
    fig.colorbar(im4, ax=axs[1, 1])

plt.tight_layout()
plt.show()


# Create a second figure to demonstrate line plots with accessibility context
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Standard plot
x = np.linspace(0, 10, 100)
for i in range(5):
    axs[0].plot(x, np.sin(x + i/2), label=f"Line {i+1}")
axs[0].set_title("Standard Colors")
axs[0].legend()

# With accessibility context
with accessible_context():
    for i in range(5):
        axs[1].plot(x, np.sin(x + i/2), label=f"Line {i+1}")
    axs[1].set_title("Accessible Colors")
    axs[1].legend()

plt.tight_layout()
plt.show()
