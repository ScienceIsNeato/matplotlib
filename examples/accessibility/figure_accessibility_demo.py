"""
Figure-level Accessibility Demo
==============================

This example demonstrates how to use the `accessible` parameter in the Figure class
to create visualizations that are more accessible to individuals with color vision
deficiencies.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/4)

# Create a problematic colormap for demonstration
colors = [(0, 1, 0), (1, 0, 0)]  # Green to Red (problematic for red-green color blindness)
cmap_problematic = LinearSegmentedColormap.from_list("GreenToRed", colors)

# Create a 2x2 grid of figures to compare different settings
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Accessibility Comparison', fontsize=16)

# Regular figure (no accessibility)
axs[0, 0].plot(x, y1, 'r-', linewidth=2, label='Red')
axs[0, 0].plot(x, y2, 'g-', linewidth=2, label='Green')
axs[0, 0].plot(x, y3, 'b-', linewidth=2, label='Blue')
axs[0, 0].set_title('Regular Colors')
axs[0, 0].legend()

# Create a heatmap with the problematic colormap
data = np.random.rand(20, 20)
im1 = axs[0, 1].imshow(data, cmap=cmap_problematic)
axs[0, 1].set_title('Problematic Colormap (Red-Green)')
plt.colorbar(im1, ax=axs[0, 1])

# Set up deuteranopia simulation
plt.rcParams['accessibility.colorblind_simulation'] = 'deuteranopia'

# Create a new figure with accessibility enabled
fig_accessible = plt.figure(figsize=(6, 5), accessible=True)
ax_accessible = fig_accessible.add_subplot(111)
ax_accessible.plot(x, y1, 'r-', linewidth=2, label='Red')
ax_accessible.plot(x, y2, 'g-', linewidth=2, label='Green')
ax_accessible.plot(x, y3, 'b-', linewidth=2, label='Blue')
ax_accessible.set_title('With Deuteranopia Simulation')
ax_accessible.legend()

# Copy the accessible figure's content to our comparison grid
# This is just for demonstration purposes
axs[1, 0].plot(x, y1, 'r-', linewidth=2, label='Red')
axs[1, 0].plot(x, y2, 'g-', linewidth=2, label='Green')
axs[1, 0].plot(x, y3, 'b-', linewidth=2, label='Blue')
axs[1, 0].set_title('With Deuteranopia Simulation')
axs[1, 0].legend()

# Create a heatmap with a better colormap
im2 = axs[1, 1].imshow(data, cmap='viridis')
axs[1, 1].set_title('Better Colormap (Viridis)')
plt.colorbar(im2, ax=axs[1, 1])

# Adjust layout
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Demonstrate the accessible_context
plt.figure(figsize=(10, 6))
plt.suptitle('Using accessible_context for Temporary Accessibility', fontsize=16)

# Create a subplot grid
ax1 = plt.subplot(121)
ax1.plot(x, y1, 'r-', linewidth=2, label='Red')
ax1.plot(x, y2, 'g-', linewidth=2, label='Green')
ax1.set_title('Regular View')
ax1.legend()

# Use the accessible_context to temporarily enable accessibility
with plt.accessible_context('protanopia'):
    ax2 = plt.subplot(122)
    ax2.plot(x, y1, 'r-', linewidth=2, label='Red')
    ax2.plot(x, y2, 'g-', linewidth=2, label='Green')
    ax2.set_title('With Protanopia Simulation')
    ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show() 