"""
Figure-level Accessibility Test

This example demonstrates the use of the `accessible` parameter in the Figure class.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create a figure with accessibility disabled (default)
fig1 = plt.figure(figsize=(8, 4))
fig1.suptitle('Accessibility Disabled (Default)')

# Create a colorful plot
ax1 = fig1.add_subplot(121)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'r-', linewidth=2, label='Red')
ax1.plot(x, np.cos(x), 'g-', linewidth=2, label='Green')
ax1.plot(x, np.sin(x+np.pi/4), 'b-', linewidth=2, label='Blue')
ax1.set_title('Regular Colors')
ax1.legend()

# Create a colormap plot
ax2 = fig1.add_subplot(122)
data = np.random.rand(10, 10)
im = ax2.imshow(data, cmap='viridis')
ax2.set_title('Regular Colormap')
fig1.colorbar(im, ax=ax2)

# Create a figure with accessibility enabled
fig2 = plt.figure(figsize=(8, 4), accessible=True)
fig2.suptitle('Accessibility Enabled')

# Create the same colorful plot
ax3 = fig2.add_subplot(121)
ax3.plot(x, np.sin(x), 'r-', linewidth=2, label='Red')
ax3.plot(x, np.cos(x), 'g-', linewidth=2, label='Green')
ax3.plot(x, np.sin(x+np.pi/4), 'b-', linewidth=2, label='Blue')
ax3.set_title('Accessible Colors')
ax3.legend()

# Create the same colormap plot
ax4 = fig2.add_subplot(122)
im2 = ax4.imshow(data, cmap='viridis')
ax4.set_title('Accessible Colormap')
fig2.colorbar(im2, ax=ax4)

# Set the colorblind simulation type
plt.rcParams['accessibility.colorblind_simulation'] = 'deuteranopia'

# Show the figures
plt.show()

# Save the figures
fig1.savefig('regular_figure.png')
fig2.savefig('accessible_figure.png')

print("Test completed. Check the figures to see the difference between regular and accessible rendering.") 