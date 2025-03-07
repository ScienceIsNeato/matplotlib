# Matplotlib Accessibility Features

This module provides accessibility enhancements for Matplotlib visualizations. The main features include:

1. **Color vision deficiency simulation**: Simulate how plots appear to people with protanopia or deuteranopia.
2. **Accessible color cycles**: Use color schemes optimized for people with color vision deficiencies.
3. **Improved readability**: Enhanced text size, line widths, and contrast settings.
4. **Context managers**: Temporarily apply accessibility features using Python context managers.

## Usage

### Basic Usage with Context Manager

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot_accessible_context import accessible_context

# Create a plot with default settings
plt.figure(figsize=(12, 5))

# Create a subplot with regular settings
plt.subplot(121)
x = np.linspace(0, 10, 100)
for i in range(5):
    plt.plot(x, np.sin(x + i/2), label=f"Line {i+1}")
plt.title("Standard Colors")
plt.legend()

# Create a subplot with accessibility settings
plt.subplot(122)
with accessible_context():
    for i in range(5):
        plt.plot(x, np.sin(x + i/2), label=f"Line {i+1}")
    plt.title("Accessible Colors")
    plt.legend()

plt.tight_layout()
plt.show()
```

### Simulating Color Vision Deficiencies

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot_accessible_context import accessible_context

# Create a colormap visualization
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create a figure with multiple views
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

# Simulate protanopia
with accessible_context('protanopia'):
    im3 = axs[1, 0].imshow(Z, cmap=cmap_rgb)
    axs[1, 0].set_title("RGB Interpolation with Protanopia")
    fig.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].imshow(Z, cmap=cmap_lab)
    axs[1, 1].set_title("LAB Interpolation with Protanopia")
    fig.colorbar(im4, ax=axs[1, 1])

plt.tight_layout()
plt.show()
```

## Configuration

Matplotlib's RCPRAMS include several accessibility settings that can be configured in your matplotlibrc file:

```
accessibility.high_contrast_mode: False        # Enable high contrast mode
accessibility.colorblind_mode: None            # None, 'protanopia', 'deuteranopia'
accessibility.colorblind_simulation: False     # Enable colorblind simulation
accessibility.large_text: False                # Enable larger text
accessibility.high_visibility_colors: False    # Use high visibility color cycles
```

Or programmatically:

```python
import matplotlib as mpl
mpl.rcParams['accessibility.high_contrast_mode'] = True
mpl.rcParams['accessibility.large_text'] = True
```

## Additional Resources

For more information, see the [Accessibility Guide](../../doc/users/explain/accessibility.rst) in the matplotlib documentation.
