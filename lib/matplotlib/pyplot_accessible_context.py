"""
Accessibility context managers for matplotlib.

This module provides context managers for temporarily enabling accessibility features
in matplotlib. This includes color vision deficiency simulation, high contrast 
options, and other accessibility-related settings.
"""

import contextlib
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np
from matplotlib.colors import simulate_colorblindness
from matplotlib.cm import ScalarMappable


class AccessibleContext:
    """
    A context manager for temporarily enabling accessibility features.

    Parameters
    ----------
    deficiency_type : str or list of str, optional
        The types of color vision deficiencies to simulate. Can be 'protanopia',
        'deuteranopia', or None. Default is None, which means no simulation.
    """
    def __init__(self, deficiency_type=None):
        self.deficiency_type = deficiency_type
        self._original_settings = {}
        self._original_colormaps = {}
        self._original_colormap_handlers = {}
        self._active_figure_managers = {}
        
    def __enter__(self):
        # Save original settings
        self._original_settings = {
            'axes.prop_cycle': rcParams['axes.prop_cycle'],
            'image.cmap': rcParams['image.cmap']
        }
        
        # Apply accessibility settings
        if self.deficiency_type:
            # Apply color vision deficiency simulation
            self._apply_cvd_simulation()
        else:
            # Apply general accessibility enhancements
            self._apply_accessibility_settings()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original settings
        for key, value in self._original_settings.items():
            rcParams[key] = value
            
        # Restore original colormaps if they were modified
        for name, cmap in self._original_colormaps.items():
            mpl.colormaps[name] = cmap
            
        # Execute any additional cleanup
        for func in self._original_colormap_handlers.values():
            if func:
                func()
    
    def _apply_cvd_simulation(self):
        """Apply color vision deficiency simulation to colormaps and colors."""
        # Apply color vision deficiency simulation to the default colormap
        default_cmap_name = rcParams['image.cmap']
        self._original_colormaps[default_cmap_name] = mpl.colormaps[default_cmap_name]
        
        # Modify default color cycle for CVD simulation
        colors = rcParams['axes.prop_cycle'].by_key()['color']
        simulated_colors = []
        
        for color in colors:
            # Convert hex color to RGB
            if color.startswith('#'):
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                rgb = [r, g, b]
            else:
                # Handle named colors by using their RGB values
                rgb = mpl.colors.to_rgb(color)
                
            # Simulate color vision deficiency
            sim_rgb = simulate_colorblindness(rgb, self.deficiency_type)
            
            # Convert back to hex
            hex_color = mpl.colors.to_hex(sim_rgb)
            simulated_colors.append(hex_color)
        
        # Update the color cycle
        from cycler import cycler
        rcParams['axes.prop_cycle'] = cycler('color', simulated_colors)
        
        # Update the default colormap for CVD simulation
        # Get 256 colors from the colormap
        cmap = mpl.colormaps[default_cmap_name]
        colors = cmap(np.linspace(0, 1, 256))
        
        # Apply CVD simulation to each color
        for i in range(len(colors)):
            colors[i, :3] = simulate_colorblindness(colors[i, :3], self.deficiency_type)
        
        # Create a new colormap
        from matplotlib.colors import ListedColormap
        cvd_cmap = ListedColormap(colors, name=f"{default_cmap_name}_cvd")
        
        # Replace the default colormap
        mpl.colormaps[default_cmap_name] = cvd_cmap
        rcParams['image.cmap'] = default_cmap_name
    
    def _apply_accessibility_settings(self):
        """Apply general accessibility settings for better readability."""
        # Use a more accessible color cycle with higher contrast
        from cycler import cycler
        # High contrast color cycle that works well for most people
        # including those with color vision deficiencies
        accessible_colors = [
            '#0072B2',  # blue
            '#D55E00',  # vermillion
            '#009E73',  # green
            '#CC79A7',  # reddish purple
            '#F0E442',  # yellow
            '#56B4E9',  # sky blue
            '#E69F00',  # orange
            '#000000',  # black
        ]
        
        rcParams['axes.prop_cycle'] = cycler('color', accessible_colors)
        
        # Set the default colormap to viridis which is perceptually uniform
        # and works well for most people with color vision deficiencies
        rcParams['image.cmap'] = 'viridis'
        
        # Increase default line width for better visibility
        rcParams['lines.linewidth'] = 2.0
        
        # Increase marker size
        rcParams['lines.markersize'] = 8
        
        # Increase font size for better readability
        rcParams['font.size'] = 12
        rcParams['axes.labelsize'] = 'large'
        rcParams['axes.titlesize'] = 'x-large'
        
        # Improve contrast
        rcParams['axes.facecolor'] = 'white'
        rcParams['figure.facecolor'] = 'white'
        rcParams['savefig.facecolor'] = 'white'


def accessible_context(deficiency_type=None):
    """
    Create a context manager for temporarily enabling accessibility features.
    
    This context manager allows users to temporarily enable accessibility features
    for a specific block of code. When the context is entered, accessibility features
    are enabled, and when it is exited, the previous settings are restored.
    
    Parameters
    ----------
    deficiency_type : str or list of str, optional
        The types of color vision deficiencies to simulate. Can be 'protanopia',
        'deuteranopia', or None. Default is None, which means no simulation.
    
    Returns
    -------
    AccessibleContext
        A context manager that enables accessibility features.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> with plt.accessible_context():
    ...     # All figures created here will be accessible
    ...     fig = plt.figure()
    ...     # ...
    >>> # Figures created here will use default settings
    
    >>> # Simulate protanopia
    >>> with plt.accessible_context('protanopia'):
    ...     fig = plt.figure()
    ...     # ...
    """
    return AccessibleContext(deficiency_type)
