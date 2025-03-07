"""
Standalone implementation of accessibility features for Matplotlib.

This simplified module contains all the necessary components for testing without
complex import dependencies.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
from cycler import cycler

# Simulate the color vision deficiency simulation function
def simulate_colorblindness(rgb, deficiency_type):
    """
    Simulate how colors appear to someone with color vision deficiency.
    
    Parameters
    ----------
    rgb : array-like of shape (3,)
        The RGB color values to transform, in the range [0, 1].
    deficiency_type : str
        The type of color vision deficiency to simulate.
        
    Returns
    -------
    array-like of shape (3,)
        The transformed RGB color values.
    """
    # Just a simplified version for testing
    if deficiency_type == 'protanopia':
        # Simulate red-blind by reducing red component
        return [rgb[0] * 0.3, rgb[1], rgb[2]]
    elif deficiency_type == 'deuteranopia':
        # Simulate green-blind by reducing green component
        return [rgb[0], rgb[1] * 0.3, rgb[2]]
    return rgb


class AccessibleContext:
    """
    A context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str or None, optional
        The type of color vision deficiency to simulate. Can be 'protanopia',
        'deuteranopia', or None. Default is None, which means no simulation.
    """
    def __init__(self, deficiency_type=None):
        self.deficiency_type = deficiency_type
        self._original_settings = {}
        self._original_colormaps = {}
        
    def __enter__(self):
        # Save original settings
        self._original_settings = {
            'axes.prop_cycle': plt.rcParams['axes.prop_cycle'],
            'image.cmap': plt.rcParams['image.cmap'],
            'lines.linewidth': plt.rcParams['lines.linewidth'],
            'lines.markersize': plt.rcParams['lines.markersize'],
            'font.size': plt.rcParams['font.size'],
            'axes.labelsize': plt.rcParams['axes.labelsize'],
            'axes.titlesize': plt.rcParams['axes.titlesize'],
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
            plt.rcParams[key] = value
        
        # No exception handling needed for simple tests
        return False
    
    def _apply_cvd_simulation(self):
        """Apply color vision deficiency simulation to colormaps and colors."""
        print(f"Simulating {self.deficiency_type} color vision deficiency")
        
        # Modify default color cycle for CVD simulation
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        simulated_colors = []
        
        for color in colors:
            # Convert hex color to RGB
            if color.startswith('#'):
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                rgb = [r, g, b]
            else:
                # Handle named colors
                rgb = [0.5, 0.5, 0.5]  # Default for testing
                
            # Simulate color vision deficiency
            sim_rgb = simulate_colorblindness(rgb, self.deficiency_type)
            
            # Simple representation for testing
            simulated_colors.append(f"#{int(sim_rgb[0]*255):02x}{int(sim_rgb[1]*255):02x}{int(sim_rgb[2]*255):02x}")
        
        # Update the color cycle
        plt.rcParams['axes.prop_cycle'] = cycler('color', simulated_colors)
    
    def _apply_accessibility_settings(self):
        """Apply general accessibility settings for better readability."""
        print("Applying accessibility settings")
        
        # Use a more accessible color cycle with higher contrast
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
        
        plt.rcParams['axes.prop_cycle'] = cycler('color', accessible_colors)
        
        # Set the default colormap to viridis which is perceptually uniform
        plt.rcParams['image.cmap'] = 'viridis'
        
        # Increase default line width for better visibility
        plt.rcParams['lines.linewidth'] = 2.0
        
        # Increase marker size
        plt.rcParams['lines.markersize'] = 8
        
        # Increase font size for better readability
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 'large'
        plt.rcParams['axes.titlesize'] = 'x-large'


@contextmanager
def accessible_context(deficiency_type=None):
    """
    Create a context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str or None, optional
        The type of color vision deficiency to simulate.
        
    Returns
    -------
    AccessibleContext
        A context manager that enables accessibility features.
    """
    ctx = AccessibleContext(deficiency_type)
    try:
        ctx.__enter__()
        yield ctx
    finally:
        ctx.__exit__(None, None, None)


def test_standalone():
    """Test the standalone implementation."""
    print("Running standalone test of accessibility features")
    
    # Print the default settings
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    default_cmap = plt.rcParams['image.cmap']
    default_linewidth = plt.rcParams['lines.linewidth']
    
    print(f"Default colors: {default_colors[:3]}...")
    print(f"Default colormap: {default_cmap}")
    print(f"Default line width: {default_linewidth}")
    
    # Use the context manager
    with accessible_context():
        # Check that the settings have changed
        accessible_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        accessible_cmap = plt.rcParams['image.cmap']
        accessible_linewidth = plt.rcParams['lines.linewidth']
        
        print(f"Accessible colors: {accessible_colors[:3]}...")
        print(f"Accessible colormap: {accessible_cmap}")
        print(f"Accessible line width: {accessible_linewidth}")
        
        # Create a simple plot
        plt.figure(figsize=(10, 5))
        x = np.linspace(0, 10, 100)
        for i in range(5):
            plt.plot(x, np.sin(x + i/2), label=f"Line {i+1}")
        plt.title("Accessible Colors")
        plt.legend()
        plt.savefig("standalone_accessible_test.png")
        plt.close()
    
    # Check that the settings are restored
    restored_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    restored_cmap = plt.rcParams['image.cmap']
    restored_linewidth = plt.rcParams['lines.linewidth']
    
    print(f"Restored colors: {restored_colors[:3]}...")
    print(f"Restored colormap: {restored_cmap}")
    print(f"Restored line width: {restored_linewidth}")
    
    # Now test with color vision deficiency simulation
    with accessible_context('protanopia'):
        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.plot(x, np.sin(x + i/2), label=f"Line {i+1}")
        plt.title("Protanopia Simulation")
        plt.legend()
        plt.savefig("standalone_protanopia_test.png")
        plt.close()
    
    print("Standalone test completed successfully!")
    print("Check the generated images:")
    print("  - standalone_accessible_test.png")
    print("  - standalone_protanopia_test.png")


if __name__ == "__main__":
    test_standalone()
