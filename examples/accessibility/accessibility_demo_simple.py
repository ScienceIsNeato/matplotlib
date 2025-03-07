"""
=========================
Accessibility Demo Simple
=========================

A simple demonstration of accessible visualizations in Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import contextlib

@contextlib.contextmanager
def accessible_context(deficiency_type=None):
    """
    Context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str, optional
        Type of color vision deficiency to simulate. Options are:
        'protanopia', 'deuteranopia', 'tritanopia', or None.
        If None, no color vision deficiency is simulated.
    """
    # Store original settings
    original_settings = {}
    for key in plt.rcParams:
        if key.startswith('axes.') or key.startswith('lines.') or key.startswith('patch.') or key.startswith('font.'):
            original_settings[key] = plt.rcParams[key]
    
    try:
        # Apply accessibility settings
        _apply_accessibility_settings()
        
        # Apply color vision deficiency simulation if requested
        if deficiency_type:
            _apply_cvd_simulation(deficiency_type)
            
        yield
    finally:
        # Restore original settings
        for key, value in original_settings.items():
            plt.rcParams[key] = value

def _apply_accessibility_settings():
    """Apply general accessibility settings for better readability."""
    # High-contrast color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', ['#0000FF', '#FF0000', '#00AA00', '#AA00AA', '#FF7700', '#000000']
    )
    
    # Increase line widths and marker sizes
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8.0
    
    # Increase font sizes
    plt.rcParams['font.size'] = 14.0
    
    # Increase patch linewidth
    plt.rcParams['patch.linewidth'] = 2.0

def _apply_cvd_simulation(deficiency_type):
    """
    Apply color vision deficiency simulation.
    
    Parameters
    ----------
    deficiency_type : str
        Type of color vision deficiency to simulate.
        Options are: 'protanopia', 'deuteranopia', 'tritanopia'.
    """
    # Define color cycles for different types of color vision deficiency
    if deficiency_type.lower() == 'protanopia':
        # For protanopia (red-blind), avoid red-green distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#0072B2', '#F0E442', '#000000', '#56B4E9', '#CC79A7', '#D55E00']
        )
    elif deficiency_type.lower() == 'deuteranopia':
        # For deuteranopia (green-blind), avoid red-green distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#0072B2', '#F0E442', '#000000', '#56B4E9', '#CC79A7', '#D55E00']
        )
    elif deficiency_type.lower() == 'tritanopia':
        # For tritanopia (blue-blind), avoid blue-yellow distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#CC79A7', '#D55E00', '#000000', '#0072B2', '#F0E442', '#56B4E9']
        )

# Create data for the demo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)
y4 = np.sin(x) * np.sin(x)
y5 = np.cos(x) * np.cos(x)

# Create a figure with default settings
plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.plot(x, y3, label='sin(x)cos(x)')
plt.plot(x, y4, label='sin²(x)')
plt.plot(x, y5, label='cos²(x)')
plt.title('Default Settings')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('default_settings.png')
plt.close()

# Create a figure with accessible settings
with accessible_context():
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label='sin(x)')
    plt.plot(x, y2, label='cos(x)')
    plt.plot(x, y3, label='sin(x)cos(x)')
    plt.plot(x, y4, label='sin²(x)')
    plt.plot(x, y5, label='cos²(x)')
    plt.title('Accessible Settings')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('accessible_settings.png')
    plt.close()

# Create figures with color vision deficiency simulation
for deficiency_type in ['protanopia', 'deuteranopia', 'tritanopia']:
    with accessible_context(deficiency_type=deficiency_type):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y1, label='sin(x)')
        plt.plot(x, y2, label='cos(x)')
        plt.plot(x, y3, label='sin(x)cos(x)')
        plt.plot(x, y4, label='sin²(x)')
        plt.plot(x, y5, label='cos²(x)')
        plt.title(f'Optimized for {deficiency_type}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{deficiency_type}_settings.png')
        plt.close()

print("Demo completed. Check the output files:")
print("- default_settings.png")
print("- accessible_settings.png")
print("- protanopia_settings.png")
print("- deuteranopia_settings.png")
print("- tritanopia_settings.png") 