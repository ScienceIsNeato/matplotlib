"""
Accessibility context managers for matplotlib.

This module provides context managers for temporarily enabling accessibility features
in matplotlib, including color vision deficiency simulation and enhanced visibility settings.
"""

import contextlib
import matplotlib as mpl


class AccessibleContext(contextlib.AbstractContextManager):
    """
    Context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str, optional
        Type of color vision deficiency to optimize for.
        Options: 'protanopia', 'deuteranopia', 'tritanopia', None.
        If None, general high-contrast settings are used.
    """
    
    def __init__(self, deficiency_type=None):
        self.deficiency_type = deficiency_type
        self._original_params = {}
        
    def __enter__(self):
        # Store original parameters
        params_to_modify = [
            'axes.prop_cycle',
            'lines.linewidth',
            'lines.markersize',
            'font.size',
            'patch.linewidth'
        ]
        for param in params_to_modify:
            if param in mpl.rcParams:
                self._original_params[param] = mpl.rcParams[param]
        
        # Set accessible parameters
        if self.deficiency_type in ('protanopia', 'deuteranopia'):
            # Colors optimized for red-green color blindness
            colors = ['#0072B2', '#F0E442', '#000000', '#56B4E9', '#CC79A7', '#D55E00']
        elif self.deficiency_type == 'tritanopia':
            # Colors optimized for blue-yellow color blindness
            colors = ['#CC79A7', '#D55E00', '#000000', '#0072B2', '#F0E442', '#56B4E9']
        else:
            # Default high-contrast colors
            colors = ['#0000FF', '#FF0000', '#00AA00', '#AA00AA', '#FF7700', '#000000']
        
        import cycler
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler(color=colors)
        mpl.rcParams['lines.linewidth'] = 2.0
        mpl.rcParams['lines.markersize'] = 8.0
        mpl.rcParams['font.size'] = 14.0
        mpl.rcParams['patch.linewidth'] = 2.0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original parameters
        for param, value in self._original_params.items():
            mpl.rcParams[param] = value
        return False  # don't suppress exceptions


def accessible_context(deficiency_type=None):
    """
    Context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str, optional
        Type of color vision deficiency to optimize for.
        Options: 'protanopia', 'deuteranopia', 'tritanopia', None.
        If None, general high-contrast settings are used.
    
    Returns
    -------
    AccessibleContext
        A context manager that enables accessibility features.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> with plt.accessible_context():
    ...     plt.plot([0, 1], [0, 1])
    ...     plt.show()
    
    >>> with plt.accessible_context('protanopia'):
    ...     plt.plot([0, 1], [0, 1])
    ...     plt.show()
    """
    return AccessibleContext(deficiency_type) 