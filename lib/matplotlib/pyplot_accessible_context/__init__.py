"""
Accessibility context managers for matplotlib.

This module provides context managers for temporarily enabling accessibility features
in matplotlib, including color vision deficiency simulation and enhanced visibility settings.
"""

from .pyplot_accessible_context import AccessibleContext, accessible_context

__all__ = ['AccessibleContext', 'accessible_context']
