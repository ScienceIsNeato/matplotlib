"""
Patch for Matplotlib Axes to use the SyncManager for linked views.

This module provides functions to patch the Matplotlib Axes class to use
the SyncManager for linked views, enabling complex view relationships
and efficient handling of large view counts.
"""

import functools
import weakref
from matplotlib.axes import Axes
from .sync_manager import SyncManager, LinkType


def _patched_set_xlim(original_set_xlim, self, *args, **kwargs):
    """
    Patched version of Axes.set_xlim that uses the SyncManager.
    
    This function wraps the original set_xlim method to propagate
    limit changes to linked views.
    """
    # Call the original method first
    result = original_set_xlim(self, *args, **kwargs)
    
    # Get the new limits
    xlim = self.get_xlim()
    
    # Propagate the changes to linked views
    sm = SyncManager.get_instance()
    sm.update_view(self, xlim=xlim)
    
    return result


def _patched_set_ylim(original_set_ylim, self, *args, **kwargs):
    """
    Patched version of Axes.set_ylim that uses the SyncManager.
    
    This function wraps the original set_ylim method to propagate
    limit changes to linked views.
    """
    # Call the original method first
    result = original_set_ylim(self, *args, **kwargs)
    
    # Get the new limits
    ylim = self.get_ylim()
    
    # Propagate the changes to linked views
    sm = SyncManager.get_instance()
    sm.update_view(self, ylim=ylim)
    
    return result


def _patched_sharex(original_sharex, self, other):
    """
    Patched version of Axes.sharex that uses the SyncManager.
    
    This function wraps the original sharex method to register
    the link with the SyncManager.
    """
    # Call the original method first
    result = original_sharex(self, other)
    
    # Register the link with the SyncManager
    sm = SyncManager.get_instance()
    
    # Create a group for this pair if it doesn't exist
    group_name = f"sharex_{id(other)}"
    try:
        sm.get_group(group_name)
    except ValueError:
        sm.create_group(group_name)
    
    # Add the views to the group
    sm.add_view_to_group(other, group_name)  # Parent
    sm.add_view_to_group(self, group_name, parent=other, link_type=LinkType.X_ONLY)
    
    return result


def _patched_sharey(original_sharey, self, other):
    """
    Patched version of Axes.sharey that uses the SyncManager.
    
    This function wraps the original sharey method to register
    the link with the SyncManager.
    """
    # Call the original method first
    result = original_sharey(self, other)
    
    # Register the link with the SyncManager
    sm = SyncManager.get_instance()
    
    # Create a group for this pair if it doesn't exist
    group_name = f"sharey_{id(other)}"
    try:
        sm.get_group(group_name)
    except ValueError:
        sm.create_group(group_name)
    
    # Add the views to the group
    sm.add_view_to_group(other, group_name)  # Parent
    sm.add_view_to_group(self, group_name, parent=other, link_type=LinkType.Y_ONLY)
    
    return result


def patch_axes():
    """
    Patch the Matplotlib Axes class to use the SyncManager.
    
    This function replaces the set_xlim, set_ylim, sharex, and sharey
    methods of the Axes class with patched versions that use the
    SyncManager for linked views.
    """
    # Save the original methods
    original_set_xlim = Axes.set_xlim
    original_set_ylim = Axes.set_ylim
    original_sharex = Axes.sharex
    original_sharey = Axes.sharey
    
    # Replace with patched versions
    Axes.set_xlim = functools.partialmethod(_patched_set_xlim, original_set_xlim)
    Axes.set_ylim = functools.partialmethod(_patched_set_ylim, original_set_ylim)
    Axes.sharex = functools.partialmethod(_patched_sharex, original_sharex)
    Axes.sharey = functools.partialmethod(_patched_sharey, original_sharey)