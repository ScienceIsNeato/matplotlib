"""
Test script for the linked views implementation.

This script demonstrates the use of the sync manager for linked views.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import pytest

# Import and apply our patches
from matplotlib.axes_sync import patch_axes
patch_axes()

def test_basic_linked_views():
    """Test basic linked views with sharex and sharey."""
    print("Testing basic linked views...")
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(12, 12))
    
    # Create a 2x2 grid of axes
    ax1 = fig.add_subplot(2, 2, 1)
    print(f"Created ax1: {ax1}")
    ax2 = fig.add_subplot(2, 2, 2, sharex=ax1)  # Share x-axis with ax1
    print(f"Created ax2 with sharex=ax1: {ax2}")
    ax3 = fig.add_subplot(2, 2, 3, sharey=ax1)  # Share y-axis with ax1
    print(f"Created ax3 with sharey=ax1: {ax3}")
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)  # Share both axes with ax1
    print(f"Created ax4 with sharex=ax1, sharey=ax1: {ax4}")
    
    # Create test data - a simple sine wave
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    
    # Plot the data on all axes
    ax1.plot(x, y, 'b-', label='Sine wave')
    ax2.plot(x, y, 'r-', label='Sine wave')
    ax3.plot(x, y, 'g-', label='Sine wave')
    ax4.plot(x, y, 'k-', label='Sine wave')
    
    # Set titles
    ax1.set_title('Primary Axis')
    ax2.set_title('Shares X with Primary')
    ax3.set_title('Shares Y with Primary')
    ax4.set_title('Shares X and Y with Primary')
    
    # Add a legend to the first axis
    ax1.legend()
    
    # Set initial view limits on the primary axis
    print(f"Setting xlim on ax1 to (2, 4)")
    ax1.set_xlim(2, 4)
    print(f"Setting ylim on ax1 to (-0.8, 0.8)")
    ax1.set_ylim(-0.8, 0.8)
    
    # Print the limits of all axes to verify they're linked
    print(f"ax1 xlim: {ax1.get_xlim()}, ylim: {ax1.get_ylim()}")
    print(f"ax2 xlim: {ax2.get_xlim()}, ylim: {ax2.get_ylim()}")
    print(f"ax3 xlim: {ax3.get_xlim()}, ylim: {ax3.get_ylim()}")
    print(f"ax4 xlim: {ax4.get_xlim()}, ylim: {ax4.get_ylim()}")
    
    # Save the figure instead of showing it
    plt.tight_layout()
    plt.savefig('basic_linked_views.png')
    print("Saved figure to basic_linked_views.png")
    plt.close(fig)

def test_linked_view_transform_consistency():
    """
    Test that transform coordinates remain consistent across multiple linked views
    during rapid view changes. This test verifies that transform calculations
    remain accurate even with many views and frequent updates.
    """
    # Create a grid of linked views (4x4 = enough to trigger issues)
    fig = plt.figure(figsize=(12, 12))
    axes = []
    n_rows = n_cols = 4
    
    # Create test data - a simple sine wave
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    
    # Create grid of linked axes
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i*n_cols + j + 1)
            ax.plot(x, y)
            axes.append(ax)
    
    # Link all axes together
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    
    # Function to check coordinate consistency
    def check_coords_consistent():
        # Get the view limits of the first axis
        xlim0 = axes[0].get_xlim()
        ylim0 = axes[0].get_ylim()
        
        # Check that all axes have the same limits
        for ax in axes[1:]:
            assert np.allclose(ax.get_xlim(), xlim0, rtol=1e-10)
            assert np.allclose(ax.get_ylim(), ylim0, rtol=1e-10)
            
        # More importantly, check actual coordinate transforms
        test_points = [(1.0, 0.0), (5.0, 0.5), (7.5, -0.5)]
        for point in test_points:
            # Convert to display coordinates in first axes
            disp0 = axes[0].transData.transform(point)
            
            # Check that all axes transform consistently
            for ax in axes[1:]:
                # Transform in this axis's coordinate system
                disp = ax.transData.transform(point)
                # Convert back to data coordinates using this axis
                data = ax.transData.inverted().transform(disp)
                # Should match original point within reasonable tolerance
                assert np.allclose(data, point, rtol=1e-3)  # Allow small differences
    
    # Simulate rapid view changes and check consistency
    for _ in range(5):
        # Zoom in to different regions
        axes[0].set_xlim(2, 4)
        check_coords_consistent()
        
        axes[0].set_xlim(6, 8)
        check_coords_consistent()
        
        # Pan around
        axes[0].set_xlim(3, 5)
        axes[0].set_ylim(-0.8, 0.8)
        check_coords_consistent()
        
        # Rapid successive updates
        for x0 in np.linspace(0, 8, 10):
            axes[0].set_xlim(x0, x0 + 2)
            check_coords_consistent()
    
    plt.close(fig)

if __name__ == "__main__":
    test_basic_linked_views()
    test_linked_view_transform_consistency()
    print("All tests passed!")