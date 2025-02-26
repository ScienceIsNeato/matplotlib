import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import pytest

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
            # Convert data coordinates to display coordinates in first axes
            disp0 = axes[0].transData.transform(point)
            
            # Check that the same point transforms consistently in all axes
            for ax in axes[1:]:
                disp = ax.transData.transform(point)
                # Allow for minimal floating point differences
                assert np.allclose(disp, disp0, rtol=1e-10)
    
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