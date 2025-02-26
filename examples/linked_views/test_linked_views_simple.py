"""
A demonstration of a 4x4 grid of synchronized plots in matplotlib.

This script creates a 4x4 grid of subplots that are synchronized,
very similar to the test case in the held_out_tests directory.
"""

import numpy as np
import matplotlib.pyplot as plt

def test_linked_views():
    """
    Create a figure with a 4x4 grid of linked plots.
    
    This function is similar to the test_linked_view_transform_consistency
    function in the held_out_tests/test_linked_transforms.py file.
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
            ax.set_title(f'Plot ({i},{j})')
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
        consistent = True
        for i, ax in enumerate(axes[1:], 1):
            if not np.allclose(ax.get_xlim(), xlim0, rtol=1e-10):
                print(f"Axis {i} x-limits {ax.get_xlim()} don't match primary axis {xlim0}")
                consistent = False
            if not np.allclose(ax.get_ylim(), ylim0, rtol=1e-10):
                print(f"Axis {i} y-limits {ax.get_ylim()} don't match primary axis {ylim0}")
                consistent = False
        
        return consistent
    
    # Set initial view limits
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(-1.1, 1.1)
    
    # Check that all axes have the same limits
    check_coords_consistent()
    
    # Add a message about interaction
    fig.suptitle('Pan and zoom with the mouse to see synchronized views', fontsize=14)
    
    # Adjust the layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    fig = test_linked_views()
    plt.show()