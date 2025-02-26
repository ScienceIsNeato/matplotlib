"""
A demonstration of a grid of synchronized plots in matplotlib.

This script creates a grid of subplots that are synchronized,
similar to the test case in the held_out_tests directory.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_grid_linked_plots(rows=2, cols=2):
    """
    Create a figure with a grid of linked plots.
    
    Parameters
    ----------
    rows : int, default: 2
        Number of rows in the grid.
    cols : int, default: 2
        Number of columns in the grid.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the grid of linked plots.
    """
    # Create some sample data
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8), sharex=True, sharey=True)
    
    # Make sure axes is a 2D array even if rows or cols is 1
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot the data in each subplot
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.plot(x, y, label=f'Plot ({i},{j})')
            ax.set_title(f'Plot ({i},{j})')
            ax.legend(loc='upper right')
    
    # Set labels for the outer axes
    for ax in axes[-1, :]:
        ax.set_xlabel('x')
    for ax in axes[:, 0]:
        ax.set_ylabel('y')
    
    # Set initial view limits
    axes[0, 0].set_xlim(0, 10)
    axes[0, 0].set_ylim(-1.1, 1.1)
    
    # Add a message about interaction
    fig.suptitle('Pan and zoom with the mouse to see synchronized views', fontsize=12)
    
    # Adjust the layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    fig = create_grid_linked_plots(2, 2)
    plt.show()