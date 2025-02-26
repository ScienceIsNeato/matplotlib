"""
A minimal demonstration of synchronized plots in matplotlib.

This script creates a figure with two subplots that are synchronized,
showing the basic functionality of linked views without any extra controls.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_simple_linked_plots():
    """
    Create a figure with two linked plots.
    
    The plots are linked so that panning and zooming in one plot
    affects the other plot as well.
    """
    # Create some sample data
    x = np.linspace(0, 10, 1000)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Plot the data
    ax1.plot(x, y1, 'b-', label='sin(x)')
    ax2.plot(x, y2, 'r-', label='cos(x)')
    
    # Add labels and legends
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Sine Wave')
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Cosine Wave')
    ax2.legend(loc='upper right')
    
    # Set initial view limits
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    
    # Add a message about interaction
    fig.suptitle('Pan and zoom with the mouse to see synchronized views', fontsize=12)
    
    # Adjust the layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    fig = create_simple_linked_plots()
    plt.show()