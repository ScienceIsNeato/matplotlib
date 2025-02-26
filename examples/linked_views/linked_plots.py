"""
A simple demonstration of synchronized plots in matplotlib.

This script creates a figure with two subplots that are synchronized,
allowing you to experiment with linked views.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

def create_linked_plots():
    """
    Create a figure with two linked plots and interactive controls.
    
    The plots are linked so that panning and zooming in one plot
    affects the other plot as well.
    """
    # Create some sample data
    x = np.linspace(0, 10, 1000)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(10, 8))
    
    # Add subplots with a bit of space at the bottom for controls
    ax1 = fig.add_subplot(211)  # Top subplot
    ax2 = fig.add_subplot(212, sharex=ax1)  # Bottom subplot shares x-axis with top subplot
    
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
    
    # Adjust the layout to make room for the controls
    plt.subplots_adjust(bottom=0.25)
    
    # Add a reset button
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset View')
    
    def reset_view(event):
        ax1.set_xlim(0, 10)
        ax1.set_ylim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        plt.draw()
    
    reset_button.on_clicked(reset_view)
    
    # Add a slider to control the frequency
    freq_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
    freq_slider = Slider(
        ax=freq_ax,
        label='Frequency',
        valmin=0.1,
        valmax=5.0,
        valinit=1.0,
    )
    
    def update_freq(val):
        # Update the data with the new frequency
        freq = freq_slider.val
        y1_new = np.sin(freq * x)
        y2_new = np.cos(freq * x)
        
        # Update the plots
        ax1.lines[0].set_ydata(y1_new)
        ax2.lines[0].set_ydata(y2_new)
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    freq_slider.on_changed(update_freq)
    
    # Add a slider to control the phase
    phase_ax = plt.axes([0.25, 0.05, 0.5, 0.03])
    phase_slider = Slider(
        ax=phase_ax,
        label='Phase',
        valmin=0,
        valmax=2*np.pi,
        valinit=0,
    )
    
    def update_phase(val):
        # Update the data with the new phase
        phase = phase_slider.val
        freq = freq_slider.val
        y1_new = np.sin(freq * x + phase)
        y2_new = np.cos(freq * x + phase)
        
        # Update the plots
        ax1.lines[0].set_ydata(y1_new)
        ax2.lines[0].set_ydata(y2_new)
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    phase_slider.on_changed(update_phase)
    
    # Set initial view limits
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    
    # Add a message about interaction
    fig.text(0.5, 0.01, 'Pan and zoom with the mouse to see synchronized views',
             ha='center', va='center', fontsize=10)
    
    # Adjust the layout
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    
    return fig

if __name__ == "__main__":
    fig = create_linked_plots()
    plt.show()