"""
Tests for the figure-level accessibility parameter.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pytest


def test_figure_accessible_parameter():
    """Test that the accessible parameter is properly set on the Figure."""
    # Test default value (should follow rcParams)
    original_rcparam = plt.rcParams.get('accessibility.enabled', False)
    
    # Temporarily set rcParam to True
    plt.rcParams['accessibility.enabled'] = True
    fig1 = plt.figure()
    assert fig1.accessible is True
    
    # Explicitly set to False
    fig2 = plt.figure(accessible=False)
    assert fig2.accessible is False
    
    # Explicitly set to True
    fig3 = plt.figure(accessible=True)
    assert fig3.accessible is True
    
    # Restore original setting
    plt.rcParams['accessibility.enabled'] = original_rcparam
    plt.close('all')


def test_figure_accessible_colorblind_simulation():
    """Test that the accessible parameter applies colorblind simulation."""
    # Create a figure with accessibility enabled
    fig = plt.figure(accessible=True)
    
    # Set colorblind simulation to deuteranopia
    original_simulation = plt.rcParams.get('accessibility.colorblind_simulation', 'none')
    plt.rcParams['accessibility.colorblind_simulation'] = 'deuteranopia'
    
    # Create a simple plot with red and green lines
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 100)
    red_line = ax.plot(x, np.sin(x), 'r-', linewidth=2)[0]
    green_line = ax.plot(x, np.cos(x), 'g-', linewidth=2)[0]
    
    # Force a draw to apply the accessibility transformations
    fig.canvas.draw()
    
    # Get the colors after drawing
    red_color_after = red_line.get_color()
    green_color_after = green_line.get_color()
    
    # The colors should be different from the original red and green
    # due to the deuteranopia simulation
    assert not np.array_equal(red_color_after, [1.0, 0.0, 0.0, 1.0])  # Original red
    assert not np.array_equal(green_color_after, [0.0, 1.0, 0.0, 1.0])  # Original green
    
    # Restore original setting
    plt.rcParams['accessibility.colorblind_simulation'] = original_simulation
    plt.close('all')


def test_figure_accessible_context():
    """Test that the accessible parameter works with the accessible_context."""
    # Create a figure with accessibility disabled
    fig = plt.figure(accessible=False)
    
    # Create a simple plot with red and green lines
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 100)
    red_line = ax.plot(x, np.sin(x), 'r-', linewidth=2)[0]
    green_line = ax.plot(x, np.cos(x), 'g-', linewidth=2)[0]
    
    # Force a draw to ensure the original colors are set
    fig.canvas.draw()
    
    # Get the original colors
    original_red = red_line.get_color()
    original_green = green_line.get_color()
    
    # Use the accessible_context to temporarily enable accessibility
    with plt.accessible_context('deuteranopia'):
        # The figure's accessible parameter should still be False
        assert fig.accessible is False
        
        # But the rcParam should be True within the context
        assert plt.rcParams['accessibility.enabled'] is True
        assert plt.rcParams['accessibility.colorblind_simulation'] == 'deuteranopia'
        
        # Force a draw to apply the accessibility transformations
        fig.canvas.draw()
        
        # Get the colors after drawing with accessibility
        context_red = red_line.get_color()
        context_green = green_line.get_color()
        
        # The colors should be different from the original red and green
        # due to the deuteranopia simulation
        assert not np.array_equal(context_red, original_red)
        assert not np.array_equal(context_green, original_green)
    
    # After the context, the rcParam should be restored
    assert plt.rcParams.get('accessibility.enabled', None) is False
    
    plt.close('all') 