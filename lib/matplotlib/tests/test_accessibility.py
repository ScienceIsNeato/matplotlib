import unittest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import contextlib

@contextlib.contextmanager
def accessible_context(deficiency_type=None):
    """
    Context manager for temporarily enabling accessibility features.
    
    Parameters
    ----------
    deficiency_type : str, optional
        Type of color vision deficiency to simulate. Options are:
        'protanopia', 'deuteranopia', 'tritanopia', or None.
        If None, no color vision deficiency is simulated.
    """
    # Store original settings
    original_settings = {}
    for key in plt.rcParams:
        if key.startswith('axes.') or key.startswith('lines.') or key.startswith('patch.') or key.startswith('font.'):
            original_settings[key] = plt.rcParams[key]
    
    try:
        # Apply accessibility settings
        _apply_accessibility_settings()
        
        # Apply color vision deficiency simulation if requested
        if deficiency_type:
            _apply_cvd_simulation(deficiency_type)
            
        yield
    finally:
        # Restore original settings
        for key, value in original_settings.items():
            plt.rcParams[key] = value

def _apply_accessibility_settings():
    """Apply general accessibility settings for better readability."""
    # High-contrast color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', ['#0000FF', '#FF0000', '#00AA00', '#AA00AA', '#FF7700', '#000000']
    )
    
    # Increase line widths and marker sizes
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8.0
    
    # Increase font sizes
    plt.rcParams['font.size'] = 14.0
    
    # Increase patch linewidth
    plt.rcParams['patch.linewidth'] = 2.0

def _apply_cvd_simulation(deficiency_type):
    """
    Apply color vision deficiency simulation.
    
    Parameters
    ----------
    deficiency_type : str
        Type of color vision deficiency to simulate.
        Options are: 'protanopia', 'deuteranopia', 'tritanopia'.
    """
    # Define color cycles for different types of color vision deficiency
    if deficiency_type.lower() == 'protanopia':
        # For protanopia (red-blind), avoid red-green distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#0072B2', '#F0E442', '#000000', '#56B4E9', '#CC79A7', '#D55E00']
        )
    elif deficiency_type.lower() == 'deuteranopia':
        # For deuteranopia (green-blind), avoid red-green distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#0072B2', '#F0E442', '#000000', '#56B4E9', '#CC79A7', '#D55E00']
        )
    elif deficiency_type.lower() == 'tritanopia':
        # For tritanopia (blue-blind), avoid blue-yellow distinctions
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            'color', ['#CC79A7', '#D55E00', '#000000', '#0072B2', '#F0E442', '#56B4E9']
        )

class TestAccessibility(unittest.TestCase):
    def test_accessible_context(self):
        """Test that the accessible_context context manager works correctly."""
        # Store original settings
        original_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        original_linewidth = mpl.rcParams['lines.linewidth']
        
        # Use the context manager
        with accessible_context():
            # Check that settings have changed
            accessible_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
            accessible_linewidth = mpl.rcParams['lines.linewidth']
            
            self.assertNotEqual(accessible_colors, original_colors)
            self.assertGreater(accessible_linewidth, original_linewidth)
        
        # Check that settings are restored
        restored_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        restored_linewidth = mpl.rcParams['lines.linewidth']
        
        self.assertEqual(restored_colors, original_colors)
        self.assertEqual(restored_linewidth, original_linewidth)
    
    def test_accessible_context_with_deficiency_type(self):
        """Test that the accessible_context context manager works with deficiency_type."""
        # Store original settings
        original_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Use the context manager with protanopia
        with accessible_context(deficiency_type='protanopia'):
            # Check that settings have changed
            protanopia_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
            
            self.assertNotEqual(protanopia_colors, original_colors)
        
        # Check that settings are restored
        restored_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.assertEqual(restored_colors, original_colors)
    
    def test_accessible_context_drawing(self):
        """Test that the accessible_context context manager works with drawing."""
        # Create a figure with default settings
        fig1, ax1 = plt.subplots()
        for i in range(3):
            ax1.plot([0, 1], [i, i+1], label=f'Line {i+1}')
        default_lines = ax1.get_lines()
        default_colors = [line.get_color() for line in default_lines]
        default_linewidth = default_lines[0].get_linewidth()
        plt.close(fig1)
        
        # Create a figure with accessible settings
        with accessible_context():
            fig2, ax2 = plt.subplots()
            for i in range(3):
                ax2.plot([0, 1], [i, i+1], label=f'Line {i+1}')
            accessible_lines = ax2.get_lines()
            accessible_colors = [line.get_color() for line in accessible_lines]
            accessible_linewidth = accessible_lines[0].get_linewidth()
            plt.close(fig2)
        
        # Check that the colors and linewidth are different
        self.assertNotEqual(accessible_colors, default_colors)
        self.assertGreater(accessible_linewidth, default_linewidth)
        
    def test_pyplot_integration(self):
        """Test that the pyplot integration works."""
        # This test is expected to fail until the pyplot integration is implemented
        try:
            # Check if accessible_context is available from pyplot
            if hasattr(plt, 'accessible_context'):
                print("plt.accessible_context is available")
                
                # Create a plot with default settings
                fig, ax = plt.subplots()
                for i in range(3):
                    ax.plot([0, 1], [i, i+1], label=f'Line {i+1}')
                ax.set_title("Before plt.accessible_context")
                
                # Check the default settings
                default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                default_cmap = plt.rcParams['image.cmap']
                
                # Create a new plot with accessibility settings
                with plt.accessible_context():
                    fig2, ax2 = plt.subplots()
                    for i in range(3):
                        ax2.plot([0, 1], [i, i+1], label=f'Line {i+1}')
                    ax2.set_title("With plt.accessible_context")
                    
                    # Check that the settings have changed
                    accessible_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    accessible_cmap = plt.rcParams['image.cmap']
                    
                    self.assertNotEqual(accessible_colors, default_colors)
                
                # Check that the settings are restored
                restored_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                restored_cmap = plt.rcParams['image.cmap']
                
                self.assertEqual(default_colors, restored_colors)
                self.assertEqual(default_cmap, restored_cmap)
            else:
                self.skipTest("plt.accessible_context is not available yet")
        except (ImportError, AttributeError):
            self.skipTest("pyplot integration is not implemented yet")

if __name__ == '__main__':
    unittest.main() 