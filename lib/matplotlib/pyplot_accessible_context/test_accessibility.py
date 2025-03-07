"""
Test script for the accessibility features.

This script tests all aspects of the accessibility context managers and features.
It creates various plots to demonstrate the functionality and ensures proper integration.
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import unittest

# Define the test data
x = np.linspace(0, 10, 100)
y_values = [np.sin(x + i/2) for i in range(5)]


class AccessibilityFeatureTests(unittest.TestCase):
    """Test suite for accessibility features."""

    def setUp(self):
        """Set up for each test."""
        # Save original settings to restore later
        self.original_settings = {
            'axes.prop_cycle': matplotlib.rcParams['axes.prop_cycle'],
            'image.cmap': matplotlib.rcParams['image.cmap']
        }
        
    def tearDown(self):
        """Clean up after each test."""
        # Restore original settings
        for key, value in self.original_settings.items():
            matplotlib.rcParams[key] = value
        plt.close('all')
    
    def test_direct_import(self):
        """Test direct import of AccessibleContext."""
        from matplotlib.pyplot_accessible_context import AccessibleContext
        
        # Create a plot with default settings
        fig, ax = plt.subplots()
        for i, y in enumerate(y_values):
            ax.plot(x, y, label=f"Line {i+1}")
        ax.set_title("Before AccessibleContext")
        
        # Check that the default settings are unchanged
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_cmap = plt.rcParams['image.cmap']
        
        # Create a new plot with accessibility settings
        with AccessibleContext():
            fig2, ax2 = plt.subplots()
            for i, y in enumerate(y_values):
                ax2.plot(x, y, label=f"Line {i+1}")
            ax2.set_title("With AccessibleContext")
            
            # Check that the settings have changed
            accessible_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            accessible_cmap = plt.rcParams['image.cmap']
            
            self.assertNotEqual(default_colors, accessible_colors, 
                               "Color cycle should change with AccessibleContext")
            self.assertEqual(accessible_cmap, 'viridis', 
                            "Default colormap should be 'viridis' with AccessibleContext")
        
        # Check that the settings are restored
        restored_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        restored_cmap = plt.rcParams['image.cmap']
        
        self.assertEqual(default_colors, restored_colors, 
                       "Color cycle should be restored after context exit")
        self.assertEqual(default_cmap, restored_cmap, 
                        "Colormap should be restored after context exit")
    
    def test_accessible_context_function(self):
        """Test the accessible_context function."""
        from matplotlib.pyplot_accessible_context import accessible_context
        
        # Create a plot with default settings
        fig, ax = plt.subplots()
        for i, y in enumerate(y_values):
            ax.plot(x, y, label=f"Line {i+1}")
        ax.set_title("Before accessible_context")
        
        # Check that the default settings are unchanged
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_cmap = plt.rcParams['image.cmap']
        
        # Create a new plot with accessibility settings
        with accessible_context():
            fig2, ax2 = plt.subplots()
            for i, y in enumerate(y_values):
                ax2.plot(x, y, label=f"Line {i+1}")
            ax2.set_title("With accessible_context")
            
            # Check that the settings have changed
            accessible_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            accessible_cmap = plt.rcParams['image.cmap']
            
            self.assertNotEqual(default_colors, accessible_colors, 
                               "Color cycle should change with accessible_context")
            self.assertEqual(accessible_cmap, 'viridis', 
                            "Default colormap should be 'viridis' with accessible_context")
        
        # Check that the settings are restored
        restored_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        restored_cmap = plt.rcParams['image.cmap']
        
        self.assertEqual(default_colors, restored_colors, 
                       "Color cycle should be restored after context exit")
        self.assertEqual(default_cmap, restored_cmap, 
                        "Colormap should be restored after context exit")
    
    def test_pyplot_integration(self):
        """Test that the pyplot integration works."""
        # Check if accessible_context is available from pyplot
        self.assertTrue(hasattr(plt, 'accessible_context'), 
                       "plt.accessible_context should be available")
        
        # Create a plot with default settings
        fig, ax = plt.subplots()
        for i, y in enumerate(y_values):
            ax.plot(x, y, label=f"Line {i+1}")
        ax.set_title("Before plt.accessible_context")
        
        # Check that the default settings are unchanged
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        default_cmap = plt.rcParams['image.cmap']
        
        # Create a new plot with accessibility settings
        with plt.accessible_context():
            fig2, ax2 = plt.subplots()
            for i, y in enumerate(y_values):
                ax2.plot(x, y, label=f"Line {i+1}")
            ax2.set_title("With plt.accessible_context")
            
            # Check that the settings have changed
            accessible_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            accessible_cmap = plt.rcParams['image.cmap']
            
            self.assertNotEqual(default_colors, accessible_colors, 
                               "Color cycle should change with plt.accessible_context")
            self.assertEqual(accessible_cmap, 'viridis', 
                            "Default colormap should be 'viridis' with plt.accessible_context")
        
        # Check that the settings are restored
        restored_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        restored_cmap = plt.rcParams['image.cmap']
        
        self.assertEqual(default_colors, restored_colors, 
                       "Color cycle should be restored after context exit")
        self.assertEqual(default_cmap, restored_cmap, 
                        "Colormap should be restored after context exit")
    
    def test_cvd_simulation(self):
        """Test color vision deficiency simulation."""
        from matplotlib.pyplot_accessible_context import accessible_context
        from matplotlib.colors import to_rgb, to_hex
        
        # Create a sample color
        original_color = "#FF0000"  # pure red
        rgb_original = to_rgb(original_color)
        
        # Test protanopia simulation
        with accessible_context('protanopia'):
            # Check that the color cycle has changed
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            # Create a simple plot to test
            fig, ax = plt.subplots()
            ax.plot(x, np.sin(x), color=original_color)
            ax.set_title("Protanopia Simulation")
            
            # Check that the pure red has been transformed
            from matplotlib.colors import simulate_colorblindness
            simulated_rgb = simulate_colorblindness(rgb_original, 'protanopia')
            
            # The simulated color should not be the same as the original
            self.assertNotEqual(rgb_original, simulated_rgb, 
                              "Protanopia simulation should change the color")
    
    def test_multiple_plots(self):
        """Test that multiple plots in the same context work correctly."""
        from matplotlib.pyplot_accessible_context import accessible_context
        
        with accessible_context():
            # First plot
            fig1, ax1 = plt.subplots()
            for i, y in enumerate(y_values):
                ax1.plot(x, y, label=f"Line {i+1}")
            ax1.set_title("First plot with accessible_context")
            
            # Second plot
            fig2, ax2 = plt.subplots()
            for i, y in enumerate(y_values):
                ax2.plot(x, y, label=f"Line {i+1}")
            ax2.set_title("Second plot with accessible_context")
            
            # Both plots should have the same accessible colors
            colors1 = [line.get_color() for line in ax1.lines]
            colors2 = [line.get_color() for line in ax2.lines]
            
            self.assertEqual(colors1, colors2, 
                           "Colors should be consistent across plots in the same context")
    
    def test_nested_contexts(self):
        """Test that nested contexts work correctly."""
        from matplotlib.pyplot_accessible_context import accessible_context
        
        # Create a plot with default settings
        fig0, ax0 = plt.subplots()
        for i, y in enumerate(y_values):
            ax0.plot(x, y, label=f"Line {i+1}")
        ax0.set_title("Default settings")
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # First level context
        with accessible_context():
            fig1, ax1 = plt.subplots()
            for i, y in enumerate(y_values):
                ax1.plot(x, y, label=f"Line {i+1}")
            ax1.set_title("First level accessible context")
            first_level_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            # Nested context with CVD simulation
            with accessible_context('protanopia'):
                fig2, ax2 = plt.subplots()
                for i, y in enumerate(y_values):
                    ax2.plot(x, y, label=f"Line {i+1}")
                ax2.set_title("Nested context with protanopia")
                nested_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                
                # The nested colors should be different from the first level
                self.assertNotEqual(first_level_colors, nested_colors, 
                                  "Nested context should have different colors")
            
            # Back to first level, colors should be restored
            restored_first_level = plt.rcParams['axes.prop_cycle'].by_key()['color']
            self.assertEqual(first_level_colors, restored_first_level, 
                           "First level colors should be restored after nested context")
        
        # Back to default, colors should be restored
        final_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.assertEqual(default_colors, final_colors, 
                       "Default colors should be restored after all contexts")


def visual_test():
    """Create visual test plots to demonstrate the functionality."""
    from matplotlib.pyplot_accessible_context import accessible_context
    
    # Create a directory for the test outputs
    os.makedirs("accessibility_test_output", exist_ok=True)
    
    # Test 1: Default vs Accessible Color Cycle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default colors
    for i, y in enumerate(y_values):
        ax1.plot(x, y, label=f"Line {i+1}")
    ax1.set_title("Default Colors")
    ax1.legend()
    
    # Accessible colors
    with accessible_context():
        for i, y in enumerate(y_values):
            ax2.plot(x, y, label=f"Line {i+1}")
        ax2.set_title("Accessible Colors")
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig("accessibility_test_output/color_cycle_comparison.png")
    plt.close()
    
    # Test 2: Color Vision Deficiency Simulation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default view
    for i, y in enumerate(y_values):
        ax1.plot(x, y, label=f"Line {i+1}")
    ax1.set_title("Normal Vision")
    ax1.legend()
    
    # Protanopia simulation
    with accessible_context('protanopia'):
        for i, y in enumerate(y_values):
            ax2.plot(x, y, label=f"Line {i+1}")
        ax2.set_title("Protanopia Simulation")
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig("accessibility_test_output/protanopia_simulation.png")
    plt.close()
    
    # Test 3: Imshow with different colormaps
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = np.exp(-(X**2 + Y**2)/2)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Default view with jet colormap (not accessible)
    im1 = axes[0, 0].imshow(Z, cmap='jet')
    axes[0, 0].set_title("jet colormap (normal vision)")
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Default view with viridis colormap (accessible)
    im2 = axes[0, 1].imshow(Z, cmap='viridis')
    axes[0, 1].set_title("viridis colormap (normal vision)")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Protanopia simulation with jet
    with accessible_context('protanopia'):
        im3 = axes[1, 0].imshow(Z, cmap='jet')
        axes[1, 0].set_title("jet colormap (protanopia)")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Protanopia simulation with viridis
        im4 = axes[1, 1].imshow(Z, cmap='viridis')
        axes[1, 1].set_title("viridis colormap (protanopia)")
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig("accessibility_test_output/colormap_comparison.png")
    plt.close()
    
    print("Visual tests completed. Check the 'accessibility_test_output' directory for results.")


if __name__ == "__main__":
    # Run the automated tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run the visual tests
    visual_test()
    
    print("All tests completed.")
