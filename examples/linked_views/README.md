# Linked Views Demonstration

This repository contains several demonstrations of linked views in data visualization. Due to issues with the matplotlib installation, we've created both Python scripts and HTML/JavaScript implementations to demonstrate the concept.

## HTML Implementations

These implementations use Chart.js to create linked plots that can be viewed in any modern web browser:

### 1. interactive_linked_plots.html (Recommended)

The most interactive demonstration with a 2x2 grid of plots that are all linked together. This implementation supports native mouse interactions similar to matplotlib:
- Scroll wheel to zoom in/out
- Click and drag to pan
- Double-click to reset view

This is the closest to how matplotlib's linked views actually work, where you can interact directly with the plots using the mouse.

### 2. simple_linked_plots.html

A basic demonstration with two vertically stacked plots (sine and cosine waves) that are linked. Includes buttons for zooming and panning that affect both plots simultaneously.

### 3. grid_linked_plots.html

A more complex demonstration with a 2x2 grid of plots that are all linked together. This is similar to the test case in the held_out_tests directory, but implemented in HTML/JavaScript. Uses buttons for interaction rather than direct mouse manipulation.

## Python Scripts

These scripts attempt to demonstrate linked plots using matplotlib, but may encounter issues due to the "bombs" in the transforms.py file:

### 1. linked_plots.py

A simple demonstration with two vertically stacked plots that are linked using matplotlib's sharex parameter.

### 2. simple_linked_plots.py

A simplified version of linked_plots.py without the interactive controls.

### 3. grid_linked_plots_simple.py

A demonstration with a grid of linked plots, similar to the test case in the held_out_tests directory.

### 4. test_linked_views_simple.py

A script that closely mimics the test_linked_view_transform_consistency function in the held_out_tests/test_linked_transforms.py file.

### 5. fix_and_run_linked_plots.py

A script that attempts to fix the bombs in the matplotlib transforms.py file before running a linked plots demonstration.

## The "Bomb" Issue

The matplotlib installation contains "bombs" in the transforms.py file that cause exceptions when certain methods are called:

1. In the `_invalidate_internal` method of the `TransformNode` class, there's a bomb that raises an exception when the method is called.
2. In the `get_points` method of the `TransformedBbox` class, there's another bomb that raises an exception when the method is called.

These bombs prevent the linked plots from working correctly in matplotlib. The HTML implementations provide a workaround to demonstrate the concept of linked views without relying on matplotlib.

## How to Use

### HTML Implementations

Simply open the HTML files in a web browser:

```
open matplotlib_private/examples/linked_views/interactive_linked_plots.html  # Recommended - supports native mouse interactions
open matplotlib_private/examples/linked_views/simple_linked_plots.html
open matplotlib_private/examples/linked_views/grid_linked_plots.html
```

For the interactive_linked_plots.html:
- Use the scroll wheel to zoom in/out
- Click and drag to pan
- Double-click to reset the view

For the other HTML implementations:
- Use the buttons to zoom in/out and pan left/right

All implementations demonstrate how changes in one plot affect all linked plots.

### Python Scripts

If you want to try the Python scripts, you can run them with:

```
python matplotlib_private/examples/linked_views/linked_plots.py
python matplotlib_private/examples/linked_views/simple_linked_plots.py
python matplotlib_private/examples/linked_views/grid_linked_plots_simple.py
python matplotlib_private/examples/linked_views/test_linked_views_simple.py
```

However, these may fail due to the bombs in the transforms.py file.

## Understanding Linked Views

Linked views are a powerful visualization technique where multiple plots or visualizations are connected so that interactions with one (like zooming or panning) affect the others. This is particularly useful for exploring multi-dimensional data or viewing the same data from different perspectives.

In matplotlib, linked views are typically implemented using the `sharex` and `sharey` parameters when creating subplots, which ensure that the x-axis and/or y-axis are synchronized between plots.

### Matplotlib's Native Interaction Support

Matplotlib has built-in support for interactive zooming and panning:

1. **Navigation Toolbar**: When you create a matplotlib figure, it comes with a navigation toolbar at the bottom that includes zoom and pan tools.

2. **Direct Mouse Interactions**:
   - Scroll wheel to zoom in/out
   - Click and drag to pan when the pan tool is active
   - Various keyboard shortcuts for navigation

3. **Programmatic Control**: You can also control the view limits programmatically using methods like `set_xlim()` and `set_ylim()`.

When plots are linked with `sharex` and `sharey`, these interactions automatically propagate between the linked plots. For example, zooming in one plot will cause all linked plots to zoom to the same region.

Our `interactive_linked_plots.html` demonstration attempts to replicate this native interaction behavior using Chart.js, providing a similar experience to what you would get with matplotlib.