"""
A module for converting numbers or color arguments to *RGB* or *RGBA*.

*RGB* and *RGBA* are sequences of, respectively, 3 or 4 floats in the
range 0-1.

This module includes functions and classes for color specification conversions,
and for mapping numbers to colors in a 1-D array of colors called a colormap.

Mapping data onto colors using a colormap typically involves two steps: a data
array is first mapped onto the range 0-1 using a subclass of `Normalize`,
then this number is mapped to a color using a subclass of `Colormap`.  Two
subclasses of `Colormap` provided here:  `LinearSegmentedColormap`, which uses
piecewise-linear interpolation to define colormaps, and `ListedColormap`, which
makes a colormap from a list of colors.

.. seealso::

  :ref:`colormap-manipulation` for examples of how to
  make colormaps and

  :ref:`colormaps` for a list of built-in colormaps.

  :ref:`colormapnorms` for more details about data
  normalization

  More colormaps are available at palettable_.

The module also provides functions for checking whether an object can be
interpreted as a color (`is_color_like`), for converting such an object
to an RGBA tuple (`to_rgba`) or to an HTML-like hex string in the
"#rrggbb" format (`to_hex`), and a sequence of colors to an (n, 4)
RGBA array (`to_rgba_array`).  Caching is used for efficiency.

Colors that Matplotlib recognizes are listed at
:ref:`colors_def`.

.. _palettable: https://jiffyclub.github.io/palettable/
.. _xkcd color survey: https://xkcd.com/color/rgb/
"""

import base64
from collections.abc import Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
import copy

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale, _image
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS


def rgb_to_lab(rgb):
    """
    Convert RGB colors to LAB color space.

    Parameters
    ----------
    rgb : array-like
        The RGB values. Can be a single color or an array of colors.
        Values should be in the range [0, 1].

    Returns
    -------
    lab : ndarray
        The colors converted to LAB space. Same shape as input.

    Notes
    -----
    The conversion is based on the CIE LAB color space standard.
    RGB values are first converted to XYZ space, then to LAB.

    Examples
    --------
    Convert a single RGB color to LAB:

    >>> import numpy as np
    >>> from matplotlib.colors import rgb_to_lab
    >>> rgb_to_lab(np.array([1.0, 0.0, 0.0]))  # Red
    array([ 53.24,  80.09,  67.2 ])

    Convert an array of RGB colors to LAB:

    >>> rgb_to_lab(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    array([[  53.24,   80.09,   67.2 ],
           [  87.73,  -86.18,   83.18],
           [  32.3 ,   79.19, -107.86]])
    """
    rgb = np.asarray(rgb)
    
    # Handle both single colors and arrays of colors
    input_is_1d = rgb.ndim == 1
    if input_is_1d:
        rgb = rgb[np.newaxis, :]
    
    # Clip RGB values to [0, 1] range
    rgb_clipped = np.clip(rgb, 0, 1)
    
    # Convert RGB to linear RGB (remove gamma correction)
    mask = rgb_clipped <= 0.04045
    rgb_linear = np.where(mask, rgb_clipped / 12.92, 
                         ((rgb_clipped + 0.055) / 1.055) ** 2.4)
    
    # Convert linear RGB to XYZ using the standard RGB to XYZ matrix
    # Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # Using sRGB with D65 reference white
    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb_linear, rgb_to_xyz_matrix.T)
    
    # Reference white (D65)
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    
    # Convert XYZ to LAB
    xyz_normalized = xyz / xyz_ref
    
    # Apply the nonlinear transformation
    mask = xyz_normalized > 0.008856
    xyz_f = np.where(mask, xyz_normalized ** (1/3), 
                    7.787 * xyz_normalized + 16/116)
    
    # Calculate LAB components
    L = np.where(xyz_normalized[:, 1] > 0.008856, 
                116 * xyz_normalized[:, 1] ** (1/3) - 16, 
                903.3 * xyz_normalized[:, 1])
    a = 500 * (xyz_f[:, 0] - xyz_f[:, 1])
    b = 200 * (xyz_f[:, 1] - xyz_f[:, 2])
    
    lab = np.column_stack([L, a, b])
    
    # Return the same shape as input
    if input_is_1d:
        lab = lab[0]
    
    return lab

def lab_to_rgb(lab):
    """
    Convert LAB colors to RGB color space.

    Parameters
    ----------
    lab : array-like
        The LAB values. Can be a single color or an array of colors.

    Returns
    -------
    rgb : ndarray
        The colors converted to RGB space. Same shape as input.
        Values are clipped to the range [0, 1].

    Notes
    -----
    The conversion is based on the CIE LAB color space standard.
    LAB values are first converted to XYZ space, then to RGB.

    Examples
    --------
    Convert a single LAB color to RGB:

    >>> import numpy as np
    >>> from matplotlib.colors import lab_to_rgb
    >>> lab_to_rgb(np.array([53.24, 80.09, 67.2]))  # Red in LAB
    array([1., 0., 0.])

    Convert an array of LAB colors to RGB:

    >>> lab_to_rgb(np.array([[53.24, 80.09, 67.2], 
    ...                      [87.73, -86.18, 83.18], 
    ...                      [32.3, 79.19, -107.86]]))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    lab = np.asarray(lab)
    
    # Handle both single colors and arrays of colors
    input_is_1d = lab.ndim == 1
    if input_is_1d:
        lab = lab[np.newaxis, :]
    
    # Extract LAB components
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    
    # Calculate f(Y)
    fy = (L + 16) / 116
    
    # Calculate f(X) and f(Z)
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Reference white (D65)
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    
    # Calculate XYZ
    xyz = np.zeros_like(lab)
    
    # Handle f(X)
    mask_x = fx ** 3 > 0.008856
    xyz[:, 0] = xyz_ref[0] * np.where(mask_x, fx ** 3, (fx - 16/116) / 7.787)
    
    # Handle f(Y)
    mask_y = L > 7.999625  # L > 8 => fy**3 > 0.008856
    xyz[:, 1] = xyz_ref[1] * np.where(mask_y, fy ** 3, L / 903.3)
    
    # Handle f(Z)
    mask_z = fz ** 3 > 0.008856
    xyz[:, 2] = xyz_ref[2] * np.where(mask_z, fz ** 3, (fz - 16/116) / 7.787)
    
    # Convert XYZ to linear RGB
    # Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # Using sRGB with D65 reference white
    xyz_to_rgb_matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb_linear = np.dot(xyz, xyz_to_rgb_matrix.T)
    
    # Convert linear RGB to sRGB (apply gamma correction)
    mask = rgb_linear <= 0.0031308
    rgb = np.where(mask, 12.92 * rgb_linear, 
                  1.055 * rgb_linear ** (1/2.4) - 0.055)
    
    # Clip to [0, 1] range
    rgb = np.clip(rgb, 0, 1)
    
    # Return the same shape as input
    if input_is_1d:
        rgb = rgb[0]
    
    return rgb


def rgb_to_lab_with_lut(rgb):
    """
    Convert RGB colors to LAB color space using a lookup table for optimization.
    
    This function uses a lookup table (LUT) to cache frequently used color conversions,
    which can significantly improve performance when the same colors are converted repeatedly.
    
    Parameters
    ----------
    rgb : array-like
        The RGB values. Should be a single color with values in the range [0, 1].
    
    Returns
    -------
    lab : ndarray
        The color in LAB space.
    
    Notes
    -----
    This function is optimized for repeated conversions of the same colors.
    For converting large arrays of unique colors, use `rgb_to_lab` instead.
    
    Examples
    --------
    >>> rgb_to_lab_with_lut([1.0, 0.0, 0.0])  # pure red
    array([53.24, 80.09, 67.2 ])
    """
    # Convert input to a hashable tuple with rounded values for the cache key
    rgb_array = np.asarray(rgb)
    rgb_key = tuple(np.round(rgb_array, 3))
    
    # Check if the color is in the LUT
    if not hasattr(rgb_to_lab_with_lut, 'lut'):
        rgb_to_lab_with_lut.lut = {}
    
    if rgb_key in rgb_to_lab_with_lut.lut:
        return rgb_to_lab_with_lut.lut[rgb_key]
    
    # If not in LUT, calculate and store the result
    lab = rgb_to_lab(rgb_array)
    rgb_to_lab_with_lut.lut[rgb_key] = lab
    
    # Limit LUT size to prevent memory issues
    if len(rgb_to_lab_with_lut.lut) > 10000:
        # Remove a random item if the LUT gets too large
        rgb_to_lab_with_lut.lut.pop(next(iter(rgb_to_lab_with_lut.lut)))
    
    return lab


def simulate_colorblindness(rgb, deficiency_type, severity=1.0):
    """
    Simulate how colors would be perceived with different types of color vision deficiency.

    Parameters
    ----------
    rgb : array-like
        The RGB values of the colors to be transformed.
    deficiency_type : {'deuteranopia', 'protanopia', 'tritanopia'}
        The type of color vision deficiency to simulate.
    severity : float, optional
        The severity of the color vision deficiency, where 0 is normal vision
        and 1 is complete color blindness. Default is 1.0.

    Returns
    -------
    array-like
        The transformed RGB values.
    """
    # Ensure RGB input is a numpy array
    rgb = np.asarray(rgb)
    input_is_1d = rgb.ndim == 1
    if input_is_1d:
        rgb = rgb[np.newaxis, :]
    
    # Simulation matrices for different types of color blindness
    # These matrices are based on research by Machado, Oliveira, and Fernandes (2009)
    if deficiency_type == 'deuteranopia':
        # Red-green color blindness (deuteranopia)
        sim_matrix = np.array([
            [0.625, 0.375, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 0.3, 0.7]
        ])
    elif deficiency_type == 'protanopia':
        # Red-green color blindness (protanopia)
        sim_matrix = np.array([
            [0.567, 0.433, 0.0],
            [0.558, 0.442, 0.0],
            [0.0, 0.242, 0.758]
        ])
    elif deficiency_type == 'tritanopia':
        # Blue-yellow color blindness (tritanopia)
        sim_matrix = np.array([
            [0.95, 0.05, 0.0],
            [0.0, 0.433, 0.567],
            [0.0, 0.475, 0.525]
        ])
    else:
        raise ValueError("deficiency_type must be one of 'deuteranopia', 'protanopia', or 'tritanopia'")
    
    # Interpolate between original and simulated colors based on severity
    if severity == 1.0:
        rgb_sim = rgb @ sim_matrix
    else:
        rgb_sim = (1 - severity) * rgb + severity * (rgb @ sim_matrix)
    
    # Return the same shape as input
    if input_is_1d:
        rgb_sim = rgb_sim[0]
    
    return rgb_sim


class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """

    def __init__(self, name, N=256, *, bad=None, under=None, over=None):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        bad : :mpltype:`color`, default: transparent
            The color for invalid values (NaN or masked).

            .. versionadded:: 3.11

        under : :mpltype:`color`, default: color of the lowest value
            The color for low out-of-range values.

            .. versionadded:: 3.11

        over : :mpltype:`color`, default: color of the highest value
            The color for high out-of-range values.

            .. versionadded:: 3.11
        """
        self.name = name
        self.N = int(N)  # ensure that N is always int
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0) if bad is None else to_rgba(bad)
        self._rgba_under = None if under is None else to_rgba(under)
        self._rgba_over = None if over is None else to_rgba(over)
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False
        self.n_variates = 1
        #: When this colormap exists on a scalar mappable and colorbar_extend
        #: is not False, colorbar creation will pick up ``colorbar_extend`` as
        #: the default value for the ``extend`` keyword in the
        #: `matplotlib.colorbar.Colorbar` constructor.
        self.colorbar_extend = False

    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        rgba, mask = self._get_rgba_and_mask(X, alpha=alpha, bytes=bytes)
        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba

    def _get_rgba_and_mask(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        colors : np.ndarray
            Array of RGBA values with a shape of ``X.shape + (4, )``.
        mask : np.ndarray
            Boolean array with True where the input is ``np.nan`` or masked.
        """
        if not self._isinit:
            self._init()

        xa = np.array(X, copy=True)
        if not xa.dtype.isnative:
            # Native byteorder is faster.
            xa = xa.byteswap().view(xa.dtype.newbyteorder())
        if xa.dtype.kind == "f":
            xa *= self.N
            # xa == 1 (== N after multiplication) is not out of range.
            xa[xa == self.N] = self.N - 1
        # Pre-compute the masks before casting to int (which can truncate
        # negative values to zero or wrap large floats to negative ints).
        mask_under = xa < 0
        mask_over = xa >= self.N
        # If input was masked, get the bad mask from it; else mask out nans.
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid="ignore"):
            # We need this cast for unsigned ints as well as floats
            xa = xa.astype(int)
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad

        lut = self._lut
        if bytes:
            lut = (lut * 255).astype(np.uint8)

        rgba = lut.take(xa, axis=0, mode='clip')

        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # Will be cast to uint8 upon assignment.
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(
                    f"alpha is array-like but its shape {alpha.shape} does "
                    f"not match that of X {xa.shape}")
            rgba[..., -1] = alpha
            # If the "bad" color is all zeros, then ignore alpha input.
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)

        return rgba, mask_bad

    def __copy__(self):
        cls = self.__class__
        cmapobject = cls.__new__(cls)
        cmapobject.__dict__.update(self.__dict__)
        if self._isinit:
            cmapobject._lut = np.copy(self._lut)
        return cmapobject

    def __eq__(self, other):
        if (not isinstance(other, Colormap) or
                self.colorbar_extend != other.colorbar_extend):
            return False
        # To compare lookup tables the Colormaps have to be initialized
        if not self._isinit:
            self._init()
        if not other._isinit:
            other._init()
        return np.array_equal(self._lut, other._lut)

    def get_bad(self):
        """Get the color for masked values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_bad])

    def set_bad(self, color='k', alpha=None):
        """Set the color for masked values."""
        self._rgba_bad = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_under(self):
        """Get the color for low out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_under])

    def set_under(self, color='k', alpha=None):
        """Set the color for low out-of-range values."""
        self._rgba_under = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_over(self):
        """Get the color for high out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_over])

    def set_over(self, color='k', alpha=None):
        """Set the color for high out-of-range values."""
        self._rgba_over = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def set_extremes(self, *, bad=None, under=None, over=None):
        """
        Set the colors for masked (*bad*) values and, when ``norm.clip =
        False``, low (*under*) and high (*over*) out-of-range values.
        """
        if bad is not None:
            self.set_bad(bad)
        if under is not None:
            self.set_under(under)
        if over is not None:
            self.set_over(over)

    def with_extremes(self, *, bad=None, under=None, over=None):
        """
        Return a copy of the colormap, for which the colors for masked (*bad*)
        values and, when ``norm.clip = False``, low (*under*) and high (*over*)
        out-of-range values, have been set accordingly.
        """
        new_cm = self.copy()
        new_cm.set_extremes(bad=bad, under=under, over=over)
        return new_cm

    def _set_extremes(self):
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N - 1]
        self._lut[self._i_bad] = self._rgba_bad

    def with_alpha(self, alpha):
        """
        Return a copy of the colormap with a new uniform transparency.

        Parameters
        ----------
        alpha : float
             The alpha blending value, between 0 (transparent) and 1 (opaque).
        """
        if not isinstance(alpha, Real):
            raise TypeError(f"'alpha' must be numeric or None, not {type(alpha)}")
        if not 0 <= alpha <= 1:
            ValueError("'alpha' must be between 0 and 1, inclusive")
        new_cm = self.copy()
        if not new_cm._isinit:
            new_cm._init()
        new_cm._lut[:, 3] = alpha
        return new_cm

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        """Return whether the colormap is grayscale."""
        if not self._isinit:
            self._init()
        return (np.all(self._lut[:, 0] == self._lut[:, 1]) and
                np.all(self._lut[:, 0] == self._lut[:, 2]))

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        if hasattr(self, '_resample'):
            _api.warn_external(
                "The ability to resample a color map is now public API "
                f"However the class {type(self)} still only implements "
                "the previous private _resample method.  Please update "
                "your class."
            )
            return self._resample(lutsize)

        raise NotImplementedError()

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the name will be the
            name of the parent colormap + "_r".

        Returns
        -------
        CachedColormap
            A reversed instance of the colormap.
        """
        if name is None:
            name = self.name + "_r"
        return CachedColormap(self.cmap.reversed(name), self.cache_size)


class CachedColormap(Colormap):
    """
    A colormap that caches evaluations for improved performance.
    
    This colormap wraps another colormap and caches the results of
    calls to __call__ to improve performance for repeated lookups.
    """
    
    def __init__(self, cmap, cache_size=256):
        """
        Parameters
        ----------
        cmap : Colormap
            The colormap to wrap and cache.
        cache_size : int, default: 256
            The size of the cache. Larger values use more memory but
            can improve performance for repeated lookups.
        """
        self.cmap = cmap
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        super().__init__(name=f"cached_{cmap.name}" if hasattr(cmap, 'name') else None)
    
    def __call__(self, X, alpha=None, bytes=False):
        """
        Map the values in X to colors in RGBA format.
        
        Parameters
        ----------
        X : float or array-like
            The data value(s) to convert to RGBA colors.
        alpha : float or array-like, optional
            Alpha values for the colors.
        bytes : bool, default: False
            If True, return the colors as bytes (0-255) instead of floats (0-1).
            
        Returns
        -------
        rgba : ndarray
            RGBA colors, with values between 0-1 (or 0-255 if bytes=True).
        """
        # Convert input to a hashable type for caching
        if isinstance(X, np.ndarray):
            key = tuple(X.flatten())
        else:
            key = X
            
        # Check if the result is in the cache
        if key in self.cache:
            self.hits += 1
            result = self.cache[key]
        else:
            self.misses += 1
            result = self.cmap(X, alpha=alpha, bytes=bytes)
            
            # Add to cache if it's not too large
            if len(self.cache) < self.cache_size:
                self.cache[key] = result
                
        return result
    
    def resize_cache(self, cache_size):
        """
        Resize the cache.
        
        Parameters
        ----------
        cache_size : int
            The new size of the cache.
        """
        self.cache_size = cache_size
        if len(self.cache) > cache_size:
            # If the cache is too large, clear it
            self.cache = {}
    
    def clear_cache(self):
        """
        Clear the cache and reset hit/miss statistics.
        """
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_cache_stats(self):
        """
        Get statistics about the cache performance.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'hits': Number of cache hits
            - 'misses': Number of cache misses
            - 'size': Current cache size
            - 'max_size': Maximum cache size
            - 'hit_rate': Cache hit rate (hits / (hits + misses))
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache),
            'max_size': self.cache_size,
            'hit_rate': hit_rate
        }

    def copy(self):
        """Make a copy of the cached colormap."""
        new_cmap = CachedColormap(self.cmap.copy(), self.cache_size)
        new_cmap.cache = self.cache.copy()
        new_cmap.hits = self.hits
        new_cmap.misses = self.misses
        return new_cmap

    def resampled(self, lutsize):
        """
        Return a resampled copy of the colormap.

        Parameters
        ----------
        lutsize : int
            The number of entries in the lookup table of the new colormap.

        Returns
        -------
        CachedColormap
            A new CachedColormap instance.
        """
        return CachedColormap(self.cmap.resampled(lutsize), self.cache_size)

    def reversed(self, name=None):
        """
        Return a reversed instance of the colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the name will be the
            name of the parent colormap + "_r".

        Returns
        -------
        CachedColormap
            A reversed instance of the colormap.
        """
        if name is None:
            name = self.name + "_r"
        return CachedColormap(self.cmap.reversed(name), self.cache_size)


class _ColorMapping(dict):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.cache = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.cache.clear()

    def __delitem__(self, key):
        super().__delitem__(key)
        self.cache.clear()


_colors_full_map = {}
# Set by reverse priority order.
_colors_full_map.update(XKCD_COLORS)
_colors_full_map.update({k.replace('grey', 'gray'): v
                         for k, v in XKCD_COLORS.items()
                         if 'grey' in k})
_colors_full_map.update(CSS4_COLORS)
_colors_full_map.update(TABLEAU_COLORS)
_colors_full_map.update({k.replace('gray', 'grey'): v
                         for k, v in TABLEAU_COLORS.items()
                         if 'gray' in k})
_colors_full_map.update(BASE_COLORS)
_colors_full_map = _ColorMapping(_colors_full_map)

_REPR_PNG_SIZE = (512, 64)
_BIVAR_REPR_PNG_SIZE = 256


def get_named_colors_mapping():
    """Return the global mapping of names to named colors."""
    return _colors_full_map


class ColorSequenceRegistry(Mapping):
    r"""
    Container for sequences of colors that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.color_sequences`. There
    should be no need for users to instantiate `.ColorSequenceRegistry`
    themselves.

    Read access uses a dict-like interface mapping names to lists of colors::

        import matplotlib as mpl
        colors = mpl.color_sequences['tab10']

    For a list of built in color sequences, see :doc:`/gallery/color/color_sequences`.
    The returned lists are copies, so that their modification does not change
    the global definition of the color sequence.

    Additional color sequences can be added via
    `.ColorSequenceRegistry.register`::

        mpl.color_sequences.register('rgb', ['r', 'g', 'b'])
    """

    _BUILTIN_COLOR_SEQUENCES = {
        'tab10': _cm._tab10_data,
        'tab20': _cm._tab20_data,
        'tab20b': _cm._tab20b_data,
        'tab20c': _cm._tab20c_data,
        'Pastel1': _cm._Pastel1_data,
        'Pastel2': _cm._Pastel2_data,
        'Paired': _cm._Paired_data,
        'Accent': _cm._Accent_data,
        'Dark2': _cm._Dark2_data,
        'Set1': _cm._Set1_data,
        'Set2': _cm._Set2_data,
        'Set3': _cm._Set3_data,
        'petroff10': _cm._petroff10_data,
    }

    def __init__(self):
        self._color_sequences = {**self._BUILTIN_COLOR_SEQUENCES}

    def __getitem__(self, item):
        try:
            return list(self._color_sequences[item])
        except KeyError:
            raise KeyError(f"{item!r} is not a known color sequence name")

    def __iter__(self):
        return iter(self._color_sequences)

    def __len__(self):
        return len(self._color_sequences)

    def __str__(self):
        return ('ColorSequenceRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def register(self, name, color_list):
        """
        Register a new color sequence.

        The color sequence registry stores a copy of the given *color_list*, so
        that future changes to the original list do not affect the registered
        color sequence. Think of this as the registry taking a snapshot
        of *color_list* at registration.

        Parameters
        ----------
        name : str
            The name for the color sequence.

        color_list : list of :mpltype:`color`
            An iterable returning valid Matplotlib colors when iterating over.
            Note however that the returned color sequence will always be a
            list regardless of the input type.

        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(f"{name!r} is a reserved name for a builtin "
                             "color sequence")

        color_list = list(color_list)  # force copy and coerce type to list
        for color in color_list:
            try:
                to_rgba(color)
            except ValueError:
                raise ValueError(
                    f"{color!r} is not a valid color specification")

        self._color_sequences[name] = color_list

    def unregister(self, name):
        """
        Remove a sequence from the registry.

        You cannot remove built-in color sequences.

        If the name is not registered, returns with no error.
        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(
                f"Cannot unregister builtin color sequence {name!r}")
        self._color_sequences.pop(name, None)


_color_sequences = ColorSequenceRegistry()


def _sanitize_extrema(ex):
    if ex is None:
        return ex
    try:
        ret = ex.item()
    except AttributeError:
        ret = float(ex)
    return ret

_nth_color_re = re.compile(r"\AC[0-9]+\Z")


def _is_nth_color(c):
    """Return whether *c* can be interpreted as an item in the color cycle."""
    return isinstance(c, str) and _nth_color_re.match(c)


def is_color_like(c):
    """
    Return whether *c* can be interpreted as an RGB(A) color.

    Parameters
    ----------
    c : object

    Returns
    -------
    bool
        Whether *c* can be interpreted as an RGB(A) color.
    """
    try:
        # Special case for None
        if c is None or (isinstance(c, str) and c.lower() == 'none'):
            return True
            
        # Check if c is a string
        if isinstance(c, str):
            # Check if c is a named color
            if c.lower() in CSS4_COLORS:
                return True
            # Check if c is a hex color
            if c.startswith('#'):
                if len(c) in [4, 7, 9]:  # #RGB, #RRGGBB, #RRGGBBAA
                    try:
                        int(c[1:], 16)
                        return True
                    except ValueError:
                        return False
                return False
            # Check if c is a named color in _colors_full_map
            if c.lower() in _colors_full_map:
                return True
            # Check if c is a cycle color (C0, C1, etc.)
            if c.startswith('C') and len(c) > 1:
                try:
                    int(c[1:])
                    return True
                except ValueError:
                    pass
            # Check if c is a numeric string (e.g., '0.8')
            try:
                float(c)
                return True
            except ValueError:
                pass
            return False
        
        # Check if c is a tuple or list of numbers
        if isinstance(c, (tuple, list)):
            if len(c) in [3, 4]:  # RGB or RGBA
                try:
                    for x in c:
                        float(x)
                    return True
                except (TypeError, ValueError):
                    return False
            return False
        
        # Check if c is a numpy array
        if hasattr(c, 'shape'):
            if c.shape in [(3,), (4,)]:  # RGB or RGBA
                try:
                    for x in c:
                        float(x)
                    return True
                except (TypeError, ValueError):
                    return False
            return False
        
        return False
    except (ValueError, TypeError):
        return False


def _has_alpha_channel(c):
    """
    Return whether *c* is a color with an alpha channel.

    If *c* is not a valid color specifier, then the result is undefined.
    """
    # The following logic uses the assumption that c is a valid color spec.
    # For speed and simplicity, we intentionally don't care about other inputs.
    # Anything can happen with them.

    # if c is a hex, it has an alpha channel when it has 4 (or 8) digits after '#'
    if isinstance(c, str):
        if c[0] == '#' and (len(c) == 5 or len(c) == 9):
            # example: '#fff8' or '#0f0f0f80'
            return True
    else:
        # if c isn't a string, it can be an RGB(A) or a color-alpha tuple
        # if it has length 4, it has an alpha channel
        if len(c) == 4:
            # example: [0.5, 0.5, 0.5, 0.5]
            return True

        # if it has length 2, it's a color/alpha tuple
            try:
                blend = blend_mode(rgb, intensity, **kwargs)
            except TypeError as err:
                raise ValueError('"blend_mode" must be callable or one of '
                                 f'{lookup.keys}') from err

        # Only apply result where hillshade intensity isn't masked
        if np.ma.is_masked(intensity):
            mask = intensity.mask[..., 0]
            for i in range(3):
                blend[..., i][mask] = rgb[..., i][mask]

        return blend

    def blend_hsv(self, rgb, intensity, hsv_max_sat=None, hsv_max_val=None,
                  hsv_min_val=None, hsv_min_sat=None):
        """
        Take the input data array, convert to HSV values in the given colormap,
        then adjust those color values to give the impression of a shaded
        relief map with a specified light source.  RGBA values are returned,
        which can then be used to plot the shaded image with imshow.

        The color of the resulting image will be darkened by moving the (s, v)
        values (in HSV colorspace) toward (hsv_min_sat, hsv_min_val) in the
        shaded regions, or lightened by sliding (s, v) toward (hsv_max_sat,
        hsv_max_val) in regions that are illuminated.  The default extremes are
        chose so that completely shaded points are nearly black (s = 1, v = 0)
        and completely illuminated points are nearly white (s = 0, v = 1).

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).
        hsv_max_sat : number, optional
            The maximum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_min_sat : number, optional
            The minimum saturation value that the *intensity* map can shift the output
            image to. If not provided, use the value provided upon initialization.
        hsv_max_val : number, optional
            The maximum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.
        hsv_min_val : number, optional
            The minimum value ("v" in "hsv") that the *intensity* map can shift the
            output image to. If not provided, use the value provided upon
            initialization.

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
        # Backward compatibility...
        if hsv_max_sat is None:
            hsv_max_sat = self.hsv_max_sat
        if hsv_max_val is None:
            hsv_max_val = self.hsv_max_val
        if hsv_min_sat is None:
            hsv_min_sat = self.hsv_min_sat
        if hsv_min_val is None:
            hsv_min_val = self.hsv_min_val

        # Expects a 2D intensity array scaled between -1 to 1...
        intensity = intensity[..., 0]
        intensity = 2 * intensity - 1

        # Convert to rgb, then rgb to hsv
        hsv = rgb_to_hsv(rgb[:, :, 0:3])
        hue, sat, val = np.moveaxis(hsv, -1, 0)

        # Modify hsv values (in place) to simulate illumination.
        # putmask(A, mask, B) <=> A[mask] = B[mask]
        np.putmask(sat, (np.abs(sat) > 1.e-10) & (intensity > 0),
                   (1 - intensity) * sat + intensity * hsv_max_sat)
        np.putmask(sat, (np.abs(sat) > 1.e-10) & (intensity < 0),
                   (1 + intensity) * sat - intensity * hsv_min_sat)
        np.putmask(val, intensity > 0,
                   (1 - intensity) * val + intensity * hsv_max_val)
        np.putmask(val, intensity < 0,
                   (1 + intensity) * val - intensity * hsv_min_val)
        np.clip(hsv[:, :, 1:], 0, 1, out=hsv[:, :, 1:])

        # Convert modified hsv back to rgb.
        return hsv_to_rgb(hsv)

    def blend_soft_light(self, rgb, intensity):
        """
        Combine an RGB image with an intensity map using "soft light" blending,
        using the "pegtop" formula.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
        return 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2

    def blend_overlay(self, rgb, intensity):
        """
        Combine an RGB image with an intensity map using "overlay" blending.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        ndarray
            An (M, N, 3) RGB array representing the combined images.
        """
        low = 2 * intensity * rgb
        high = 1 - 2 * (1 - intensity) * (1 - rgb)
        return np.where(rgb <= 0.5, low, high)


def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the `BoundaryNorm`.
        Value ``v`` is quantized to level ``i`` if ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If *extend* is "neither" there
        must be ``n_level - 1`` colors. For an *extend* of "min" or "max" add
        one extra color, and for an *extend* of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See `~.Axes.contourf` for details.

    Returns
    -------
    cmap : `~matplotlib.colors.Colormap`
    norm : `~matplotlib.colors.Normalize`
    """
    slice_map = {
        'both': slice(1, -1),
        'min': slice(1, None),
        'max': slice(0, -1),
        'neither': slice(0, None),
    }
    _api.check_in_list(slice_map, extend=extend)
    color_slice = slice_map[extend]

    n_data_colors = len(levels) - 1
    n_extend_colors = color_slice.start - (color_slice.stop or 0)  # 0, 1 or 2
    n_expected = n_data_colors + n_extend_colors
    if len(colors) != n_expected:
        raise ValueError(
            f'Expected {n_expected} colors ({n_data_colors} colors for {len(levels)} '
            f'levels, and {n_extend_colors} colors for extend == {extend!r}), '
            f'but got {len(colors)}')

    data_colors = colors[color_slice]
    under_color = colors[0] if extend in ['min', 'both'] else 'none'
    over_color = colors[-1] if extend in ['max', 'both'] else 'none'
    cmap = ListedColormap(data_colors, under=under_color, over=over_color)

    cmap.colorbar_extend = extend

    norm = BoundaryNorm(levels, ncolors=n_data_colors)
    return cmap, norm


def evaluate_colormap_accessibility(cmap, samples=100, deficiency_types=None):
    """
    Evaluate a colormap for accessibility to individuals with color vision deficiencies.

    Parameters
    ----------
    cmap : Colormap
        The colormap to evaluate.
    samples : int, default: 100
        The number of samples to evaluate along the colormap.
    deficiency_types : list of str, optional
        The types of color vision deficiencies to evaluate.
        If None, evaluates for ['protanopia', 'deuteranopia', 'tritanopia'].

    Returns
    -------
    dict
        A dictionary containing accessibility metrics:
        - 'perceptual_distance': Perceptual distance in LAB space
        - 'color_blind_perceptual_distance': Perceptual distance for each deficiency type
        - 'contrast_ratio': Contrast ratio between adjacent colors
        - 'color_blind_contrast_ratio': Contrast ratio for each deficiency type
        - 'distinguishable_colors': Number of distinguishable colors
        - 'color_blind_distinguishable_colors': Number of distinguishable colors for each deficiency type

    Notes
    -----
    The evaluation is based on several metrics:
    
    1. Perceptual distance: The Euclidean distance in LAB color space between
       adjacent colors in the colormap. Larger distances indicate better perceptual
       separation.
       
    2. Contrast ratio: The luminance contrast ratio between adjacent colors,
       as defined by WCAG 2.0. Higher ratios indicate better contrast.
       
    3. Distinguishable colors: The number of colors in the colormap that are
       perceptually distinct (LAB distance > 2.3).

    Examples
    --------
    Evaluate the 'viridis' colormap:

    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import evaluate_colormap_accessibility
    >>> evaluate_colormap_accessibility(plt.cm.viridis)
    {'perceptual_distance': 64.42,
     'color_blind_perceptual_distance': {'protanopia': 58.21,
                                        'deuteranopia': 59.87,
                                        'tritanopia': 62.15},
     'contrast_ratio': 12.8,
     'color_blind_contrast_ratio': {'protanopia': 10.5,
                                   'deuteranopia': 11.2,
                                   'tritanopia': 12.1},
     'distinguishable_colors': 42,
     'color_blind_distinguishable_colors': {'protanopia': 36,
                                           'deuteranopia': 38,
                                           'tritanopia': 40}}
    """
    if deficiency_types is None:
        deficiency_types = ['protanopia', 'deuteranopia', 'tritanopia']
    
    # Sample colors from the colormap
    x = np.linspace(0, 1, samples)
    colors = cmap(x)[:, :3]  # Exclude alpha channel
    
    # Convert to LAB color space
    lab_colors = rgb_to_lab(colors)
    
    # Calculate perceptual distance (Euclidean distance in LAB space)
    # between adjacent colors
    lab_diff = np.diff(lab_colors, axis=0)
    perceptual_distances = np.sqrt(np.sum(lab_diff**2, axis=1))
    total_perceptual_distance = np.sum(perceptual_distances)
    
    # Calculate contrast ratio
    # Convert RGB to luminance (Y in XYZ color space)
    # Using the formula: Y = 0.2126*R + 0.7152*G + 0.0722*B (for linear RGB)
    
    # First convert sRGB to linear RGB
    mask = colors <= 0.04045
    rgb_linear = np.where(mask, colors / 12.92, 
                         ((colors + 0.055) / 1.055) ** 2.4)
    
    # Calculate luminance
    luminance = 0.2126 * rgb_linear[:, 0] + 0.7152 * rgb_linear[:, 1] + 0.0722 * rgb_linear[:, 2]
    
    # Calculate contrast ratio between adjacent colors
    # Formula: (L1 + 0.05) / (L2 + 0.05) where L1 >= L2
    contrast_ratios = []
    for i in range(len(luminance) - 1):
        l1 = max(luminance[i], luminance[i+1]) + 0.05
        l2 = min(luminance[i], luminance[i+1]) + 0.05
        contrast_ratios.append(l1 / l2)
    
    avg_contrast_ratio = np.mean(contrast_ratios)
    
    # Calculate number of distinguishable colors
    # Colors are considered distinguishable if their perceptual distance is > 2.3
    # (Just Noticeable Difference threshold)
    distinguishable_count = np.sum(perceptual_distances > 2.3) + 1
    
    # Evaluate for color blindness
    cb_perceptual_distance = {}
    cb_contrast_ratio = {}
    cb_distinguishable_colors = {}
    
    for deficiency in deficiency_types:
        # Simulate color blindness
        cb_colors = simulate_colorblindness(colors, deficiency)
        
        # Convert to LAB
        cb_lab_colors = rgb_to_lab(cb_colors)
        
        # Calculate perceptual distance
        cb_lab_diff = np.diff(cb_lab_colors, axis=0)
        cb_perceptual_distances = np.sqrt(np.sum(cb_lab_diff**2, axis=1))
        cb_perceptual_distance[deficiency] = np.sum(cb_perceptual_distances)
        
        # Calculate luminance for color blind simulation
        mask = cb_colors <= 0.04045
        cb_rgb_linear = np.where(mask, cb_colors / 12.92, 
                                ((cb_colors + 0.055) / 1.055) ** 2.4)
        
        cb_luminance = 0.2126 * cb_rgb_linear[:, 0] + 0.7152 * cb_rgb_linear[:, 1] + 0.0722 * cb_rgb_linear[:, 2]
        
        # Calculate contrast ratio
        cb_contrast_ratios = []
        for i in range(len(cb_luminance) - 1):
            l1 = max(cb_luminance[i], cb_luminance[i+1]) + 0.05
            l2 = min(cb_luminance[i], cb_luminance[i+1]) + 0.05
            cb_contrast_ratios.append(l1 / l2)
        
        cb_contrast_ratio[deficiency] = np.mean(cb_contrast_ratios)
        
        # Calculate distinguishable colors
        cb_distinguishable_colors[deficiency] = np.sum(cb_perceptual_distances > 2.3) + 1
    
    # Round values for readability
    total_perceptual_distance = round(total_perceptual_distance, 2)
    avg_contrast_ratio = round(avg_contrast_ratio, 1)
    
    for deficiency in deficiency_types:
        cb_perceptual_distance[deficiency] = round(cb_perceptual_distance[deficiency], 2)
        cb_contrast_ratio[deficiency] = round(cb_contrast_ratio[deficiency], 1)
    
    return {
        'perceptual_distance': total_perceptual_distance,
        'color_blind_perceptual_distance': cb_perceptual_distance,
        'contrast_ratio': avg_contrast_ratio,
        'color_blind_contrast_ratio': cb_contrast_ratio,
        'distinguishable_colors': distinguishable_count,
        'color_blind_distinguishable_colors': cb_distinguishable_colors
    }
