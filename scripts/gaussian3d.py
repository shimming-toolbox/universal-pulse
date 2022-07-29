# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Mathieu Guay-Paquet

"""A Gaussian3D convolution kernel for astropy."""
# pylint: disable=line-too-long, too-many-lines, too-many-arguments, invalid-name
import math
import numpy as np
from astropy.convolution.core import Kernel
from astropy.convolution.utils import discretize_model
from astropy.modeling.core import Model, FittableModel, custom_model
from astropy.modeling.parameters import Parameter
from astropy.units import UnitsError

__all__ = ['Fittable3DModel', 'Gaussian3D', 'discretize_model_3d', 'Kernel3D', 'Gaussian3DKernel']

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)


class Fittable3DModel(FittableModel):
    """
    Base class for three-dimensional fittable models.
    """
    n_inputs = 3
    n_outputs = 1


class Gaussian3D(Fittable3DModel):
    r"""
    Three dimensional Gaussian model.
    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Amplitude (peak value) of the Gaussian.
    stddev : float or `~astropy.units.Quantity`.
        Standard deviation of the Gaussian with FWHM = 2 * stddev * np.sqrt(2 * np.log(2)).
    Notes
    -----
    In each dimension, ``mean``, ``stddev``, and the input must all have the
    same units or all be unitless.
    Model formula:
        .. math:: f(x) = A e^{- \frac{(x - x_0)^2}{2 \sigma^2}}
    """

    amplitude = Parameter(default=1, description="Amplitude (peak value) of the Gaussian")
    stddev = Parameter(default=1, bounds=(FLOAT_EPSILON, None),
                       description="Standard deviation of the Gaussian")

    def bounding_box(self, factor=5.5):
        """
        Tuple defining the default ``bounding_box`` limits in each dimension,
        ``((z_low, z_high), (y_low, y_high), (x_low, x_high))``
        The default offset from the mean is 5.5-sigma, corresponding
        to a relative error < 1e-7.
        Parameters
        ----------
        factor : float, optional
            The multiple of `x_stddev` and `y_stddev` used to define the limits.
            The default is 5.5.
        """
        print("CALLED bounding_box")
        return ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))

    @staticmethod
    def evaluate(x, y, z, amplitude, stddev):
        """
        Gaussian3D model function.
        """
        print("CALLED evaluate")
        return amplitude * np.exp(-0.5 * (x ** 2 + y ** 2 + z ** 2) / stddev ** 2)

    @staticmethod
    def fit_deriv(x, y, z, amplitude, stddev):
        """
        Gaussian3D model function derivatives.
        """
        print("CALLED fit_deriv")
        scaled_norm2 = (x ** 2 + y ** 2 + z ** 2) / stddev ** 2
        d_amplitude = np.exp(-0.5 * scaled_norm2)
        d_stddev = amplitude * d_amplitude * scaled_norm2 / stddev
        return [d_amplitude, d_stddev]

    @property
    def input_units(self):
        print("CALLED input_units")
        return None

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        print("CALLED _parameter_units_for_data_units")
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        unit_in = inputs_unit[self.inputs[0]]
        if unit_in == inputs_unit[self.inputs[1]] == inputs_unit[self.inputs[2]]:
            return {'amplitude': outputs_unit[self.outputs[0]],
                    'stddev': unit_in}
        raise UnitsError("Units of 'x' and 'y' and 'z' inputs should match")


def discretize_model_3d(model, x_range, y_range, z_range):
    """
    Function to evaluate analytical model functions on a 3-d grid.
    The model is discretized by taking the value at the center of the bin.
    Parameters
    ----------
    model : `~astropy.modeling.Model` or callable.
        Analytic model function to be discretized. Callables, which are not an
        instances of `~astropy.modeling.Model` are passed to
        `~astropy.modeling.custom_model` and then evaluated.
    x_range : tuple
        x range in which the model is evaluated. The difference between the
        upper an lower limit must be a whole number, so that the output array
        size is well defined.
    y_range : tuple
        x range in which the model is evaluated. The difference between the
        upper an lower limit must be a whole number, so that the output array
        size is well defined.
    z_range : tuple
        x range in which the model is evaluated. The difference between the
        upper an lower limit must be a whole number, so that the output array
        size is well defined.
    Returns
    -------
    array : `numpy.array`
        Model value array
    """
    if not callable(model):
        raise TypeError('Model must be callable.')
    if not isinstance(model, Model):
        model = custom_model(model)()
    if model.n_inputs != 3:
        raise ValueError('discretize_model_3d only supports 3-d models.')

    x_start, x_stop = x_range
    if not float(x_stop - x_start).is_integer():
        raise ValueError("The difference between the upper and lower limit of"
                         " 'x_range' must be a whole number.")
    y_start, y_stop = y_range
    if not float(y_stop - y_start).is_integer():
        raise ValueError("The difference between the upper and lower limit of"
                         " 'y_range' must be a whole number.")
    z_start, z_stop = z_range
    if not float(z_stop - z_start).is_integer():
        raise ValueError("The difference between the upper and lower limit of"
                         " 'z_range' must be a whole number.")

    x, y, z = np.meshgrid(np.arange(x_start, x_stop),
                          np.arange(y_start, y_stop),
                          np.arange(z_start, z_stop))
    return model(x, y, z)


class Kernel3D(Kernel):
    """
    Base class for 3D filter kernels.
    Parameters
    ----------
    model : `~astropy.modeling.FittableModel`
        Model to be evaluated.
    x_size : int or None, optional
        Size in x direction of the kernel array. Default = ⌊8*width + 1⌋.
        Only used if ``array`` is None.
    y_size : int or None, optional
        Size in y direction of the kernel array. Default = ⌊8*width + 1⌋.
        Only used if ``array`` is None.
    z_size : int or None, optional
        Size in z direction of the kernel array. Default = ⌊8*width + 1⌋.
        Only used if ``array`` is None.
    array : ndarray or None, optional
        Kernel array.
    """

    def __init__(self, model=None, x_size=None, y_size=None, z_size=None, array=None):
        # Initialize from model
        if self._model:
            if array is not None:
                # Reject "array" keyword for kernel models, to avoid them not being
                # populated as expected.
                raise TypeError("Array argument not allowed for kernel models.")

            if x_size is None:
                x_size = self._default_size
            elif x_size != int(x_size):
                raise TypeError("x_size should be an integer")

            if y_size is None:
                y_size = self._default_size
            elif y_size != int(y_size):
                raise TypeError("y_size should be an integer")

            if z_size is None:
                z_size = self._default_size
            elif z_size != int(z_size):
                raise TypeError("z_size should be an integer")

            # Set ranges where to evaluate the model

            if x_size % 2 == 0:  # even kernel
                x_range = (-(int(x_size)) // 2 + 0.5, (int(x_size)) // 2 + 0.5)
            else:  # odd kernel
                x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)

            if y_size % 2 == 0:  # even kernel
                y_range = (-(int(y_size)) // 2 + 0.5, (int(y_size)) // 2 + 0.5)
            else:  # odd kernel
                y_range = (-(int(y_size) - 1) // 2, (int(y_size) - 1) // 2 + 1)

            if z_size % 2 == 0:  # even kernel
                z_range = (-(int(z_size)) // 2 + 0.5, (int(z_size)) // 2 + 0.5)
            else:  # odd kernel
                z_range = (-(int(z_size) - 1) // 2, (int(z_size) - 1) // 2 + 1)

            array = discretize_model_3d(self._model, x_range, y_range, z_range)

        # Initialize from array
        elif array is None:
            raise TypeError("Must specify either array or model.")

        super().__init__(array)


class Gaussian3DKernel(Kernel3D):
    """
    3D Gaussian filter kernel.
    The Gaussian filter is a filter with great smoothing properties. It is
    isotropic and does not produce artifacts.
    The generated kernel is normalized so that it integrates to 1.
    Parameters
    ----------
    stddev : number
        Standard deviation of the Gaussian kernel.
    x_size : int or None, optional
        Size in x direction of the kernel array. Default = ⌊8*stddev + 1⌋.
    y_size : int or None, optional
        Size in y direction of the kernel array. Default = ⌊8*stddev + 1⌋.
    z_size : int or None, optional
        Size in z direction of the kernel array. Default = ⌊8*stddev + 1⌋.
    """

    _separable = True  # TODO: Check if this is true
    _is_bool = False

    def __init__(self, stddev, **kwargs):
        self._model = Gaussian3D(1. / (np.sqrt(2 * np.pi) * stddev), stddev)
        self._default_size = math.ceil(8 * stddev) | 1
        super().__init__(**kwargs)
        self.normalize()
