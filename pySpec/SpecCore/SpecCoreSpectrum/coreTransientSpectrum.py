"""
This file is part of pySpec
    Copyright (C) 2024  Markus Bauer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pySpec is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from scipy.interpolate import interp2d, RegularGridInterpolator
import numpy as np
from copy import deepcopy

from .coreAbstractSpectrum import AbstractSpectrum
from .coreSpectrum import Spectrum
from .coreSpectrumSlice import SliceFactory

from ..SpecCoreData.coreTwoDimensionalData import TwoDimensionalData

from ..SpecCoreAxis.coreAbstractAxis import AbstractAxis
from ..SpecCoreAxis.coreTimeAxis import TimeAxis
from ..SpecCoreAxis.coreEnergyAxis import EnergyAxis
from ..SpecCoreAxis.coreWavelengthAxis import WavelengthAxis

from ..coreFunctions import inPlaceOp


class TransientSpectrum(AbstractSpectrum):
    """
    Class to represent the time evolution of spectra. Therefore, it possesses an energy axis (Wavelength/Wavenumber/etc.)
    as well as a time axis, with the respective units.

    :param x_array: The energy-axis as a np.ndarray or a subclass of AbstractAxis
    :param t_array: The time axis as a np.ndarray or a subclass of AbstractAxis
    :param data_array: The data-array as a np.ndarray or an instance of TwoDimensionalData
    :param x_unit: The unit of the x-axis. Can be 'wn', 'wl' or 'ev'.
    :param t_unit: The unit of the t-axis. Supports different magnitudes like 'ps', 'ns', 'us', 'ms' etc.
    :param data_unit: The unit of the data array.
    """

    def __init__(self,
                 x_array: np.ndarray or AbstractAxis,
                 t_array: np.ndarray or AbstractAxis,
                 data_array: np.ndarray or TwoDimensionalData,
                 x_unit='wn', t_unit='s', data_unit='mdod'):

        if isinstance(data_array, np.ndarray):
            super().__init__(TwoDimensionalData(data_array, data_unit))
        elif isinstance(data_array, TwoDimensionalData):
            super().__init__(deepcopy(data_array))
        else:
            raise ValueError(f"data_array has to be of type: np.ndarray or TwoDimensionalData, not: {type(data_array)}")

        if isinstance(t_array, np.ndarray):
            self._t_axis = TimeAxis(t_array, t_unit)
        elif isinstance(t_array, TimeAxis):
            self._t_axis = deepcopy(t_array)
        else:
            raise ValueError(f"t_array has to be of type: np.ndarray or TimeAxis, not: {type(t_array)}")

        if isinstance(x_array, np.ndarray):
            if x_unit in ('wn', 'ev'):
                self._x_axis = EnergyAxis(x_array, x_unit)
            elif x_unit == 'wl':
                self._x_axis = WavelengthAxis(x_array, x_unit)
            else:
                raise ValueError(f"x_unit has to be 'wn', 'ev', or 'wl' not {x_unit}")

        elif isinstance(x_array, AbstractAxis):
            self._x_axis = deepcopy(x_array)
        else:
            raise ValueError(f"x_array has to be of type: np.ndarray or EnergyAxis, not: {type(x_array)}")

        self._check_dimensions()
        self._pre_scan = None

    def __and__(self, other):
        return self.extend(other)

    def extend(self, other: AbstractSpectrum):
        """
        Extends the data along the time axis.

        :param other: Another Transient Spectrum to be added
        :return: New TransientSpectrum instance
        """
        if not isinstance(other, TransientSpectrum):
            raise ValueError("Can only concatenate TransientSpectrum with another instance of TransientSpectrum!")

        other.interpolate_to(self.x, None)

        self.orient_data('x')
        other.orient_data('x')

        t: np.ndarray = np.concatenate((self.t.array, other.t.array))
        y: np.ndarray = np.concatenate((self.y.array, other.y.array), axis=1)

        spec = TransientSpectrum(self.x, t, y, self._x_axis.unit, self._t_axis.unit, self._data.unit)

        return spec

    def __or__(self, other):
        return self.append(other)

    def append(self, other):
        if not isinstance(other, TransientSpectrum):
            raise ValueError("Can only concatenate TransientSpectrum with another instance of TransientSpectrum!")

        other.interpolate_to(None, self.t)

        self.orient_data('t')
        other.orient_data('t')

        x = np.concatenate((self.x.array, other.x.array))
        y = np.concatenate((self.y.array, other.y.array), axis=1)

        return TransientSpectrum(x, self.t, y, self._x_axis.unit, self._t_axis.unit, self._data.unit)

    def subtract(self, other):
        if not isinstance(other, TransientSpectrum) and not isinstance(other, Spectrum):
            raise ValueError("Can only subtract TransientSpectrum from another instance of TransientSpectrum or "
                             "an Instance of Spectrum!")

        if isinstance(other, TransientSpectrum):
            other.interpolate_to(self.x, self.t)
            other.orient_data('x')

        if isinstance(other, Spectrum):
            other.interpolate_to(self.x)

        self.orient_data('x')

        return TransientSpectrum(self.x,
                                 self.t,
                                 self.y - other.y,
                                 self._x_axis.unit,
                                 self._t_axis.unit,
                                 self._data.unit)
    
    @property
    def pre_scan(self):
        return self._pre_scan

    @property
    def x(self):
        return self._x_axis

    @x.setter
    def x(self, array):
        if isinstance(array, np.ndarray):
            self._x_axis.array = array.copy()
            return
        if isinstance(array, WavelengthAxis) or isinstance(array, EnergyAxis):
            self._x_axis = deepcopy(array)
            return

        raise ValueError(f"array must be of type np.ndarray, not {type(array)}")

    @property
    def t(self):
        return self._t_axis

    @t.setter
    def t(self, array):
        if isinstance(array, np.ndarray):
            self._t_axis.array = array
            return
        if isinstance(array, TimeAxis):
            self._t_axis = deepcopy(array)
            return

        raise ValueError(f"array must be of type np.ndarray, not {type(array)}")

    @property
    def transient(self):
        return SliceFactory(self._t_axis, self._data, self._x_axis)

    @property
    def spectrum(self):
        return SliceFactory(self._x_axis, self._data, self._t_axis)

    @inPlaceOp
    def average(self, x_width=None, t_width=None):
        if x_width is not None:
            self.orient_data('x')
            self.x, self.y = self._reductive_average(self.x, x_width)
        if t_width is not None:
            self.orient_data('t')
            self.t, self.y = self._reductive_average(self.t, t_width)

        return self

    @inPlaceOp
    def sort(self):
        sorting_mask = self._x_axis.sort_array()
        self._data.sort_by(sorting_mask)
        sorting_mask = self._t_axis.sort_array()
        self._data.sort_by(sorting_mask)

        return self

    @inPlaceOp
    def interpolate_to(self,
                       x_axis: None or EnergyAxis or WavelengthAxis = None,
                       t_axis: None or TimeAxis = None):
        if not isinstance(x_axis, EnergyAxis) and not isinstance(x_axis, WavelengthAxis) and x_axis is not None:
            raise ValueError(f"x_axis must be of type EnergyAxis or WavelengthAxis, not {type(x_axis)}")
        if not isinstance(t_axis, TimeAxis) and t_axis is not None:
            raise ValueError(f"t_axis must be of type TimeAxis, not {type(t_axis)}")

        self.orient_data('t')

        f = interp2d(self.x, self.t, np.nan_to_num(self.y),
                     bounds_error=False, fill_value=np.nan)

        if x_axis is None:
            x_axis = self._x_axis
        if t_axis is None:
            t_axis = self._t_axis

        self._data.array = f(x_axis.array, t_axis.array)
        self.x = x_axis
        self.t = t_axis

        return self

    def save(self, path):
        self.orient_data('t')

        with open(path, 'w') as file:
            file.write('NaN')

            for x in self._x_axis:
                file.write(f' {x}')
            file.write('\n')

            for i, t in enumerate(self._t_axis):
                file.write(f'{t}')
                for y in self._data[i, :]:
                    file.write(f' {y}')
                file.write('\n')

        return self

    @inPlaceOp
    def eliminate_repetition(self):
        self.orient_data('x')
        self.x, self.y = self._eliminate_repetition_1d(self.x)
        self.orient_data('t')
        self.t, self.y = self._eliminate_repetition_1d(self.t)

        return self

    @inPlaceOp
    def subtract_prescans(self, until_time, from_time=None):
        self.sort()

        if self._pre_scan is not None:
            raise AttributeError("PreScans cannot be subtracted twice!")

        self.orient_data('t')
        until_time_idx = self.t.closest_to(until_time)[0]

        from_time_idx = 0
        if from_time is not None:
            from_time_idx = self.t.closest_to(from_time)[0]

        idx = slice(from_time_idx, until_time_idx)
        self._pre_scan = self.spectrum[idx]
        self.y = self._data.array - self._pre_scan.y.array

        return self

    @inPlaceOp
    def truncate_to(self, x_range=None, t_range=None):
        if x_range is not None:
            self.orient_data('x')
            x_range = [self.x.closest_to(x)[0] for x in x_range]
            self.x, self.y = self._truncate_one_dimension(x_range, self.x)
        if t_range is not None:
            self.orient_data('t')
            t_range = [self.t.closest_to(t)[0] for t in t_range]
            self.t, self.y = self._truncate_one_dimension(t_range, self.t)

        return self

    @inPlaceOp
    def truncate_like(self, x_array=None, t_array=None):
        if x_array is not None:
            self.orient_data('x')
            self.x, self.y = self._truncate_like_array_one_dimension(x_array, self.x)
        if t_array is not None:
            self.orient_data('t')
            self.t, self.y = self._truncate_like_array_one_dimension(t_array, self.t)

        return self

    def orient_data(self, direction='x'):
        if direction == 'x':
            if self._data.shape[0] == len(self._x_axis):
                return
        elif direction == 't':
            if self._data.shape[0] == len(self._t_axis):
                return
        else:
            raise ValueError(f"direction can only be x or t but was {direction}")

        self._data.transpose(inplace=True)

        return self

    @inPlaceOp
    def eliminate_positions(self, x_pos: float or None = None, t_pos: float or None = None):
        if x_pos is not None:
            x_pos = [self.x.closest_to(x)[0] for x in x_pos]
        if t_pos is not None:
            t_pos = [self.t.closest_to(t)[0] for t in t_pos]

        return self.eliminate_idx(x_idx=x_pos, t_idx=t_pos)

    @inPlaceOp
    def eliminate_idx(self, x_idx: int or None = None, t_idx: int or None = None):
        if x_idx is not None:
            self.orient_data('x')
            self.x, self.y = self._eliminate_pos_1d(self.x, x_idx)

        if t_idx is not None:
            self.orient_data('t')
            self.t, self.y = self._eliminate_pos_1d(self.t, t_idx)

        return self

    @inPlaceOp
    def correct_delay_drift(self, offset_end: float, time_zero: float = 0, time_end: float | None = None):
        self.orient_data('x')

        if time_end is None:
            time_end = np.max(self.t.array)

        slope = offset_end / (self.t.closest_to(time_end)[1] - self.t.closest_to(time_zero)[1])

        data_y = []
        for idx in range(len(self.t)):
            step: Spectrum = self.spectrum[idx]

            step.x += self.t[idx] * slope
            step.interpolate_to(self.x)

            data_y.append(step.y.array.copy())

        self.y = np.array(data_y)

    def _check_dimensions(self):
        if not (len(self._x_axis), len(self._t_axis)) == self._data.shape \
                and not (len(self._x_axis), len(self._t_axis)) == self._data.shape[::-1]:
            raise ValueError("Shape mismatch!")

    @classmethod
    def average_from(cls, spec_list: list):
        # TODO: Interpolating all data sets onto each other?
        y = []

        for item in spec_list:
            item.orient_data('x')
            y.append(item.y.array)

        if len(y) > 1:
            y = np.nanmean(y, axis=0)
        elif len(y) == 1:
            y = y[0]

        return cls(deepcopy(spec_list[0].x),
                   deepcopy(spec_list[0].t),
                   y,
                   spec_list[0].x.unit,
                   spec_list[0].t.unit,
                   spec_list[0].y.unit)

    @staticmethod
    def from_file(path, **kwargs):
        from ..coreParser import Parser

        path = Parser.parse_path(path)[0]

        args = {'import_type': 'x_first', 'x_unit': 'wn', 't_unit': 's', 'data_unit': 'od'}
        args.update(kwargs)

        return Parser(path, **args)[0]

    @classmethod
    def calculate_from(cls, spec, ref: Spectrum):
        spec.orient_data('x')

        y = cls.calculate_od(spec.y.array,
                             np.broadcast_to(
                                 ref.y.array[..., None],
                                 ref.y.array.shape+(len(spec.t),)
                                            )
                             )

        return cls(spec.x,
                   spec.t,
                   y,
                   spec.x.unit,
                   spec.t.unit,
                   spec.y.unit)

