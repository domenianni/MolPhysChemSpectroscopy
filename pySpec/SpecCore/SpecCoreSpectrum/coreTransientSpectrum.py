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

from scipy.interpolate import interp2d
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


class TransientSpectrum(AbstractSpectrum):

    def __init__(self,
                 x_array: np.ndarray or AbstractAxis,
                 t_array: np.ndarray or AbstractAxis,
                 data_array: np.ndarray or TwoDimensionalData,
                 x_unit='wn', t_unit='s', data_unit='dod'):

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

    def __add__(self, other):
        return self.extend(other)

    def __sub__(self, other):
        return self.subtract(other)

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

    def average(self, x_width=None, t_width=None):
        if x_width is not None:
            self.orient_data('x')
            self.x, self.y = self._reductive_average(self.x, x_width)
        if t_width is not None:
            self.orient_data('t')
            self.t, self.y = self._reductive_average(self.t, t_width)

        return self

    def sort(self):
        sorting_mask = self._x_axis.sort_array()
        self._data.sort_by(sorting_mask)
        sorting_mask = self._t_axis.sort_array()
        self._data.sort_by(sorting_mask)

        return self

    def interpolate_to(self,
                       x_axis: None or EnergyAxis or WavelengthAxis = None,
                       t_axis: None or TimeAxis = None,
                       inplace=True):
        data = self._inplace(inplace)

        if not isinstance(x_axis, EnergyAxis) and not isinstance(x_axis, WavelengthAxis) and x_axis is not None:
            raise ValueError(f"x_axis must be of type EnergyAxis or WavelengthAxis, not {type(x_axis)}")
        if not isinstance(t_axis, TimeAxis) and t_axis is not None:
            raise ValueError(f"t_axis must be of type TimeAxis, not {type(t_axis)}")

        data.orient_data('t')

        f = interp2d(data.x, data.t, np.nan_to_num(data.y),
                     bounds_error=False, fill_value=np.nan)

        if x_axis is None:
            x_axis = data._x_axis
        if t_axis is None:
            t_axis = data._t_axis

        data._data.array = f(x_axis.array, t_axis.array)
        data.x = x_axis
        data.t = t_axis

        return data

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

    def eliminate_repetition(self):
        self.orient_data('x')
        self.x, self.y = self._eliminate_repetition_1d(self.x)
        self.orient_data('t')
        self.t, self.y = self._eliminate_repetition_1d(self.t)

        return self

    def extend(self, other):
        if not isinstance(other, TransientSpectrum):
            raise ValueError("Can only concatenate TransientSpectrum with another instance of TransientSpectrum!")

        other.interpolate_to(self.x, None)

        self.orient_data('x')
        other.orient_data('x')

        t = np.concatenate((self.t.array, other.t.array))
        y = np.concatenate((self.y.array, other.y.array), axis=1)

        spec = TransientSpectrum(self.x, t, y, self._x_axis.unit, self._t_axis.unit, self._data.unit)

        return spec

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

    def truncate_to(self, x_range=None, t_range=None, inplace=True):
        data = self._inplace(inplace)

        if x_range is not None:
            data.orient_data('x')
            x_range = [self.x.closest_to(x)[0] for x in x_range]
            data.x, data.y = data._truncate_one_dimension(x_range, data.x)
        if t_range is not None:
            data.orient_data('t')
            t_range = [self.t.closest_to(t)[0] for t in t_range]
            data.t, data.y = data._truncate_one_dimension(t_range, data.t)

        return data

    def truncate_like(self, x_array=None, t_array=None, inplace=True):
        data = self._inplace(inplace)

        if x_array is not None:
            data.orient_data('x')
            data.x, data.y = data._truncate_like_array_one_dimension(x_array, data.x)
        if t_array is not None:
            data.orient_data('t')
            data.t, data.y = data._truncate_like_array_one_dimension(t_array, data.t)

        return data

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

    def eliminate_positions(self, x_pos=None, t_pos=None):
        if x_pos is not None:
            x_pos = [self.x.closest_to(x)[0] for x in x_pos]
        if t_pos is not None:
            t_pos = [self.t.closest_to(t)[0] for t in t_pos]

        return self.eliminate_idx(x_idx=x_pos, t_idx=t_pos)

    def eliminate_idx(self, x_idx=None, t_idx=None):
        if x_idx is not None:
            self.orient_data('x')
            self.x, self.y = self._eliminate_pos_1d(self.x, x_idx)

        if t_idx is not None:
            self.orient_data('t')
            self.t, self.y = self._eliminate_pos_1d(self.t, t_idx)

        return self

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
    def from_file(path, parser_args: dict = None):
        from Spectroscopy.SpecCore.coreParser import Parser

        path = Parser.parse_path(path)[0]

        args = {'import_type': 'x_first', 'x_unit': 'wn', 't_unit': 's', 'data_unit': 'od'}
        if parser_args is not None:
            args.update(parser_args)

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

