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

from .coreAbstractSpectrum import AbstractSpectrum
from ..SpecCoreAxis.coreAbstractAxis import AbstractAxis
from ..SpecCoreData.coreOneDimensionalData import OneDimensionalData
from ..SpecCoreAxis.coreTimeAxis import TimeAxis
from ..coreFunctions import inPlaceOp

import numpy as np
from copy import deepcopy


class Transient(AbstractSpectrum):
    """
    Class representing a temporal evolution at a specific energy position. Usually created from a
    :class:`TransientSpectrum`-object via its .transient property.

    :param t_array:
    :param data_array:
    :param t_unit:
    :param data_unit:
    :param position: Optional. The energy position (in wavenumber/wavelength etc.) for assignment.
    """

    __slots__ = ('_t_axis', '_position')

    def __init__(self, t_array, data_array, t_unit='s', data_unit='od', position=None):

        if isinstance(data_array, np.ndarray):
            super().__init__(OneDimensionalData(data_array, data_unit))
        elif isinstance(data_array, OneDimensionalData):
            super().__init__(deepcopy(data_array))
        else:
            raise ValueError(f"data_array has to be of type: np.ndarray or OneDimensionalData, not: {type(data_array)}")

        if isinstance(t_array, np.ndarray):
            self._t_axis = TimeAxis(t_array, t_unit)
        elif isinstance(t_array, TimeAxis):
            self._t_axis = deepcopy(t_array)
        else:
            raise ValueError(f"t_array has to be of type: np.ndarray or TimeAxis, not: {type(t_array)}")

        self._position = position

        self._check_dimensions()

    @property
    def position(self):
        return self._position

    @property
    def t(self):
        return self._t_axis

    @t.setter
    def t(self, array: np.ndarray):
        if isinstance(array, np.ndarray):
            self._t_axis = TimeAxis(array, self._t_axis.unit)
        elif isinstance(array, TimeAxis):
            self._t_axis = array
        else:
            raise ValueError(f"array must be of type np.ndarray or TimeAxis, not {type(array)}")

    def subtract(self, other: 'Transient'):
        other.interpolate_to(self.t)

        return Transient(self.t,
                         self.y - other.y,
                         self.t.unit,
                         self.y.unit)

    @inPlaceOp
    def average(self, width: int = 1):
        self._t_axis.array, self._data.array = self._reductive_average(self._t_axis.array, width)

        return self

    @inPlaceOp
    def sort(self):
        sorting_mask = self._t_axis.sort_array()
        self._data.sort_by(sorting_mask)

        return self

    @inPlaceOp
    def interpolate_to(self, t_axis: np.ndarray[float] or TimeAxis):
        if not np.all(np.diff(self._t_axis.array) > 0):
            raise ValueError("t-Axis not sorted for interpolation!")

        if isinstance(t_axis, type(self._t_axis)):
            self._data.array = np.interp(t_axis.array, self._t_axis.array, np.nan_to_num(self.y), left=np.nan, right=np.nan)
        elif isinstance(t_axis, np.ndarray):
            self._data.array = np.interp(t_axis, self._t_axis.array, np.nan_to_num(self.y), left=np.nan, right=np.nan)
        else:
            raise ValueError(f"array must be of type EnergyAxis or WavelengthAxis, or be a np.ndarray, not {type(t_axis)}")

        return self

    def save(self, path: str):
        self._save_one_dimension(self._t_axis.array, self._data.array, path)

        return self

    @inPlaceOp
    def eliminate_idx(self, t_idx: list[int] or int or None = None):
        if t_idx is not None:
            self._t_axis.array, self._data.array = self._eliminate_pos_1d(self._t_axis.array, t_idx)

        return self

    @inPlaceOp
    def eliminate_repetition(self):
        self._t_axis.array, self._data.array = self._eliminate_repetition_1d(self._t_axis.array)

        return self

    @inPlaceOp
    def truncate_to(self, t_range: list[float] or None = None):

        # TODO: IMPLEMENT SLICING
        t_range = [self.t.closest_to(t)[0] for t in t_range]
        self._t_axis.array, self._data.array = self._truncate_one_dimension(t_range, self._t_axis.array)

        return self

    @inPlaceOp
    def truncate_like(self, array: np.ndarray[float] or AbstractAxis = None):

        if array is not None:
            self._t_axis.array, self._data.array = self._truncate_like_array_one_dimension(array, self._t_axis.array)

        return self

    def _check_dimensions(self):
        if not self._data.length == len(self._t_axis):
            raise ValueError(f"x_array and data_array need to have the same size but have lengths of:"
                             f"{len(self._t_axis)} and {self._data.length}!")

    @classmethod
    def average_from(cls, spec_list: list['Transient']):
        y = []

        for item in spec_list:
            y.append(item.y.array)

        if len(y) > 1:
            y = np.nanmean(y, axis=0)
        elif len(y) == 1:
            y = y[0]

        return cls(spec_list[0].t.array,
                   y,
                   spec_list[0].t.unit,
                   spec_list[0].y.unit)

    @staticmethod
    def from_file(path: str, parser_args: dict or None = None):
        from pySpec.SpecCore.coreParser import Parser

        args = {'t_unit': 's', 'data_unit': 'od'}
        if parser_args is not None:
            args.update(parser_args)

        return Parser(path,  import_type='transient', **args)[0]


if __name__ == '__main__':
    pass
