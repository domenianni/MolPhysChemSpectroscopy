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
from ..SpecCoreAxis.coreWavelengthAxis import WavelengthAxis
from ..SpecCoreAxis.coreEnergyAxis import EnergyAxis
from ..SpecCoreData import OneDimensionalData
from ..coreFunctions import inPlaceOp
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d


class Spectrum(AbstractSpectrum):
    """
    :param x_array:
    :param data_array:
    :param x_unit:
    :param data_unit:
    :param time:

    Implementation of a Spectrum. This class is an aggregate of a data object holding the intensity data and a
    corresponding energy axis (cf. the definition of a spectrum), which can be represented by a wavenumber, wavelength
    or electron volt axis.
    """

    def __init__(self,
                 x_array:    np.ndarray,
                 data_array: np.ndarray,
                 x_unit:     str  = 'wn',
                 data_unit:  str  = 'od',
                 time:       dict = None):

        if len(np.shape(data_array)) != 1:
            raise ValueError("Data Object only accepts one-dimensional arrays.")

        if isinstance(data_array, np.ndarray):
            super().__init__(OneDimensionalData(data_array.copy(), data_unit))
        elif isinstance(data_array, OneDimensionalData):
            super().__init__(deepcopy(data_array))
        else:
            raise ValueError(f"data_array has to be of type: np.ndarray or TwoDimensionalData, not: {type(data_array)}")

        if isinstance(x_array, np.ndarray):
            if x_unit in ('wn', 'ev'):
                self._x_axis = EnergyAxis(x_array, x_unit)
            elif x_unit == 'wl':
                self._x_axis = WavelengthAxis(x_array, x_unit)

        elif isinstance(x_array, AbstractAxis):
            self._x_axis = deepcopy(x_array)
        else:
            raise ValueError(f"x_array has to be of type: np.ndarray or EnergyAxis, not: {type(x_array)}")

        self._time = time

        self._check_dimensions()

    @property
    def time(self):
        """
        Returns the timestep at which this spectrum lies. Only useful if assigned at object creation.
        """
        return self._time

    @property
    def x(self):
        """
        The x-Axis instance of the spectrum. Either WavelengthAxis or EnergyAxis. Can be set to a new array,
        of the same size, which is then incorporated into the Axis Object.
         """
        return self._x_axis

    @x.setter
    def x(self, array):
        if isinstance(array, np.ndarray):
            self._x_axis = EnergyAxis(array, self._x_axis.unit)
            return
        if isinstance(array, WavelengthAxis) or isinstance(array, EnergyAxis):
            self._x_axis = deepcopy(array)
            return

        raise ValueError(f"array must be of type np.ndarray, not {type(array)}")

    @inPlaceOp
    def average(self, x_width: int = 2):
        """
        :param x_width: The step size in integers for reductive averaging.
        :type x_width: int

        Reduces the resolution of the spectrum via reductive averaging inplace.
        """
        self.x, self.y = self._reductive_average(self.x, x_width)

        return self

    @inPlaceOp
    def sort(self):
        """
        Sorts the x-axis and the data in ascending fashion.
        """
        sorting_mask = self._x_axis.sort_array()
        self._data.sort_by(sorting_mask)

        return self

    @inPlaceOp
    def interpolate_to(self, x_axis: EnergyAxis or WavelengthAxis or np.ndarray):
        """
        Interpolates the data to a new x-axis and overwrites the existing x- and y-axis. If the x-axis is
        out-of-bounds of the old x-axis all missing values are set as `np.nan`.
        """

        f = interp1d(self.x, np.nan_to_num(self.y), bounds_error=False, fill_value=np.nan)

        if isinstance(x_axis, EnergyAxis) and not isinstance(x_axis, WavelengthAxis):
            self.y = f(x_axis.array)
        elif isinstance(x_axis, np.ndarray):
            self.y = f(x_axis)
        else:
            raise ValueError(f"array must be of type EnergyAxis or WavelengthAxis, or be a np.ndarray, not {type(x_axis)}")

        self.x = x_axis

        return self

    @inPlaceOp
    def truncate_to(self, x_range: list = None):
        """
        :param x_range: List of two values denoting the lower and upper bound to truncate to.
        :param inplace: Modify values inplace or return a new instance.
        """

        x_range = [self.x.closest_to(x)[0] for x in x_range]
        self.x, self.y = self._truncate_one_dimension(x_range, self.x)

        return self

    @inPlaceOp
    def truncate_like(self, x_array):
        """
        :param x_array: Reference to truncate the spectrum to.
        :type x_array: array_like
        :param inplace: Selects whether to change the existing object or return a new one. Defaults to `True`.

        Truncate the data according to the supplied `x_array`.
        """
        self.x, self.y = self._truncate_like_array_one_dimension(x_array, self.x)

        return self

    def save(self, path):
        """
        :param path: The path to save at.

        Saves the spectrum as an ascii-formatted file.
        """
        self._save_one_dimension(self.x, self.y, path)

        return self

    @inPlaceOp
    def eliminate_repetition(self):
        self.x, self.y = self._eliminate_repetition_1d(self.x)

        return self

    @inPlaceOp
    def eliminate_idx(self, x_idx=None):
        if x_idx is not None:
            self.x, self.y = self._eliminate_pos_1d(self.x, x_idx)

        return self

    def _check_dimensions(self):
        if not self._data.length == len(self._x_axis):
            raise ValueError(f"x_array and data_array need to have the same size but have lengths of:"
                             f"{len(self._x_axis)} and {self._data.length}!")

    @classmethod
    def average_from(cls, spec_list: list):
        y = []

        for item in spec_list:
            y.append(item.y.array)

        if len(y) > 1:
            y = np.nanmean(y, axis=0)
        elif len(y) == 1:
            y = y[0]

        return cls(spec_list[0].x.array,
                   y,
                   spec_list[0].x.unit,
                   spec_list[0].y.unit)

    @classmethod
    def calculate_from(cls, spec, ref):
        return cls(spec.x,
                   cls.calculate_od(spec.y, ref.y),
                   x_unit=spec.x.unit,
                   data_unit=spec.y.unit
                   )

    @staticmethod
    def from_file(path, parser_args: dict = None):
        from ..coreParser import Parser

        path = Parser.parse_path(path)[0]

        args = {'x_unit': 'wn', 'data_unit': 'od'}

        if isinstance(parser_args, dict):
            args.update(parser_args)

        return Parser(path, import_type='spectrum', **args)[0]

    @classmethod
    def average_from_files(cls, path, parser_args: dict = None):
        from ..coreParser import Parser

        args = {'x_unit': 'wn', 'data_unit': 'od'}

        if parser_args is not None:
            args.update(parser_args)

        spectra = [Parser(p, import_type='spectrum', **args)[0] for p in Parser.parse_path(path)]

        return cls.average_from(spectra)


    @classmethod
    def calculate_from_files(cls, path, solvent_path, parser_args: dict = None):
        from ..coreParser import Parser

        args = {'x_unit': 'wn', 'data_unit': 'od'}

        if parser_args is not None:
            args.update(parser_args)

        parent_spectra = [Parser(p, import_type='spectrum', **args)[0] for p in Parser.parse_path(path)]
        solvent_spectra = [Parser(p, import_type='spectrum', **args)[0] for p in Parser.parse_path(solvent_path)]

        parent = Spectrum.average_from(parent_spectra)

        if len(solvent_spectra) > 0:
            solvent = Spectrum.average_from(solvent_spectra)
            solvent.interpolate_to(parent.x)
            parent.y = cls.calculate_od(parent.y, solvent.y)

        return parent


if __name__ == '__main__':
    pass
