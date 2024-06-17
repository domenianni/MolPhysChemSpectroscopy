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
from ..SpecCoreData.coreOneDimensionalData import OneDimensionalData
from ..SpecCoreAxis.coreWavelengthAxis import WavelengthAxis
from ..SpecCoreAxis.coreEnergyAxis import EnergyAxis

import numpy as np
from warnings import warn


class StickSpectrum(AbstractSpectrum):
    """
    :param positions:
    :param intensities:
    :param pos_unit:
    :param int_unit:

    Class to represent the spectrum resulting from quantum chemical calculations. The usual output contains only
    positions and intensities for the calculated transitions, which can be accessed directly by the attributes .pos and
    .int, as well as a list of tuples containing additionally the root number via .roots.
    """

    def __init__(self,
                 positions: np.ndarray,
                 intensities: np.ndarray,
                 pos_unit: str,
                 int_unit: str):

        warn(DeprecationWarning)

        if len(np.shape(intensities)) != 1:
            raise ValueError("Data Object only accepts one-dimensional arrays.")

        super().__init__(OneDimensionalData(intensities, int_unit))

        if pos_unit in ('wn', 'ev'):
            self._pos_axis = EnergyAxis(positions, pos_unit)
        if pos_unit == 'wl':
            self._pos_axis = WavelengthAxis(positions, pos_unit)

        self._check_dimensions()

    @property
    def pos(self):
        return self._pos_axis

    @property
    def int(self):
        return self._data

    @property
    def roots(self):
        return [(i + 1, pos, self._data[i]) for i, pos in enumerate(self._pos_axis.array)]

    def __repr__(self):
        return str([(i + 1, pos, self._data[i]) for i, pos in enumerate(self._pos_axis.array)])

    def __getitem__(self, item):
        if isinstance(item, int):
            return item, self._pos_axis[item-1], self._data[item-1]

        if isinstance(item, slice):
            return [ (i+1, self._pos_axis[i], self._data[i]) for i in range(*item.indices(len(self._pos_axis))) ]

        if isinstance(item, str):
            idx = self._pos_axis.closest_to(float(item))
            return idx + 1, self._pos_axis[idx], self._data[idx]

    def _check_dimensions(self):
        if not self._data.length == len(self._pos_axis):
            raise ValueError(f"x_array and data_array need to have the same size but have lengths of:"
                             f"{len(self._pos_axis)} and {self._data.length}!")

    def sort(self):
        sorting_mask = self._pos_axis.sort_array()
        self._data.sort_by(sorting_mask)

        return self

    def save(self, path):
        self._save_one_dimension(self._pos_axis, self.y, path)

        return self

    def eliminate_repetition(self):
        self._pos_axis.array, self._data.array = self._eliminate_repetition_1d(self._pos_axis)

        return self

    @staticmethod
    def from_file(path, parser_args: dict):
        pass


if __name__ == '__main__':
    pass
