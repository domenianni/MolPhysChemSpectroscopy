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

from abc import ABC, abstractmethod
from warnings import warn

from ..coreFunctions import inPlaceOp

import numpy as np


class AbstractData(ABC):
    """
    :param array: The underlying array. Is copied in the Data structure.
    :param unit: The unit of the data represented by the object.

    The abstract object from which data objects inherit from. It wraps the array holding the data and implements some
    arithmetic operations: Addition, Subtraction, Multiplication, Division.
    Also works as an iterator and supports indexing.
    Additionally, implements basic calculation of extinction coefficients for the held data.
    """

    __slots__ = ('_array', '_unit')

    _label_mapping = {
        'mdod': r'$\Delta$mOD',
        'dod':  r'$\Delta$OD',
        'od':    'OD',
        'mod':   'mOD',
        'eps':  r'$\epsilon$ / M$^{-1}$cm$^{-1}$'
    }

    @property
    def label(self):
        return self._label_mapping.get(self._unit)

    @property
    def array(self) -> np.ndarray:
        """Returns the underlying numpy-array. Can be set to a new numpy array. This operation does not check sizes!"""
        return self._array

    @array.setter
    def array(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise ValueError(f"array must be of type np.ndarray, not {type(array)}")

        self._array = array.copy()

    @property
    def unit(self) -> str:
        """The unit of the data array."""
        return self._unit

    @property
    def length(self) -> int:
        """The length of the data array."""
        return self._array.size

    @property
    def shape(self) -> tuple[int]:
        """The shape of the data array."""
        return np.shape(self._array)

    def __init__(self, array: np.ndarray, unit: str):
        self._array = array.copy()
        self._unit = unit

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, key, value):
        self._array[key] = value

    def __len__(self):
        return np.shape(self._array)[0]

    def __add__(self, other):
        other = self._reshape_other(other)
        return self.__class__(self._array + other, self._unit)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._reshape_other(other)
        return self.__class__(self._array - other, self._unit)

    def __rsub__(self, other):
        other = self._reshape_other(other)
        return self.__class__(other - self._array, self._unit)

    def __mul__(self, other):
        other = self._reshape_other(other)
        return self.__class__(self._array * other, self._unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._reshape_other(other)
        return self.__class__(self._array / other, self._unit)

    def __rtruediv__(self, other):
        other = self._reshape_other(other)
        return self.__class__(other / self._array, self._unit)

    def __neg__(self):
        return self.__class__(- self._array, self._unit)

    def _reshape_other(self, other: np.ndarray) -> np.ndarray:
        """
        :param other: Array to reshape.
        :return: Reshaped 'other'-array to fit the internal data array in its dimensionality.
        """
        if len(np.shape(other)) > 0 and np.shape(other) != np.shape(self._array):
            if np.shape(other)[::-1] == np.shape(self._array):
                other = np.transpose(other)
            else:
                other = other[:, np.newaxis]

        return other

    @inPlaceOp
    def calc_ex_coeff(self, layer_thickness: float, concentration: float) -> np.ndarray:
        """
        :param layer_thickness: Cell layer thickness in cm
        :param concentration: Concentration of substance in mol/L
        :returns: the extinction coefficients calculated by lambert-beers law.
        """
        if self._unit == 'eps':
            warn("Extinction Coefficient has already been calculated!")

        self._unit = 'eps'

        self._array = self._array / (layer_thickness * concentration)

        return self

    @inPlaceOp
    def calc_ex_coeff_mv(self, layer_thickness: float, mass: float, molar_mass: float, volume: float) -> np.ndarray:
        """
         :param layer_thickness: Cell layer thickness in cm
         :type layer_thickness: float
         :param mass: Mass of the solved compound.
         :param molar_mass: Molar mass of the solved compound
         :param volume: Volume of the solvent
         :returns: The extinction coefficients calculated by lambert-beers law.
        """
        return self.calc_ex_coeff(layer_thickness, (mass / molar_mass) / volume)

    @abstractmethod
    def sort_by(self, sorting_mask):
        """Subclasses need to implement a sorting function."""
        pass

    @abstractmethod
    def truncate_to(self, region=None):
        """Subclasses need to implement a function to truncate the data."""
        pass


if __name__ == '__main__':
    pass
