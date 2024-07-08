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
import numpy as np


class AbstractAxis(ABC):
    """
    :param array: The Array to be wrapped within the class. Has to be passed to the class by the constructor of the
        subclass.

    Abstract Base Class for creating axes, utilised in the :class:`Spectrum`, :class:`Transient` and
    :class:`TransientSpectrum` classes. Implements most of the arithmetic interface and indexing, to be used like numpy
    arrays, though the arithmetic and indexing methods returns the array directly and the array itself can also be
    accessed directly by the `.array` property.
    Supports addition, subtraction, multiplication and division.
    """

    __slots__ = ("_array", "_unit")

    def __init__(self, array: np.ndarray, unit: str):

        if len(array.shape) != 1:
            raise ValueError("Axis Object only accepts one-dimensional arrays.")

        self._array = array.copy()
        self._unit = unit

    def __getitem__(self, idx):
        return self._array[idx]

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return np.size(self._array)

    def __neg__(self):
        return self.__class__(- self._array, self._unit)

    def __add__(self, other):
        return self.shift_by(other, anchor=None)

    __radd__ = __add__

    def __sub__(self, other):
        return self.shift_by(- other, anchor=None)

    __rsub__ = __sub__

    def __mul__(self, other):
        return self.__class__(self._array * other, self._unit)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.__class__(self._array / other, self._unit)

    def __rtruediv__(self, other):
        return self.__class__(other / self._array, self._unit)

    @property
    def size(self):
        """
        The length of the axis.
        """
        return np.size(self._array)

    @property
    def array(self):
        """
        The underlying np.array. Can be overwritten.
        """
        return self._array

    @array.setter
    def array(self, array):
        """
        Overwrites the underlying np.array.
        """
        self._array = array.copy()

    @property
    def unit(self):
        """
        The unit specified at the point of axis creation.
        """
        return self._unit

    def closest_to(self, value):
        """
        Returns the closest match of the axis' array with `value` as a tuple of the index and the matching value.

        :param value: The value to search for.
        :type value: float
        :return: tuple( index, value )
        """
        idx = int((np.abs(self._array - value)).argmin())
        return idx, self._array[idx]

    def sort_array(self) -> np.ndarray:
        """
        Sorts the array in increasing order and return the sorting mask.

        :return: Sorting mask.
        """
        sorting_mask = np.argsort(self._array)

        self._array = self._array[sorting_mask]
        return sorting_mask

    @abstractmethod
    def shift_by(self, amount: float, anchor: float or None):
        """
        Abstract method to shift an axis by a certain amount. Necessary for EnergyAxis, since shifting the axis measured
        with gratings is only linear in the Wavelength domain. Shifts inplace!

        :param amount: Amount to shift by.
        :type amount: float
        :param anchor: Position to shift around.
        :type anchor: float
        """
        pass

    @abstractmethod
    def convert_to(self, axis_type):
        """
        Abstract conversion function to convert axis into another axis type.

        :param axis_type: Axis type to convert to.
        :type axis_type: str
        :return: New Axis Object.
        """
        pass
