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

from .coreAbstractData import AbstractData
from ..coreFunctions import inPlaceOp

import numpy as np


class OneDimensionalData(AbstractData):
    """
    :param array: The underlying array. Is copied in the Data structure.
    :param unit: The unit of the data represented by the object.

    The specific wrapper around a one-dimensional array holding the intensity values of spectra or time traces.
    Implements sorting. This class should only be used within the context of the SpecCoreSpectrum classes.
    """

    ndim = 1

    def __init__(self, array, unit):
        if len(np.shape(array)) != 1:
            raise TypeError("data_array must have only one dimension!")

        super().__init__(array, unit)

    @inPlaceOp
    def sort_by(self, sorting_mask):
        """
        :param sorting_mask: The mask to sort the data by. The mask must be of the same size as the array and contain
                             the indices to be sorted to.

        Sorts the data according to an external corresponding axis, eg. time or wavenumber.
        """
        if np.shape(sorting_mask) != np.shape(self._array):
            raise TypeError(f"sorting_mask must have the same dimensions as data_array but has dimensions "
                            f"{np.shape(sorting_mask)}; {np.shape(self._array)}")

        self._array = self._array[sorting_mask]

        return self

    def truncate_to(self, region=None):
        pass


if __name__ == '__main__':
    pass
