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


class TwoDimensionalData(AbstractData):
    """
    :param array: The underlying array. Is copied in the Data structure.
    :param unit: The unit of the data represented by the object.

    The specific wrapper around a two-dimensional array holding the intensity values of transient spectra or 2D-spectra.
    Implements sorting. This class should only be used within the context of the SpecCoreSpectrum classes.
    """

    ndim = 2

    def __init__(self, array, unit):
        if len(np.shape(array)) != 2:
            raise TypeError(f"data_array must have only two dimensions, but has dimensions {np.shape(array)}!")

        super().__init__(array, unit)

    @inPlaceOp
    def sort_by(self, sorting_mask: np.ndarray):
        """
        :param sorting_mask: The mask to sort the data by. The mask must be of the same size and dimensionality as the
                             array and contain the indices to be sorted to.

        Sorts the data according to two external corresponding axis, eg. time/wavenumber or wavenumber/wavenumber.
        """

        if len(np.shape(sorting_mask)) != 1:
            raise ValueError("sorting mask must be an array of dimensionality 1!")

        if not np.size(sorting_mask) in np.shape(self._array):
            raise TypeError(f"sorting_mask must have the same size as one dimension of data_array but has size"
                            f"{np.shape(sorting_mask)}; {np.shape(self._array)}")

        # search for the dimension where the sorting mask has the same length as the data array
        shape_index = np.shape(self._array).index(sorting_mask.size)

        if shape_index != 0:
            self.transpose()

        self._array = self._array[sorting_mask]

        return self

    @inPlaceOp
    def transpose(self):
        """
        :param inplace: Whether to copy the data or modify the data inplace.

        Transposes the data.
        """

        self._array = self._array.T
        return self

    def truncate_to(self, region=None):
        pass


if __name__ == '__main__':
    pass
