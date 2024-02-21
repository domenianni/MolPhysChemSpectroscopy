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
from copy import deepcopy
from scipy.integrate import trapezoid
from scipy.signal import convolve

from Spectroscopy.SpecCore.SpecCoreData.coreAbstractData import AbstractData
from Spectroscopy.SpecCore.coreLineShapes import gaussian


class AbstractSpectrum(ABC):
    """
    :param data: Spectrum data as a subclass of AbstractData.

    Abstract class for all Spectrum classes. Spectrum classes hold a combination of intensity values, as a subclass of
    AbstractData and one or more corresponding axes conferring spectral/temporal information.

    The AbstractSpectrum class supplies methods to save the data and axis, to truncate, eliminate repeated values,
    eliminate requested indices and average reductively in ONE DIMENSION.
    Additionally, a method to convolve the data with a gaussian function is implemented, to increase the FWHM of the
    data.

    Spectrum classes must implement the following methods:
        - sorting of the data according to their axes
        - saving and loading of the data in an ascii format
        - checking the dimensionality of the supplied data/axis aggregate
    """

    def __init__(self, data):
        if not isinstance(data, AbstractData):
            raise ValueError(f"data must be of type AbstractData, not {type(data)}!")

        self._data = data

    @property
    def y(self):
        """
        The data object. Can be overwritten with a new array.
        """
        return self._data

    @y.setter
    def y(self, array: np.ndarray):
        if not isinstance(array, np.ndarray) and not isinstance(array, AbstractData):
            raise ValueError(f"data can only be ndarray or AbstractData but is {type(array)}!")

        if isinstance(array, AbstractData):
            self._data = deepcopy(array)
        else:
            self._data.array = array.copy()

    @abstractmethod
    def _check_dimensions(self):
        pass

    @abstractmethod
    def sort(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    def _save_one_dimension(x, y, path):
        with open(path, 'w') as file:

            for value in x:
                file.write(f"\t{value}")

            file.write("\n")

            for value in y:
                file.write(f"{value}\t")

    @staticmethod
    def calculate_od(y: np.ndarray,
                     y_ref: np.ndarray) -> np.ndarray:

        return - np.log10(np.divide(y, y_ref))

    @abstractmethod
    def eliminate_repetition(self):
        pass

    def _inplace(self, flag: bool = True):
        if not flag:
            return deepcopy(self)
        return self

    def _eliminate_pos_1d(self, x, idx):
        mask = np.ones_like(x, dtype=bool)
        mask[idx] = False

        return x[mask], self.y[mask]

    def _eliminate_repetition_1d(self, x):
        pos = []
        mask = [True]
        for i, t in enumerate(x[1:], start=1):
            if x[i - 1] == t:
                pos.append(i)
                mask.append(False)
            else:
                mask.append(True)

        for i in pos:
            self.y[i - 1] = np.nanmean(self.y[i - 1:i + 1], axis=0)

        return x[mask], self.y[mask]

    def _truncate_one_dimension(self, region, x):
        """Truncates INCLUSIVELY!!"""
        if all(isinstance(i, list) for i in region):
            x_temp = []
            y_temp = []

            for reg in region:
                x_temp.extend(x[slice(reg[0], reg[1]+1, None)])
                y_temp.extend(self.y[slice(reg[0], reg[1]+1, None)])

            x = np.array(x_temp)
            y = np.array(y_temp)

        elif not any(isinstance(i, list) for i in region):
            region = sorted(region)
            x = x[slice(region[0], region[1]+1, None)]
            y = self.y[slice(region[0], region[1]+1, None)]
        elif isinstance(region, slice):
            x = x[region]
            y = self.y[region]
        else:
            raise ValueError(f"region can only either be a list, a slice or a list of lists! Not {type(region)}")

        return x, y

    def _truncate_like_array_one_dimension(self, array, x):
        mask = np.logical_and(np.where((np.min(array) <= x), 1, 0),
                              np.where((x <= np.max(array)), 1, 0))
        return x[mask], self._data[mask]

    def _reductive_average(self, x, width):
        i = 0
        j = width - 1

        x_temp = []
        y_temp = []
        while i < np.size(x):
            s = slice(i, j, None)
            x_temp.append(np.nanmean(x[s]))
            y_temp.append(np.nanmean(self.y[s], axis=0))

            i += width
            j += width

        return np.array(x_temp), np.array(y_temp)

    @staticmethod
    def convolve_gaussian(x: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray:
        from scipy.interpolate import interp1d

        OVERSAMPLING = 10

        norm = trapezoid(np.nan_to_num(y), x)

        x_gauss = np.linspace(np.min(x), np.max(x), OVERSAMPLING * len(x))
        y_large = interp1d(x, np.nan_to_num(y))(x_gauss)

        y_gauss = gaussian(x_gauss - np.mean(x), 1, 0, fwhm)

        y_conv = convolve(y_gauss, y_large, mode='same')

        y = interp1d(x_gauss, y_conv)(x)

        y *= norm / trapezoid(np.nan_to_num(y), x)

        return y

    @staticmethod
    @abstractmethod
    def from_file(path, parser_args: dict):
        pass


if __name__ == '__main__':
    pass
