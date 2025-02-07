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

from ..SpecCoreData.coreAbstractData import AbstractData
from ..SpecCoreData.coreTwoDimensionalData import TwoDimensionalData
from ..SpecCoreData.coreOneDimensionalData import OneDimensionalData

from ..SpecCoreAxis.coreAbstractAxis import AbstractAxis
from ..SpecCoreAxis.coreTimeAxis import TimeAxis
from ..SpecCoreAxis.coreEnergyAxis import EnergyAxis
from ..SpecCoreAxis.coreWavelengthAxis import WavelengthAxis

from ..coreLineShapes import gaussian


class AbstractSpectrum(ABC):
    """
    Abstract class for all Spectrum classes. Spectrum classes hold a combination of intensity values as a subclass of
    AbstractData and one or more corresponding axes conferring spectral/temporal information.
    """

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

    def __init__(self, data: AbstractData):
        if not isinstance(data, AbstractData):
            raise ValueError(f"data must be of type AbstractData, not {type(data)}!")

        self._data = data

    @staticmethod
    def _initialize_axis(array: np.ndarray or AbstractAxis, unit: str) -> AbstractAxis:
        """
        Initializes the axis object based on the input array and unit.

        :param array: A numpy array or AbstractAxis to initialize.
        :param unit: The unit for the axis, e.g., 'wn', 'ev', 'wl', or time units.
        :return: An appropriate AbstractAxis object based on the array and unit.
        """

        if isinstance(array, np.ndarray):
            if unit in ('wn', 'ev'):
                return EnergyAxis(array, unit)
            elif unit == 'wl':
                return WavelengthAxis(array, unit)
            elif unit[-1] == 's':
                return TimeAxis(array, unit)
            else:
                raise ValueError(f"unit has to be 'wn', 'ev', 'wl', or '<T>s' not {unit}")
        elif isinstance(array, AbstractAxis):
            return deepcopy(array)
        else:
            raise ValueError(f"array has to be of type: np.ndarray or be derived from AbstractAxis, not: {type(array)}")

    @staticmethod
    def _set_array(array, old_axis) -> AbstractAxis:
        """
        Sets the array values for the given axis object.

        :param array: The numpy array to set.
        :param old_axis: The existing axis object.
        :return: The updated axis object.
        """

        if isinstance(array, np.ndarray):
            old_axis.array = array.copy()
            return old_axis
        if isinstance(array, WavelengthAxis) or isinstance(array, EnergyAxis):
            return deepcopy(array)

        raise ValueError(f"array must be of type np.ndarray, not {type(array)}")

    def __sub__(self, other):
        """
        Subtracts the spectrum from another spectrum using the `subtract` method.

        :param other: The spectrum to subtract.
        :return: The result of the subtraction.
        """

        return self.subtract(other)

    @abstractmethod
    def subtract(self, other):
        """
        Subtracts the data from another spectrum.

        :param other: The spectrum to subtract.
        :return: The resulting spectrum after subtraction.
        """

        pass

    @property
    def y(self):
        """
        Returns the data object containing the spectrum values.
        """
        return self._data

    @y.setter
    def y(self, array: np.ndarray[float] or AbstractData):
        """
        Sets the data object for the spectrum.

        :param array: The new data array or AbstractData object.
        :raises ValueError: If the array is not of type np.ndarray or AbstractData.
        """

        if not isinstance(array, np.ndarray) and not isinstance(array, AbstractData):
            raise ValueError(f"data can only be ndarray or AbstractData but is {type(array)}!")

        if isinstance(array, AbstractData):
            self._data = deepcopy(array)
        else:
            self._data.array = array

    @abstractmethod
    def _check_dimensions(self):
        """
        Abstract method to check the dimensions of the spectrum data.
        """

        pass

    @abstractmethod
    def sort(self):
        """
        Abstract method to sort the data according to the axis.
        """

        pass

    @abstractmethod
    def save(self, path):
        """
        Abstract method to save the spectrum data to a specified path.

        :param path: Path where the data should be saved.
        """

        pass

    @staticmethod
    def _save_one_dimension(x: np.ndarray[float], y: np.ndarray[float], path: str) -> None:
        """
        Saves the 1D data arrays (x, y) to an ASCII file at the given path.

        :param x: The axis values.
        :param y: The corresponding data values.
        :param path: Path, including file name and suffix, where to save the data.
        :return: None
        """
        with open(path, 'w') as file:

            for xv, yv in zip(x, y):
                file.write(f"{xv}\t{yv}\n")

            # for value in x:
            #     file.write(f"\t{value}")
#
            # file.write("\n")
#
            # for value in y:
            #     file.write(f"{value}\t")

    @staticmethod
    def calculate_od(y: np.ndarray[float] or float,
                     y_ref: np.ndarray[float] or float) -> np.ndarray[float] or float:
        """
        Calculates the optical density (OD) based on the given data and reference values.

        :param y: The measured spectrum values.
        :param y_ref: The reference spectrum values.
        :return: The optical density (OD).
        """

        return - np.log10(np.divide(y, y_ref))

    @abstractmethod
    def eliminate_repetition(self):
        """
        Abstract method to eliminate repeated data values.
        """

        pass

    def _eliminate_pos_1d(self, x: np.ndarray, idx: list[int] or int) -> (np.ndarray[float], np.ndarray[float]):
        """
        Removes the data at the specified indices (idx) from the spectrum.

        :param x: The axis values.
        :param idx: List or single index of values to remove.
        :return: The new axis and data arrays with the specified values removed.
        """

        mask = np.ones_like(x, dtype=bool)
        mask[idx] = False

        return x[mask], self._data.array[mask]

    def _eliminate_repetition_1d(self, x: np.ndarray[float]) -> (np.ndarray[float], np.ndarray[float]):
        """
        Eliminates repeated consecutive values in the spectrum data.

        :param x: The axis values.
        :return: The axis and data arrays with repeated values removed.
        """

        pos = []
        mask = [True]
        for i, t in enumerate(x[1:], start=1):
            if x[i - 1] == t:
                pos.append(i)
                mask.append(False)
            else:
                mask.append(True)

        for i in pos:
            self._data.array[i - 1] = np.nanmean(self._data.array[i - 1:i + 1], axis=0)

        return x[mask],  self._data.array[mask]

    def _truncate_one_dimension(self, region, x: np.ndarray) -> (np.ndarray[float], np.ndarray[float]):
        # TODO Should this be split up and use truncation within the axis and data objects?
        """
        Truncates the spectrum data in the specified region, inclusively.

        :param region: The region(s) to truncate (either a list of indices or a slice).
        :param x: The axis values.
        :return: The truncated axis and data arrays.
        """

        if all(isinstance(i, list) for i in region):
            x_temp = []
            y_temp = []

            for reg in region:
                x_temp.extend(x[slice(reg[0], reg[1]+1, None)])
                y_temp.extend(self._data.array[slice(reg[0], reg[1]+1, None)])

            x = np.array(x_temp)
            y = np.array(y_temp)

        elif not any(isinstance(i, list) for i in region):
            region = sorted(region)
            x = x[slice(region[0], region[1]+1, None)]
            y = self._data.array[slice(region[0], region[1]+1, None)]
        elif isinstance(region, slice):
            x = x[region]
            y = self._data.array[region]
        else:
            raise ValueError(f"region can only either be a list, a slice or a list of lists! Not {type(region)}")

        return x, y

    def _truncate_like_array_one_dimension(self, array: np.ndarray[float], x: np.ndarray[float]) -> (np.ndarray[float], np.ndarray[float]):
        """
        Truncates the spectrum data to match the range of the given array.

        :param array: The reference array for truncation.
        :param x: The axis values.
        :return: The truncated axis and data arrays.
        """

        mask = np.logical_and(np.where((np.min(array) <= x), 1, 0),
                              np.where((x <= np.max(array)), 1, 0))
        return x[mask], self._data.array[mask]

    def _reductive_average(self, x: np.ndarray[float], width: int) -> (np.ndarray[float], np.ndarray[float]):
        """
        Computes a reductive average of the data with the specified width.

        :param x: The axis values.
        :param width: The width of the window for averaging.
        :return: The averaged axis and data arrays.
        """

        i = 0
        j = width - 1

        x_temp = []
        y_temp = []
        while i < np.size(x):
            s = slice(i, j, None)
            x_temp.append(np.nanmean(x[s]))
            y_temp.append(np.nanmean(self._data.array[s], axis=0))

            i += width
            j += width

        return np.array(x_temp), np.array(y_temp)

    @staticmethod
    def convolve_gaussian(x: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray[float]:
        """
        Convolves the data with a Gaussian function of the specified FWHM.

        :param x: The axis values.
        :param y: The data values.
        :param fwhm: The full width at half maximum of the Gaussian function.
        :return: The convolved data.
        """

        if not np.all(np.diff(x) > 0):
            raise ValueError("X-Axis not sorted for interpolation!")

        OVERSAMPLING = 10

        norm = np.nanmax(y)

        x_gauss = np.linspace(np.min(x), np.max(x), OVERSAMPLING * len(x))
        y_large = np.interp(x_gauss, x, np.nan_to_num(y), left=np.nan, right=np.nan)

        y_gauss = gaussian(x_gauss - np.mean(x), 1, 0, fwhm)

        y_conv = convolve(y_gauss, y_large, mode='same')

        y_new = np.interp(x, x_gauss, y_conv, left=np.nan, right=np.nan)

        y_new *= norm / np.nanmax(y_new)

        return y_new

    @staticmethod
    @abstractmethod
    def from_file(path, parser_args: dict):
        """
        Abstract method to load data from a file.

        :param path: The path to the file.
        :param parser_args: Additional arguments for parsing the file.
        :return: The spectrum object.
        """

        pass


if __name__ == '__main__':
    pass
