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

import numpy as np
from math import floor
import warnings

from ..SpecCoreAxis.coreAbstractAxis import AbstractAxis
from ..SpecCoreData.coreAbstractData import AbstractData
from ..SpecCoreAxis.coreTimeAxis import TimeAxis
from ..SpecCoreAxis.coreWavelengthAxis import WavelengthAxis
from ..SpecCoreAxis.coreEnergyAxis import EnergyAxis

from .coreSpectrum import Spectrum
from .coreTransient import Transient


class SliceFactory:
    """
    :param axis: Axis to slice data along.
    :param data: Data to be sliced.
    :param names: The other axis, representing the position of the slices.

    This class slices the 2D data of TransientSpectrum instances and creates Transient / Spectra respectively. Handles
    slicing as a call for averaging. Does not need to be called directly.
    """

    def __init__(self, axis: AbstractAxis, data: AbstractData, names: AbstractAxis):
        self._axis = axis
        self._data = data
        self._names = names

    def __getitem__(self, item: str or slice or list or tuple):
        """
        :param item: The slice to get from this object.

        Operator[]
        Implements slicing of the data based on the Index and the spectral/temporal position.
        Supports additionally slice averaging, which can be accessed either via slice or a list/tuple. The latter either
        expects two integers for the indices between which to average, or a string (representing the center position in
        the requested dimension) followed by an integer (the width used for averaging).
        """

        # arrange data, so that the first dimension corresponds to the requested axis.
        if self._data.shape[0] == len(self._axis):
            self._data = self._data.transpose()

        data, name = self._process_item(item)

        if isinstance(self._axis, TimeAxis):
            return Transient(t_array=self._axis.array,
                             data_array=data,
                             t_unit=self._axis.unit,
                             data_unit=self._data.unit,
                             position=name)
        elif isinstance(self._axis, EnergyAxis) or isinstance(self._axis, WavelengthAxis):
            return Spectrum(x_array=self._axis.array,
                            data_array=data,
                            x_unit=self._axis.unit,
                            data_unit=self._data.unit,
                            time=name)
        else:
            raise TypeError()

    def _process_item(self, item: str or slice or list or tuple):
        """
        Called by Operator[] to select and average the requested slice.
        See Operator[] for details.
        """

        # Check type of item from operator[]
        if isinstance(item, int):
            data = self._data[item]
            name = self._names[item]

        elif isinstance(item, slice):
            idx = []

            for i in range(*item.indices(len(self._names))):
                idx.append(i)

            data = np.nanmean(self._data[idx], axis=0)
            name = np.nanmean(self._names[idx])

        elif isinstance(item, str):
            item = float(item)

            idx, _ = self._names.closest_to(item)
            data = self._data[idx]
            name = self._names[idx]

        elif isinstance(item, list) or isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("List of positions has to be only 2!")

            if isinstance(item[0], str):
                if (item[1] % 2) == 0:

                    warnings.warn(f" Even sized average windows do not center the average on the requested position!"
                                  f" The average is now centered between: "
                                  f" {self._names.closest_to(float(item[0]))[1]} and"
                                  f" {self._names[self._names.closest_to(float(item[0]))[0] + 1]}!",
                                  category=SyntaxWarning, stacklevel=0)

                idx = [i + self._names.closest_to(float(item[0]))[0] - floor(item[1]/2) for i in range(item[1])]
            else:
                idx = [self._axis.closest_to(float(x))[0] for x in item]
                idx = [x for x in range(idx[0], idx[1])]

            data = np.nanmean(self._data[idx], axis=0)
            name = np.nanmean(self._names[idx])

        else:
            raise TypeError(f"index must be 'int', 'str', 'slice' or a 'list of two positions' not {type(item)}!")

        unit = {'value': name, 'unit': self._names.unit}

        return data, unit


if __name__ == '__main__':
    pass
