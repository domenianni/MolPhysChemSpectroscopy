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

from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum

from copy import deepcopy
import numpy as np


class ProductSpectrum(TransientSpectrum):

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._static_data.y *= value / self._amplitude
        self.y = self._calculate_product()

        self._amplitude = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._static_data.y += value - self._offset
        self._offset = value

        self.y = self._calculate_product()


    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value: float):
        if value < 0:
            raise ValueError("FWHM can only be positive!")

        self._fwhm = value

        self._static_data.y = self._amplitude * self._backup['static_data'].y.array.copy() + self._offset

        if value > 0:
            self._static_data.y = self.convolve_gaussian(self._static_data.x.array,
                                                         self._static_data.y.array,
                                                         self._fwhm)
        self._data.array = self._calculate_product()

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, value):
        self._shift = value

        self._transient_data = deepcopy(self._backup['transient_data'])
        self._transient_data.x += self._shift
        self._data.array = self._calculate_product()
        self._x_axis.array = self._transient_data.x.array

    @property
    def transient_data(self):
        return self._transient_data

    @property
    def static_data(self):
        return self._static_data.truncate_like(self._x_axis.array, inplace=False)

    def __init__(self,
                 transient_data: TransientSpectrum,
                 static_data: Spectrum):

        transient_data.orient_data('x')

        self._offset = 0
        self._amplitude = 1
        self._fwhm = 0
        self._shift = 0

        self._transient_data = deepcopy(transient_data)
        self._static_data = deepcopy(static_data)

        self._backup = {'transient_data': deepcopy(transient_data), 'static_data': deepcopy(static_data)}

        super().__init__(
            self._transient_data.x,
            self._transient_data.t,
            self._calculate_product(),
            self._transient_data.x.unit,
            self._transient_data.t.unit,
            'ddmod'
        )

    def _calculate_product(self) -> np.ndarray[float]:
        return (
                self._transient_data.y + self._static_data.interpolate_to(self._transient_data.x, inplace=False).y
        ).array
