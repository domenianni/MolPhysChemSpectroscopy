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

Based on the work of Gabriel F.Dorlhiac : https://github.com/gadorlhiac/PyLDM
                                          Access: 13.06.2025
                                          DOI: 10.1371/journal.pcbi.1005528
And David Ehrenberg: https://github.com/deef128/trtoolbox
                    Access: 13.06.2025
"""

import numpy as np
from pySpec.SpecCore.SpecCoreAxis.coreAbstractAxis import AbstractAxis
from abc import ABC, abstractmethod


class LdaAxis(AbstractAxis):

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self.updated = True

        self._n = self._validate_n(value)
        self._array = self._calc_array()

    @staticmethod
    def _validate_n(value):
        if value <= 0:
            ValueError("n must be positive!")
        return value

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, values):
        self.updated = True

        self._limits = self._validate_limits(values)
        self._array = self._calc_array()

    @staticmethod
    def _validate_limits(values):
        if np.any(np.array(values) < 0) or len(set(values)) != 2:
            ValueError("tau_limit set incorrectly!")

        return np.sort(values)

    def __init__(self, n: int = 100, limits = (1, 5)):
        self._n = self._validate_n(n)
        self._limits = self._validate_limits(limits)

        self.updated = False

        super().__init__(self._calc_array(), '')

    @abstractmethod
    def _calc_array(self):
        pass

    def shift_by(self, amount: float, anchor: float or None):
        pass

    def convert_to(self, axis_type):
        pass


class AlphaAxis(LdaAxis):

    def __init__(self, n: int = 100, limits = (1, 5), spacing: str ='log'):
        self._spacing = spacing

        super().__init__(n, limits)

    def _calc_array(self):
        if self._spacing == 'log':
            array = np.logspace(np.log10(self._limits[0]), np.log10(self._limits[1]), self._n)
        elif self._spacing == 'lin':
            array = np.linspace(self._limits[0], self._limits[1], self._n)
        else:
            raise ValueError("Spacing set incorrectly!")

        return array


class TauAxis(LdaAxis):

    def _calc_array(self):
        array = np.logspace(np.log10(self._limits[0]), np.log10(self._limits[1]), self._n)

        return array
