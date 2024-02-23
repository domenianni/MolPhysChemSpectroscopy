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

from ..SpecCore import *
from copy import deepcopy


class Baseline:

    def __init__(self, spectrum: TransientSpectrum or Spectrum):
        self._reference = deepcopy(spectrum)
        self._reference.sort()

    def one_point_baseline(self, x_region, t_region):
        reference = self._reference.truncate_to(x_range=x_region, t_region=t_region, inplace=False)

        return np.mean(reference.y.array)

    def static_baseline(self, regions: list or None = None, t_region: list or None = None):
        reference = self._reference.truncate_to(x_range=regions, inplace=False)
        reference.average(x_width=len(reference.x))

        if t_region is not None and len(t_region) == 2:
            t_region.sort()
            reference.y = np.where(t_region[0] > self._reference.t.array, 0, reference.y.array)
            reference.y = np.where(t_region[1] < self._reference.t.array, 0, reference.y.array)

        if isinstance(reference, TransientSpectrum):
            y = np.repeat(reference.y, len(self._reference.x),
                          axis=np.shape(reference.y.array).index(np.size(reference.x.array))
                          )
            if np.shape(y) != np.shape(self._reference.y.array):
                y = y.T

            baseline = TransientSpectrum(self._reference.x,
                                         self._reference.t,
                                         y,
                                         self._reference.x.unit,
                                         self._reference.t.unit,
                                         self._reference.y.unit)
        else:
            y = np.repeat(reference.y, len(self._reference.x))
            baseline = Spectrum(self._reference.x,
                                y,
                                self._reference.x.unit,
                                self._reference.y.unit)

        return baseline

    def linear_baseline(self, regions: list[float] or None = None):
        from scipy.stats import linregress

        references = [self._reference.truncate_to(x_range=region, inplace=False) for region in regions]

        x = np.concatenate([ref.x.array for ref in references])
        y = np.concatenate([ref.y.array for ref in references])
        t = references[0].t.array

        y_bsl = []
        for i, ti in enumerate(t):
            slope, intercept, _, _, _ = linregress(x, y[:, i])
            y_bsl.append(slope * self._reference.x + intercept)

        return TransientSpectrum(self._reference.x,
                                 self._reference.t,
                                 np.array(y_bsl),
                                 self._reference.x.unit,
                                 self._reference.t.unit,
                                 self._reference.y.unit)

    def polynomial_baseline(self, regions: list[float] or None = None):
        reference = self._reference.truncate_to(x_region=regions, inplace=False)
        pass


if __name__ == '__main__':
    pass
