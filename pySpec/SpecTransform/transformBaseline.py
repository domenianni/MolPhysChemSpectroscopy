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
    """
    Class for performing baseline correction on spectral data.

    :param spectrum: The spectrum (either `TransientSpectrum` or `Spectrum`) to perform baseline correction on.
    """

    def __init__(self, spectrum: TransientSpectrum or Spectrum):
        self._reference = deepcopy(spectrum)
        self._reference.sort()

    def one_point_baseline(self, x_region, t_region):
        """
        Computes a baseline using the mean of the spectrum within the given regions.

        :param x_region: The x-axis range (region) over which to compute the baseline.
        :param t_region: The time region over which to compute the baseline.
        :return: The mean of the spectrum values within the specified region.
        """
        reference = self._reference.truncate_to(x_range=x_region, t_region=t_region, inplace=False)

        return np.mean(reference.y.array)

    def static_baseline(self, regions: list or None = None, t_region: list or None = None):
        """
        Computes a static baseline, which is an average value of the spectrum across the entire x-axis.

        :param regions: Optional x-axis range to restrict the baseline calculation to.
        :param t_region: Optional time region to restrict the baseline calculation to.
        :return: A new spectrum object with the baseline data.
        """

        # Truncates the spectrum to the specified x-range (if any).
        reference = self._reference.truncate_to(x_range=regions, inplace=False)
        # Averages the spectrum across the entire x-axis.
        reference.average(x_width=len(reference.x))

        # If a time region is specified, apply zeroing of the y-values outside the region.
        if t_region is not None and len(t_region) == 2:
            t_region.sort()
            reference.y = np.where(t_region[0] > self._reference.t.array, 0, reference.y.array)
            reference.y = np.where(t_region[1] < self._reference.t.array, 0, reference.y.array)

        # If the reference spectrum is a TransientSpectrum, repeat the y-values to match the x-axis length.
        if isinstance(reference, TransientSpectrum):
            y = np.repeat(reference.y, len(self._reference.x),
                          axis=np.shape(reference.y.array).index(np.size(reference.x.array))
                          )

            # If the shapes don't match, transpose the y-values.
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
        """
        Computes a linear baseline correction using linear regression.

        :param regions: A list of x-axis regions over which to compute the linear regression.
        :return: A new `TransientSpectrum` object with the linear baseline.
        """
        from scipy.stats import linregress

        if not np.iterable(regions[0]):
            regions = [regions]

        references = [self._reference.truncate_to(x_range=region, inplace=False) for region in regions]

        x = np.concatenate([ref.x.array for ref in references])
        y = np.concatenate([ref.y.array for ref in references])

        if isinstance(self._reference, TransientSpectrum):
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

        slope, intercept, _, _, _ = linregress(x, y)

        return Spectrum(self._reference.x,
                        np.array(slope * self._reference.x + intercept),
                        self._reference.x.unit,
                        self._reference.y.unit)


    def polynomial_baseline(self, regions: list[float] or None = None):
        raise NotImplementedError

        reference = self._reference.truncate_to(x_region=regions, inplace=False)


if __name__ == '__main__':
    pass
