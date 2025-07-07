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
from scipy.interpolate import interp2d, LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator

from scipy.signal import correlate, correlation_lags
from copy import deepcopy

from ..SpecCore import *


class CrossCorrelationAlignment:

    __DEFAULT_PARAMS = {
        'x_range': [0, 10000],  # just magic numbers for now
        't_range': [0, 20],     # just magic numbers for now
        'supersampling': 8,
        'threshhold': 0.1
    }

    @property
    def correlation_matrix(self):
        return self._correlation_matrix

    @property
    def shift_vector(self):
        return self._shift_vector

    @property
    def target(self):
        return self._target

    def __init__(self, target: TransientSpectrum, reference: TransientSpectrum, **kwargs):
        self.__parameter = deepcopy(self.__DEFAULT_PARAMS)
        self.__parameter |= kwargs

        self._target = deepcopy(target)
        self._reference = deepcopy(reference)

        self._correlation_matrix = self._cross_correlate()
        self._shift_vector = self._calculate_shift_vector()

    def _prepare_data(self):
        self._target.x = self._target.x.convert_to('wl')
        self._target.sort()

        self._reference.x = self._reference.x.convert_to('wl')
        self._reference.sort()

        tar_small = self._target.truncate_to(x_range=self.__parameter.get('x_range'),
                                             t_range=self.__parameter.get('t_range'),
                                             inplace=False)
        tar_small.orient_data('x')

        ref_small = self._reference.truncate_to(x_range=self.__parameter.get('x_range'),
                                             t_range=self.__parameter.get('t_range'),
                                             inplace=False)
        ref_small.orient_data('x')

        return tar_small, ref_small

    def _create_cross_correlation_grid(self, axis: np.ndarray) -> (float, np.ndarray):
        diff = axis[:-1] - axis[1:]
        val = np.min(np.abs(diff))
        val /= self.__parameter['supersampling']

        return val, np.arange(axis[1], axis[-2], val)

    def _cross_correlate(self):
        target_small, reference_small = self._prepare_data()

        x_val, x = self._create_cross_correlation_grid(reference_small.x.array)
        t_val, t = self._create_cross_correlation_grid(reference_small.t.array)

        target_small.interpolate_to( WavelengthAxis(x, 'wl'), TimeAxis(t, 'ps') )
        reference_small.interpolate_to( WavelengthAxis(x, 'wl'), TimeAxis(t, 'ps') )

        y_corr = correlate(reference_small.y.array, target_small.y.array, method='auto', mode='same')

        x_corr = correlation_lags(np.shape(reference_small.y.array)[1], np.shape(target_small.y.array)[1], mode='same') * x_val
        t_corr = correlation_lags(np.shape(reference_small.y.array)[0], np.shape(target_small.y.array)[0], mode='same') * t_val

        return TransientSpectrum(x_corr, t_corr, y_corr,
                                 x_unit=reference_small.x.unit,
                                 t_unit=reference_small.t.unit,
                                 data_unit=reference_small.y.unit)

    def _calculate_shift_vector(self):
        offset = np.unravel_index(np.argmax(self._correlation_matrix.y.array), self._correlation_matrix.y.shape)

        max_t = int(np.average(np.argmax(self._correlation_matrix.y.array, axis=0), weights=np.max(self._correlation_matrix.y.array, axis=0)))
        max_x = int(np.average(np.argmax(self._correlation_matrix.y.array, axis=1), weights=np.max(self._correlation_matrix.y.array, axis=1)))

        s_t = self._check_shift(self._correlation_matrix.t[max_t], self._target.t.array) # offset[0]
        s_x = self._check_shift(self._correlation_matrix.x[max_x], self._target.x.array) # offset[1]
        print(f'Data is shifted by x = {s_x} nm and y = {s_t} time!')

        return {'x': s_x, 't': s_t}

    def _check_shift(self, shift, axis):
        if abs(shift / (axis[-1] - axis[0])) > self.__parameter['threshhold']:
            print(
                f"x-Axis shift vector larger than {self.__parameter['threshhold'] * 100}% of the entire Axis! "
                f"({shift}); "
                f"Setting shift-vector to 0!"
            )
            shift = 0

        return shift
