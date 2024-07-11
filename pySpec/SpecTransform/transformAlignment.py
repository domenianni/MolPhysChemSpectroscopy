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
from scipy.signal import correlate, correlation_lags
from copy import deepcopy

from ..SpecCore import *


class CrossCorrelationAlignment:

    __DEFAULT_PARAMS = {
        'x_range': [0, 10000],  # just magic numbers for now
        't_range': [0, 20],     # just magic numbers for now
        'supersampling': 4,
        'threshhold': 0.1
    }

    @property
    def correlation_matrix(self):
        return self._correlation_matrix

    @property
    def shift_vector(self):
        offset = np.unravel_index(np.argmax(self._correlation_matrix.y.array), self._correlation_matrix.y.shape)

        return {'x': self._correlation_matrix.x[offset[1]], 't': self._correlation_matrix.t[offset[0]]}

    @property
    def result(self):
        return self._result

    def __init__(self, target: TransientSpectrum, reference: TransientSpectrum, **kwargs):
        self.__parameter = deepcopy(self.__DEFAULT_PARAMS)
        self.__parameter |= kwargs

        self._target = deepcopy(target)
        self._reference = deepcopy(reference)

        self._correlation_matrix = self._cross_correlate()

        self._result = self._shift_data()

    def _prepare_data(self):
        self._target.x = self._target.x.convert_to('wl')
        self._target.sort()
        self._target.orient_data('x')

        self._reference.x = self._reference.x.convert_to('wl')
        self._reference.sort()
        self._reference.orient_data('x')

        tar_small = self._target.truncate_to(x_range=self.__parameter.get('x_range'),
                                             t_range=self.__parameter.get('t_range'),
                                             inplace=False)
        ref_small = self._reference.truncate_to(x_range=self.__parameter.get('x_range'),
                                             t_range=self.__parameter.get('t_range'),
                                             inplace=False)

        return tar_small, ref_small

    def _create_cross_correlation_grid(self, axis: np.ndarray) -> (float, np.ndarray):
        diff = axis[:-1] - axis[1:]
        val = np.min(np.abs(diff))
        val /= self.__parameter['supersampling']

        return val, np.arange(axis[1], axis[-2], val)

    def _cross_correlate(self):
        target_small, reference_small = self._prepare_data()

        target_interpolator = interp2d(target_small.x.array, target_small.t.array, target_small.y.array,
                                       kind='linear', copy=True, bounds_error=False, fill_value=float('NaN'))
        reference_interpolator = interp2d(reference_small.x.array, reference_small.t.array, reference_small.y.array,
                                          kind='linear', copy=True, bounds_error=False, fill_value=float('NaN'))

        x_val, x = self._create_cross_correlation_grid(reference_small.x.array)
        t_val, t = self._create_cross_correlation_grid(reference_small.t.array)

        y_target = target_interpolator(x, t, assume_sorted=True)
        y_reference = reference_interpolator(x, t, assume_sorted=True)

        del (target_interpolator, reference_interpolator)

        index = np.isnan(y_target).any(axis=0) | np.isnan(y_reference).any(axis=0)

        y_corr = correlate(y_reference[:, ~index], y_target[:, ~index], method='fft')

        x_corr = correlation_lags(np.shape(y_reference[:, ~index])[1], np.shape(y_target[:, ~index])[1]) * x_val
        t_corr = correlation_lags(np.shape(y_reference)[0], np.shape(y_target)[0]) * t_val

        return TransientSpectrum(x_corr, t_corr, y_corr,
                                 x_unit=reference_small.x.unit,
                                 t_unit=reference_small.t.unit,
                                 data_unit=reference_small.y.unit)

    def _shift_data(self):
        x = self._target.x.array
        t = self._target.t.array
        y = self._target.y.array

        s_t = self._check_shift(self.shift_vector['t'], t)
        s_x = self._check_shift(self.shift_vector['x'], x)
        print(f'Data is shifted by x = {s_x} nm and y = {s_t} time!')

        if (s_t == 0) and (s_x == 0):
            return self._target

        nan_map = np.zeros_like(y)
        nan_map[np.isnan(y)] = 1

        f = interp2d(t + s_t, x + s_x, np.nan_to_num(y), bounds_error=False, fill_value=float('NaN'), kind='linear')
        f_nan = interp2d(t + s_t, x + s_x, nan_map, bounds_error=False, fill_value=float('NaN'), kind='linear')

        nan_new = f_nan(self._reference.t.array, self._reference.x.array)
        y_new = f(self._reference.t.array, self._reference.x.array)
        y_new[nan_new > 0.5] = float('NaN')

        return TransientSpectrum(self._reference.x,
                                 self._reference.t,
                                 y_new,
                                 self._reference.x.unit,
                                 self._reference.t.unit,
                                 self._reference.y.unit)

    def _check_shift(self, shift, axis):
        if abs(shift / (axis[-1] - axis[0])) > self.__parameter['threshhold']:
            print(
                f"x-Axis shift vector larger than {self.__parameter['threshhold'] * 100}% of the entire Axis! "
                f"({shift}); "
                f"Setting shift-vector to 0!"
            )
            shift = 0

        return shift
