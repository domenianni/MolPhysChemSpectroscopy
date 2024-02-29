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

from Spectroscopy.SpecImport.importBase import ImportTimeResolvedBase

from Spectroscopy.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from Spectroscopy.SpecCore.coreParser import Parser

from Spectroscopy.pyStitchCorr import stitchCorr

import os
import numpy as np
from copy import deepcopy


class ImportUVmIR(ImportTimeResolvedBase):

    def __init__(self, data_list: list[TransientSpectrum]):
        super().__init__(data_list)

        self._pre_scans = None

    def subtract_prescans(self, until_time, from_time=None):
        self._orient_all_data('x')

        self._pre_scans = []

        for data in self._data_list:
            until_time_idx = data.t.closest_to(until_time)[0]

            from_time_idx = 0
            if from_time is not None:
                from_time_idx = data.t.closest_to(from_time)[0]

            idx = slice(from_time_idx, until_time_idx)
            self._pre_scans.append(data.spectrum[idx])
            data.y = data.y.array - np.tile(self._pre_scans[-1].y.array, (len(data.t), 1)).T

    def correct_stitch(self):
        for data in self._data_list:
            stitchCorr(data.x.array, data.t.array, data.y.array, 4, False)

            data.sort()

    @classmethod
    def from_files(cls, path):
        files = Parser.parse_path(path)

        data_list = []
        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ps', data_unit='dod'):
                data_list.append(data)

        return cls(data_list)

    @classmethod
    def from_elDelay_files(cls, path, t_zero_idx=7, step_size=11830):
        files = Parser.parse_path(path)

        data_list = []

        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ns', data_unit='dod'):
                data.t = step_size * (data.t.array - t_zero_idx)
                data_list.append(data)

        return cls(data_list)


if __name__ == '__main__':
    pass
