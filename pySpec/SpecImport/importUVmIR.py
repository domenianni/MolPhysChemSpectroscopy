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

from .importBase import ImportTimeResolvedBase

from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from ..SpecCore.coreParser import Parser

from ..pyStitchCorr import stitchCorr

import os
import numpy as np
from copy import deepcopy


class ImportUVmIR(ImportTimeResolvedBase):

    def __init__(self, data_list: list[TransientSpectrum]):
        super().__init__(data_list)

    def __len__(self):
        return len(self._data_list)

    def correct_stitch(self, block_amount: int = 4, reference: int = None,
                       is_asymmetric: bool = True, is_linear: bool = False):
        # For some weird reason, the linear stitching correction does not work when the stitching blocks have been
        # sorted...
        # For now: Always sort, except when a linear correction is requested.
        sort = True
        if is_linear:
            sort = False

        if reference is None:
            reference = -1

        for data in self._data_list:
            if sort:
                data.sort()

            stitchCorr(data.x.array,
                       data.t.array,
                       data.y.array,
                       block_amount,
                       reference,
                       sort,
                       is_asymmetric,
                       is_linear)

        return self

    @classmethod
    def from_files(cls, path):
        files = Parser.parse_path(path)

        data_list = []
        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ps', data_unit='mdod'):
                data_list.append(data)

        return cls(data_list)

    @classmethod
    def from_elDelay_files(cls, path, t_zero_idx=7, step_size=11830):
        files = Parser.parse_path(path)

        data_list = []

        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ns', data_unit='mdod'):
                data.t = step_size * (data.t.array - t_zero_idx)
                data_list.append(data)

        return cls(data_list)


if __name__ == '__main__':
    pass
