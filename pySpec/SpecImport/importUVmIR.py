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
from pySpec import EnergyAxis
from pySpec.SpecImport.importBase import ImportTimeResolvedBase

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from pySpec.SpecCore.coreParser import Parser

from pySpec.pyStitchCorr import stitchCorr

import os
import numpy as np
from copy import deepcopy


class ImportUVmIR(ImportTimeResolvedBase):
    """
    A class for importing and processing UV-pump-mIR-probe data.
    """

    def __init__(self, data_list: list[TransientSpectrum]):
        """
        Initializes the ImportUVmIR object with the provided list of TransientSpectrum objects.

        :param data_list: A list of TransientSpectrum objects to process.
        """
        super().__init__(data_list)

    def __len__(self):
        """
        Returns the number of data sets in the object.

        :returns: The number of TransientSpectrum objects in the data list.
        """
        return len(self._data_list)

    def correct_stitch(self, block_amount: int = 4, reference: int = None,
                       is_asymmetric: bool = True, is_linear: bool = False):
        """
        Corrects for stitching artifacts in the data. It applies a correction based on the specified parameters.

        :param block_amount: The number of stitching blocks to consider for the correction (default is 4).
        :param reference: The reference index for stitching (default is None, which means the last block will be used).
        :param is_asymmetric: Whether the stitching correction is asymmetric (default is True).
        :param is_linear: Whether to apply linear stitching correction (default is False). If True, no sorting will be applied.
        :returns: The current ImportUVmIR object with corrected stitching.
        """

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
        """
        Imports data from the provided file path. It reads the data and returns an ImportUVmIR object.

        :param path: The file path or directory containing the data files to import.
        :returns: An ImportUVmIR object containing the imported data.
        """

        files = Parser.parse_path(path)

        data_list = []
        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ps', data_unit='mdod'):
                data_list.append(data)

        return cls(data_list)

    @classmethod
    def from_elDelay_files(cls, path, t_zero_idx=7, step_size=11830):
        """
        Imports data from files where time delays are given in units of steps. It adjusts the time axis by applying
        the specified step size and time zero index.

        :param path: The file path or directory containing the data files to import.
        :param t_zero_idx: The time index corresponding to time zero (default is 7).
        :param step_size: The step size to convert the time delay (default is 11830).
        :returns: An ImportUVmIR object containing the imported and adjusted data.
        """

        files = Parser.parse_path(path)

        data_list = []

        for file in files:
            for data in Parser(file, import_type='x_first', x_unit='wl', t_unit='ns', data_unit='mdod'):
                data.t = step_size * (data.t.array - t_zero_idx)
                data_list.append(data)

        return cls(data_list)


if __name__ == '__main__':
    pass
