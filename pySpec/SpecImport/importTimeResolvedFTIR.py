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
import concurrent.futures as ftr
from difflib import SequenceMatcher
from copy import deepcopy

from scipy.optimize import minimize
from scipy.stats import linregress

from ..SpecCore.coreParser import Parser
from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from ..SpecCore.SpecCoreAxis.coreTimeAxis import TimeAxis

from .importBase import ImportTimeResolvedBase


class ImportTimeResolvedFTIR(ImportTimeResolvedBase):
    """
    This class can be used to process time-resolved ftir data as they are recorded for example by the Bruker Vertex 70
    spectrometers rapid-scan and step-scan modes. These spectrometers record a set of intensity spectra, partially
    before and after a trigger signal (E.g. a Nd:YAG-laser pulse). Therefore, an early call to `calculate_dod` should be
    made, before baseline-correction and other processing steps are taken. This class can also use a reference spectrum
    to calculate a parent-spectrum from the pre-scans.
    """

    def __init__(self, data_list, pre_scans_amount: int = 1, pre_scans_offset: int = 0, file_list: str = None):
        super().__init__(data_list, file_list)

        self._pre_scans_amount = pre_scans_amount
        self._pre_scans_offset = pre_scans_offset
        self._solvent = None
        self._intensities_list = [deepcopy(data) for data in self._data_list]

        self.__dod = False
        self._calculate_dod()

    def __len__(self):
        return len(self._intensities_list)

    @property
    def intensities(self):
        return self._intensities_list

    @property
    def intensities_average(self):
        return TransientSpectrum.average_from(self._intensities_list)

    @property
    def spectra(self):
        if self._solvent is None:
            return self._intensities_list
        else:
            return [TransientSpectrum.calculate_from(intensity, self._solvent) for intensity in self._intensities_list]

    @property
    def spectra_average(self):
        return TransientSpectrum.average_from(self.spectra)

    def set_solvent_reference(self, path, parser_args: dict = None):
        self._solvent = Spectrum.average_from_files(path, parser_args)
        self._solvent.interpolate_to(self._intensities_list[0].x)

    def __singular_timeaxis(self, time_axis: TimeAxis):
        for data in self._data_list:
            data.t = time_axis

    def __averaged_timeaxis(self, time_axes):
        axis = TimeAxis.from_average(time_axes)

        for data in self._data_list:
            data.t = axis

    def __individual_timeaxis(self, time_axes, paths, match):
        if match:
            for t, name in zip(time_axes, paths):
                idx = np.argmax([SequenceMatcher(None, name.name, data_name.name).ratio() for data_name in self._file_list])
                self._data_list[idx].t = t
        else:
            for data, t in zip(self._data_list, time_axes):
                data.t = t

    def timeaxis_from(self, path, match=True):
        # Assign the time axis read in from (a) supplied path(s).
        # First parse the supplied path
        paths = Parser.parse_path(path)

        time_axes: list = []
        for file in paths:
            time_axes.append(TimeAxis.from_file(file, unit=self._data_list[0].t.unit))

        # If only one file was supplied, assign this to all data sets
        if len(time_axes) == 1:
            self.__singular_timeaxis(time_axes[0])

        # Assign them if they are the same amount of files as data sets.
        elif len(time_axes) == len(self._data_list):
            self.__individual_timeaxis(time_axes, paths, match)

        # Else average all supplied timeaxes and assign to all data sets.
        else:
            self.__averaged_timeaxis(time_axes)

        for data in self._data_list:
            data.t -= data.t[self._pre_scans_amount]

    def define_timeaxis(self, step_size):
        t = TimeAxis.from_parameters(steps          = len(self._data_list[0].t),
                                     step_size      = step_size,
                                     time_zero_step = self._pre_scans_amount + self._pre_scans_offset,
                                     unit           = self._data_list[0].t.unit)

        for data in self._data_list:
            data.t = t

    def _calculate_dod(self):
        if not self.__dod:
            for data in self._data_list:
                data.orient_data('t')
                data.y = self._dod(data, self._pre_scans_amount)

            self.__dod = True

        else:
            print("DeltaOD was already computed!")

    def correct_drift(self, t_region: float, position: int or str or slice, drift_type: str = 'individual'):
        """
        Uses the defined `position` to correct intensity loss due to diffusion- or pump-effects. starting at
        `start_time`. The position uses the same syntax as the `transient` and `spectrum` properties of
        :class:`TransientSpectrum`, so `int` for an index, `slice` for a region, which will be averaged, and `string`
        for an x-position.

        :param t_region: The time-point used for the drift-correction and also the starting time for the correction.
        :param position: The position which can be used to correct for an occurring drift.
        """

        if not isinstance(t_region, list):
            t_region = [t_region, np.max(self._data_list[0].t) + 1]

        for data in self._data_list:

            if drift_type == 'individual':
                reference = data.transient[position]
                reference.y = np.where(reference.t.array > t_region[0], reference.y[t_region[0]] / reference.y, 1)

                data.y = data.y * reference.y

            elif drift_type == 'linear':
                t = data.t.array[data.t.closest_to(t_region[0])[0]:data.t.closest_to(t_region[1])[0]]
                y = data.transient[position].y.array[data.t.closest_to(t_region[0])[0]:data.t.closest_to(t_region[1])[0]]

                slope, intercept, rvalue, pvalue, stderr = linregress(t, y)

                ref_y = (slope * data.t.closest_to(t_region[0])[1] + intercept) / (slope * data.t.array + intercept)
                ref_y[0:data.t.closest_to(t_region[0])[0]] = 1

                data.y = data.y * ref_y

    def atmospheric_correction(self, atm_data: Spectrum, x_range=None):
        self._orient_all_data('t')

        atm_data_short = atm_data.truncate_to(x_range=x_range, inplace=False)

        for data in self._data_list:
            data_short = data.truncate_to(x_range=x_range, inplace=False)

            atm_data_short.interpolate_to(data_short.x)
            atm_ar = np.nan_to_num(atm_data_short.y.array)

            amp_array = [0]
            amp_max_idx = np.argmax(atm_ar)

            for i in range(self._pre_scans_amount, len(data_short.t)):
                def target(amp, dat_array, atm_array):
                    return np.sum(np.abs(dat_array - amp * atm_array))

                dat_ar = np.nan_to_num(data_short.spectrum[i].y.array)
                start_val = atm_ar[amp_max_idx] / dat_ar[amp_max_idx]

                res = minimize(target, start_val, args=(dat_ar, atm_ar), method='Nelder-Mead', tol=1e-10)
                amp_array.append(res.x[0])

            atm_data.interpolate_to(data.x)
            data.y = data.y - np.outer(np.array(amp_array), atm_data.y.array)

            del(data_short)

    @staticmethod
    def read_data(file, sep='\t'):
        x = []
        y = []

        with open(file, 'r') as f:
            fl = f.readlines()
        for line in fl:
            x.append(float(line.split(sep)[0]))
            y.append([float(x) for x in line.split(sep)[1:]])
        return np.array(x), np.array(y)

    @staticmethod
    def _dod(data, pre_scans_amount):
        return - np.log10(np.divide(data.y.array, data.spectrum[:pre_scans_amount].y.array))

    @classmethod
    def _read_single_file(cls,
                          file,
                          pre_scans_amount: int = 1,
                          pre_scans_offset: int = 0,
                          x_unit: str = 'wn',
                          t_unit: str = 's',
                          sep: str = '\t'):
        x, y = cls.read_data(file, sep)
        t = TimeAxis.from_parameters(np.shape(y)[1], 1, pre_scans_amount + pre_scans_offset, t_unit)
        return TransientSpectrum(x, t, y, x_unit=x_unit, t_unit=t_unit, data_unit='dod')

    @classmethod
    def from_files(cls,
                   path,
                   pre_scans_amount: int = 1,
                   pre_scans_offset: int = 0,
                   x_unit: str = 'wn',
                   t_unit: str = 's',
                   sep: str = '\t'):
        from ..SpecCore.coreParser import Parser

        files = Parser.parse_path(path)

        with ftr.ThreadPoolExecutor() as executor:
            file_ftr = [
                executor.submit(cls._read_single_file, file, pre_scans_amount, pre_scans_offset, x_unit, t_unit, sep)
                for file in files
            ]

        data_list = [f.result() for f in file_ftr]

        return cls(data_list, pre_scans_amount, file_list=files)


if __name__ == '__main__':
    pass
