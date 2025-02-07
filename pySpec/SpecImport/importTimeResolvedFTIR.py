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
        """
        Initializes the ImportTimeResolvedFTIR object with provided data and parameters.

        :param data_list: List of data sets (TransientSpectrum objects) to be processed.
        :param pre_scans_amount: Number of pre-scans to consider for delta-OD calculation.
        :param pre_scans_offset: Time offset for the pre-scans.
        :param file_list: List of file names associated with the data sets.
        """
        super().__init__(data_list, file_list)

        self._pre_scans_amount = pre_scans_amount
        self._pre_scans_offset = pre_scans_offset
        self._solvent = None
        self._intensities_list = [deepcopy(data) for data in self._data_list]

        self.__dod = False
        self._calculate_dod()

    def __len__(self):
        """
        Returns the number of intensity spectra available.

        :returns: Length of the intensities list.
        """
        return len(self._intensities_list)

    @property
    def intensities(self):
        """
        Returns the list of intensity spectra.

        :returns: List of intensity data (TransientSpectrum objects).
        """
        return self._intensities_list

    @property
    def intensities_average(self):
        """
        Returns the average of the intensities data.

        :returns: Averaged TransientSpectrum object.
        """
        return TransientSpectrum.average_from(self._intensities_list)

    @property
    def spectra(self):
        """
        Returns the calculated spectra (either intensity spectra or solvent-corrected spectra).

        :returns: List of Spectra objects (solvent-corrected if applicable).
        """
        if self._solvent is None:
            return self._intensities_list
        else:
            return [TransientSpectrum.calculate_from(intensity, self._solvent) for intensity in self._intensities_list]

    @property
    def spectra_average(self):
        """
        Returns the averaged spectra.

        :returns: Averaged TransientSpectrum object of the spectra.
        """
        return TransientSpectrum.average_from(self.spectra)

    def set_solvent_reference(self, path, **kwargs):
        """
        Sets the solvent reference spectrum by reading and interpolating it.

        :param path: Path to the solvent reference file.
        :param kwargs: Additional arguments passed to Spectrum.average_from_files.
        :returns: The current ImportTimeResolvedFTIR object with solvent reference set.
        """
        self._solvent = Spectrum.average_from_files(path, **kwargs)
        self._solvent.interpolate_to(self._intensities_list[0].x, inplace=True)

        return self

    def __singular_timeaxis(self, time_axis: TimeAxis):
        """
        Sets the same time axis for all data sets.

        :param time_axis: The TimeAxis object to assign to all data sets.
        """
        for data in self._data_list:
            data.t = time_axis

    def __averaged_timeaxis(self, time_axes):
        """
        Averages multiple time axes and assigns the result to all data sets.

        :param time_axes: List of TimeAxis objects to average.
        """
        axis = TimeAxis.from_average(time_axes)

        for data in self._data_list:
            data.t = axis

    def __individual_timeaxis(self, time_axes, paths, match):
        """
        Assigns time axes individually to each data set based on file names and matching criteria.

        :param time_axes: List of TimeAxis objects.
        :param paths: List of file paths.
        :param match: Whether to match file names with time axes.
        """
        if match:
            for t, name in zip(time_axes, paths):
                idx = np.argmax(
                    [SequenceMatcher(None, name.name, data_name.name).ratio() for data_name in self._file_list]
                )
                self._data_list[idx].t = t
        else:
            for data, t in zip(self._data_list, time_axes):
                data.t = t

    def timeaxis_from(self, path, match=True):
        """
        Assigns the time axis to the data sets based on files provided in the specified path.

        :param path: Path(s) to files containing time axis information.
        :param match: Whether to match file names with data sets.
        :returns: The current ImportTimeResolvedFTIR object with assigned time axis.
        """
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

        return self

    def define_timeaxis(self, step_size):
        """
       Defines a custom time axis for the data sets based on the provided step size.

       :param step_size: The time step size for the time axis.
       :returns: The current ImportTimeResolvedFTIR object with the defined time axis.
       """
        t = TimeAxis.from_parameters(steps          = len(self._data_list[0].t),
                                     step_size      = step_size,
                                     time_zero_step = self._pre_scans_amount + self._pre_scans_offset,
                                     unit           = self._data_list[0].t.unit)

        for data in self._data_list:
            data.t = t

        return self

    def _calculate_dod(self):
        """
        Calculates the delta-OD (optical density change) for the intensity spectra.

        :raises: Prints message if delta-OD is already computed.
        """
        if not self.__dod:
            for data in self._data_list:
                data.orient_data('t')
                data.y = self._dod(data, self._pre_scans_amount)

            self.__dod = True

        else:
            print("DeltaOD was already computed!")

    def correct_drift(self, t_region: float, position: int or str or slice, drift_type: str = 'individual'):
        """
        Corrects drift in intensity data by normalizing the spectra over a defined time region.

        :param t_region: Time region to use for drift correction.
        :param position: Position in the spectrum to use for drift correction (index, string, or slice).
        :param drift_type: Type of drift correction ('individual' or 'linear').
        :returns: The current ImportTimeResolvedFTIR object with drift-corrected data.
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

        return self

    def atmospheric_correction(self, atm_data: Spectrum, x_range=None):
        """
       Applies atmospheric correction to the data using the provided atmospheric spectrum.

       :param atm_data: The atmospheric spectrum to use for correction.
       :param x_range: The x-range to use for correction (optional).
       :returns: The current ImportTimeResolvedFTIR object with atmospheric correction applied.
       """
        for data in self._data_list:
            data.atmospheric_correction(atm_data, x_range, self._pre_scans_amount)

        return self

    @staticmethod
    def read_data(file, sep='\t'):
        """
        Reads data from a file and returns the x and y values.

        :param file: The file path to read the data from.
        :param sep: The delimiter used in the file (default is tab).
        :returns: A tuple containing the x and y data arrays.
        """
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
        """
        Computes the delta optical density (delta-OD) from the intensity data.

        :param data: The TransientSpectrum data object containing intensity data.
        :param pre_scans_amount: The number of pre-scans to use for reference.
        :returns: The calculated delta-OD values.
        """
        return - np.log10(np.divide(data.y.array, data.spectrum[:pre_scans_amount].y.array))

    @classmethod
    def _read_single_file(cls,
                          file,
                          pre_scans_amount: int = 1,
                          pre_scans_offset: int = 0,
                          x_unit: str = 'wn',
                          t_unit: str = 's',
                          sep: str = '\t'):
        """
        Reads a single file and returns a TransientSpectrum object.

        :param file: The file to read.
        :param pre_scans_amount: Number of pre-scans.
        :param pre_scans_offset: Offset for the pre-scans.
        :param x_unit: The unit for the x-axis (default is 'wn').
        :param t_unit: The unit for the time axis (default is 's').
        :param sep: The delimiter used in the file (default is tab).
        :returns: A TransientSpectrum object created from the file.
        """
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
        """
        Reads multiple files from a path and returns an ImportTimeResolvedFTIR object.

        :param path: Path to the files to be read.
        :param pre_scans_amount: Number of pre-scans.
        :param pre_scans_offset: Offset for the pre-scans.
        :param x_unit: The unit for the x-axis (default is 'wn').
        :param t_unit: The unit for the time axis (default is 's').
        :param sep: The delimiter used in the files (default is tab).
        :returns: An ImportTimeResolvedFTIR object containing the processed data.
        """
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
