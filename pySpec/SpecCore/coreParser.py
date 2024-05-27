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

from .SpecCoreSpectrum.coreSpectrum import Spectrum
from .SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from .SpecCoreSpectrum.coreTransient import Transient

import numpy as np

import re
import os.path
from pathlib import Path
from glob import glob


class Parser:
    """
    :param file_path: Full path to the file to be read in.
    :type file_path: str
    :param import_type: The orientation of a two-dimensional file. Can either be 'x_first' or 't_first'.
    :type import_type: str
    :param x_unit: The unit for the x-axis-object of the returned data object.
    :type x_unit: str
    :param t_unit: The unit for the t-axis-object of the returned data object.
    :type t_unit: str
    :param data_unit: The unit for the data-object of the returned data object
    :type data_unit: str
    :param interface: 'new' or 'legacy'. Legacy for deprecated output data object.
    :type interface: str

    This Parser is built to import both one- and two-dimensional spectroscopic data sets from multiple different data
    formats. It can only read in one file at a time but will split two-dimensional data sets into different runs if
    they are spaced apart within the file.
    One-dimensional data can be oriented as two column vectors or two row vectors, representing the x- and
    y-axis respectively, with the separators being determined by data type.
    Two-dimensional data sets have to consist of a placeholder at the very first position followed by the axes as one
    column- and one row-vector on the upper and left edges of the file and the actual data as a two-dimensional matrix
    in the open space spanned by the two vectors.

    This class also acts as a list/iterator of the read in data sets, which then can be accessed via::

       foo = Parser('path_to_file', 'x_first', 'wn', 'ps', 'dod', 'new')
       first_data = foo[0]

       for data in foo:
           plt.plot(data.x, data.y)
    """

    _seperator = {
        '.CSV': ';',
        '.csv': ';',
        '.dat': '\t',
        '.dpt': '\t'
    }

    def __init__(self,
                 file_path,
                 import_type: str = 'x_first',
                 x_unit: str      = 'wn',
                 t_unit: str      = 's',
                 data_unit: str   = 'od',
                 header: int      = 0,
                 style: str       = 'eng'):

        self.t_unit = t_unit
        self.count = -1

        if not (x_unit == 'wn' or x_unit == 'wl' or x_unit == 'eV'):
            raise ValueError("Energy type can only be one of: 'wn', 'wl' or 'eV'!")
        self.x_unit = x_unit
        self.data_unit = data_unit
        self.import_type = import_type
        self.file_path = file_path

        self.file_content = self._read_file()
        self.file_content = self.file_content[header:]

        if style == 'ger':
            self.file_content = [s.replace(',', '.') for s in self.file_content]

        if import_type in ('one_dimension', 'spectrum', 'transient'):
            self.data = self._read_data(self._one_dimensional_data)
        elif import_type == 'x_first' or 't_first':
            self.data = self._read_data(self._two_dimensional_data)
        else:
            ValueError("import_type has to be either 'x_first', 't_first' or 'one_dimension' !")

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count < len(self.data):
            return self.data[self.count]
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.data[item]

    def _read_file(self):
        with open(self.file_path, 'r') as file:
            string_data = file.readlines()

        return string_data

    def _read_data(self, method):
        string_data = []
        data_list = []

        for line in self.file_content:
            if not (line == '\n') and not (not line):
                string_data.append(line)
                continue

            if string_data:
                data_list.append(self._extract_data(string_data, method))
                string_data = []

        if string_data:
            data_list.append(self._extract_data(string_data, method))

        return data_list

    def _extract_data(self, string_data, method):
        first_axis, second_axis, data = method(string_data)

        return self._create_data(first_axis, second_axis, data)

    def _create_data(self, first_axis, second_axis, data):
        if self.import_type == 'spectrum':
            return Spectrum(first_axis, data, self.x_unit, self.data_unit)
        elif self.import_type == 'transient':
            return Transient(first_axis, data, self.t_unit, self.data_unit)

        elif self.import_type == 'x_first':
            return TransientSpectrum(first_axis, second_axis, data, self.x_unit, self.t_unit, self.data_unit)
        elif self.import_type == 't_first':
            return TransientSpectrum(second_axis, first_axis, data, self.x_unit, self.t_unit, self.data_unit)

        else:
            raise ValueError("import_type must be of: spectrum, transient, x_first or t_first.")

    @staticmethod
    def _two_dimensional_data(string_data):
        success = False
        first_axis = []
        while not success:
            string = string_data.pop(0)
            invalid_line = False

            for x in string.split()[1:]:
                if invalid_line:
                    break
                try:
                    first_axis.append(float(x))
                except ValueError:
                    invalid_line = True
                    first_axis = []

            if first_axis and len(first_axis) > 1:
                success = True

        second_axis = []
        data = []

        for line in string_data:
            try:
                second_axis.append(float(line.split()[0]))
            except ValueError or IndexError:
                continue
            try:
                data.append([float(y) for y in line.split()[1:]])
            except ValueError or IndexError:
                data.append([float('NaN') for _ in first_axis])

        return np.array(first_axis), np.array(second_axis), np.array(data)

    def _one_dimensional_data(self, string_data):

        if len(string_data) > 2:
            first_axis = []
            data = []

            sep = self._seperator.get(self.file_path.name[-4:])
            if sep is None:
                sep = ' '

            try:
                for line in string_data:
                    line = line.lstrip()
                    first_axis.append(float(line.split(sep)[0]))
                    data.append(float(line.split(sep)[1]))

            except ValueError:
                for line in string_data:
                    line = line.lstrip()
                    first_axis.append(float(re.split('\s\s*', line)[0]))
                    data.append(float(re.split('\s\s*', line)[1]))
        else:
            first_axis = [float(x) for x in string_data[0].split()]
            data = [float(x) for x in string_data[1].split()]

        # TODO: More sophisticated selection algorithm

        return np.array(first_axis), None, np.array(data)

    @staticmethod
    def parse_path(path):

        if isinstance(path, str):
            if os.path.isdir(path):
                return [Path(x) for x in glob(os.path.join(path, "*")) if os.path.isfile(x)]
            elif os.path.isfile(path):
                return [Path(path)]

        if isinstance(path, list):
            path_list = []
            for i in path:
                if os.path.isdir(i):
                    path_list.extend(glob(os.path.join(i, "*")))
                elif os.path.isfile(i):
                    path_list.append(i)

            return [Path(x) for x in path_list if os.path.isfile(x)]

        if isinstance(path, Path):
            if os.path.isfile(path):
                return [path]
            elif os.path.isdir(path):
                return [Path(x) for x in path.iterdir() if os.path.isfile(x)]

        raise ValueError(f"Not recognized path: {path}, {type(path)}")


if __name__ == '__main__':
    pass
