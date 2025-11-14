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
from pySpec.SpecCore.SpecCoreSpectrum import Calculation

import numpy as np
import warnings
from pathlib import Path

class Atom:
    def __init__(self, name, x, y, z):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Atom({self.name}, {self.x}, {self.y}, {self.z})"

class ImportOrcaCalculation:
    __UVVIS = {
        'character': '->',
        'slice': slice(3, None),
        'position': 1,
        'intensity': 3,
        'calc_type': 'uvvis'
    }

    __UVVIS_ORCA5 = {
        'character': '.',
        'slice': slice(3, None),
        'position': 1,
        'intensity': 3,
        'calc_type': 'uvvis'
    }

    __IR = {
        'character': ':',
        'slice': slice(1, 5),
        'position': 0,
        'intensity': 2,
        'calc_type': 'ir'
    }
    
    @property
    def uvvis(self):
        return self._uvvis
    
    @property
    def ir(self):
        return self._ir
    
    @property
    def soc(self):
        return self._soc_uvvis

    @property
    def energy(self):
        return self._energy

    @property
    def geometry(self):
        return self._geometry

    def __init__(self, path):
        self.path = Path(path)

        self._energy = []
        self._geometry = []

        self._uvvis = None
        self._soc_uvvis = None
        self._ir = None

        self._read_file(path)

    @staticmethod
    def _read_spectrum(file, params):
        spectrum  = []
        spectrum_started = False

        while True:
            line = file.readline()
            if line == '':
                break

            if params['character'] in line:
                spectrum_started = True
                line = line.split()
                spectrum.append([float(x) for x in line[params['slice']]])

                if spectrum[-1][0] < 0:
                    warnings.warn(f"Imaginary frequency of {spectrum[-1][0]} cm-1 found! Be careful!")

                continue

            if spectrum_started:
                break

        spectrum = np.array(spectrum)
        return Calculation(spectrum[:, params['position']],
                           spectrum[:, params['intensity']],
                           pos_unit='wn', int_unit='f_osc', calc_type=params['calc_type'])

    @staticmethod
    def _read_xyz(file):
        coords = []
        coords_started = False

        while True:
            line = file.readline()
            if line == '':
                break

            if '.' in line:
                coords_started = True
                line = line.split()
                coords.append(Atom(line[0], line[1], line[2], line[3]))
                continue

            if coords_started:
                break

        return coords


    def _read_file(self, filename):
        with open(filename, 'r') as file:
            uvvis_params = self.__UVVIS

            while True:
                line = file.readline()

                if line == '':
                    break

                if line == '\n':
                    continue

                if 'Program Version 5.0' in line:
                    uvvis_params = self.__UVVIS_ORCA5

                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    self._geometry.append(self._read_xyz(file))

                if 'FINAL SINGLE POINT ENERGY' in line:
                    self._energy.append(float(line.split()[-1]))

                if 'IR SPECTRUM' in line:
                    self._ir = self._read_spectrum(file, self.__IR)

                if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
                    uvvis = self._read_spectrum(file, uvvis_params)
                    uvvis.fwhm = 3000
                    uvvis.lineshape = 'gaussian'

                    if 'SOC CORRECTED' in line:
                        self._soc_uvvis = uvvis
                        continue

                    self._uvvis = uvvis

if __name__ == '__main__':
    pass
