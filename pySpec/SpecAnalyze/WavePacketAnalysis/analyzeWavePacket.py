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

import matlab.engine
import numpy as np

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecCore.SpecCoreSpectrum.coreTransient import Transient


class AnalyzeWavePacket:

    MODULE_PATH = r"./"

    def __init__(self, data: TransientSpectrum or Transient):
        self._data = data
        self._wpa = None
        self._t0 = None
        self._parameter = {}

        self.__eng = matlab.engine.connect_matlab()
        self.__eng.addpath(self.MODULE_PATH, nargout=0)

    @property
    def parameter(self):
        return self._parameter

    @property
    def fit(self):
        y = []
        for i, x in enumerate(self._data.x):
            y.append(self._calculate_fit(self._data.t, i))
            print(f"Fitted Position {x:.1f}, Index {i}!")

        return TransientSpectrum(self._data.x, self._data.t, np.array(y),
                                 self._data.x.unit, self._data.t.unit, self._data.y.unit)
    @property
    def fit_ml(self):
        y = []
        for i, x in enumerate(self._data.x):
            t, y_step = self.__eng.fitData(self._wpa, i+1, nargout=2)
            y.append(np.array(y_step))
            print(f"Fitted Position {x:.1f}, Index {i}!")

        return TransientSpectrum(self._data.x, np.array(t)[0], np.array(y)[..., 0],
                                 self._data.x.unit, self._data.t.unit, self._data.y.unit)

    @property
    def t0(self):
        return self._t0

    def _calculate_fit(self, t, idx):
        parameter = self._parameter[idx]

        print(parameter)

        y = np.zeros_like(t)
        for amp, freq, damp, phase in zip(parameter['amplitude'],
                                          parameter['frequency'],
                                          parameter['delay'],
                                          parameter['phase']):
            y = y + amp * np.cos(2 * np.pi * freq * t + phase * (2*np.pi/360)) * np.exp(-t/damp)

        return y

    def cleanup(self):
        self.__eng.quit()

    def hsvd(self, order: int = 25, t0: float or None = None, t0_idx: int = 0):
        if t0 is not None:
            t0_idx, _ = self._data.t.closest_to(t0)

        self._t0 = (self._data.t[t0_idx], t0_idx)

        self._wpa = self.__eng.AnalyzeWavePacket(self._data.x.array,
                                                 self._data.t.array,
                                                 self._data.y.array,
                                                 t0_idx + 1)

        for i, x in enumerate(self._data.x):
            self.__eng.HSVD(self._wpa, i+1, order, nargout=0)
            print(f"Analyzed Position {x}, Index {i}!")
        self._retrieve_parameter()

    def _retrieve_parameter(self):
        for i, _ in enumerate(self._data.x):
            result = np.array(self.__eng.getResults(self._wpa, i+1, nargout=1))

            self._parameter[i] = {'amplitude': result[:, 0],
                                  'frequency': result[:, 1],
                                  'delay': result[:, 2],
                                  'phase': result[:, 3]}

    def low_pass(self, frequency):
        if self._wpa is None:
            raise ValueError()

        for i, _ in enumerate(self._data.x):
            mask = abs(self._parameter[i]['frequency']) < frequency
            for key, val in self._parameter[i].items():
                self._parameter[i][key] = val[mask]

    def high_pass(self, frequency):
        if self._wpa is None:
            raise ValueError()

        for i, _ in enumerate(self._data.x):
            mask = abs(self._parameter[i]['frequency']) > frequency
            for key, val in self._parameter[i].items():
                self._parameter[i][key] = val[mask]

    def isolate(self, frequency, margin=0.05):
        self._retrieve_parameter()

        freq_lo = (1 - margin) * frequency
        freq_hi = (1 + margin) * frequency

        self.high_pass(freq_lo)
        self.low_pass(freq_hi)


if __name__ == '__main__':
    from pySpec.SpecImport.importTimeResolvedFTIR import ImportTimeResolvedFTIR
    import matplotlib.pyplot as plt

    paths = ['']

    f = ImportTimeResolvedFTIR.from_files(paths, 190)

    g = f.average

    hsvd = AnalyzeWavePacket(g)
    hsvd.hsvd(25, 50)
    hsvd.isolate(1 / 500, 0.5)

    fit_res = hsvd.fit
    clean_data = g - fit_res

    fig, ax = plt.subplots(1, 1)
    for i in range(0, len(g.x), 20):
        ax.plot(g.t, g.transient[i].y + i/2000, marker='o', color='grey', ls='')
        ax.plot(fit_res.t, fit_res.transient[i].y + i/2000, lw=2)
        ax.plot(clean_data.t, clean_data.transient[i].y + i/2000)

    plt.show()
