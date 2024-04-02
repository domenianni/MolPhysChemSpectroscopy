"""
This file is part of pySpec
    Copyright (C) 2024  Markus Bauer
    By an Idea of L. I. Domenianni

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

from .analyzeGlobalFit import GlobalFit
from .analyzeKineticModel import KineticModel

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecCore.SpecCoreSpectrum import Spectrum

from lmfit import Parameters
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


class PartialTargetFit(GlobalFit):

    _WEIGHTING = 10

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value: Spectrum):
        if not isinstance(value, Spectrum):
            raise ValueError(f"parent has to be of type Spectrum, not {type(value)}!")

        value = value.interpolate_to(self.data.x, False)
        value.y = np.nan_to_num(value.y)

        self._parent = value

    @property
    def positive_data(self):
        return self._positive_data

    @property
    def _calculated_data(self):
        if self._concentrations is not None and self._lineshapes is not None:
            return ((self._concentrations[:, :-1] @ self._lineshapes) +
                    self._parameter['A'] * np.outer(self._concentrations[:, -1], self._parent.y.array))
        else:
            return None

    def __init__(self, data, model):
        super().__init__(data, model)

        self._parent: Spectrum or None = None
        self._positive_data: np.ndarray = None

    def _target_function(self, params: Parameters):
        self._lineshapes, self._concentrations, self._positive_data = self._kernel(params, self.model, self.data)

        return (
                np.sum(np.abs(self._residuals)) +
                self._WEIGHTING * np.sum(np.abs(np.where(self._positive_data < 0, self._positive_data, 0)))
                )

    def _kernel(self,
                params: Parameters,
                model: KineticModel,
                data: TransientSpectrum):
        concentrations = self.model.calculate_concentrations(self.data.t.array, [params[x] for x in model.parameter]).T

        lineshapes, positive_data = self._calculate_lineshapes(params['A'],
                                                               self._parent.y.array,
                                                               concentrations,
                                                               self.data.y.array)

        return lineshapes, concentrations, positive_data

    @staticmethod
    def _calculate_lineshapes(amplitude: float, parent: np.ndarray, concentrations: np.ndarray, data: np.ndarray):
        positive_data = np.matrix(data - amplitude * np.outer(concentrations[:, -1], parent))
        remaining_concentrations = np.matrix(concentrations[:, :-1])

        lineshapes, resid, rank, singul_val = lstsq(
            remaining_concentrations, positive_data
        )

        return lineshapes, positive_data

    def _prepare_parameter(self, init_values):
        parameter = self._prepare_k_parameter(init_values[:-1])
        return self._add_amplitude_param(init_values[-1], parameter)

    @staticmethod
    def _add_amplitude_param(init_amp, parameter):
        if init_amp is None:
            init_amp = 1

        parameter.add("A", value=init_amp, min=0, expr=None, brute_step=None)
    
        return parameter

    def plot(self):
        fig, axs = self._plot_setup()

        axs[0, 2].plot(self.data.x, self._parameter['A'] * self._parent.y)

        plt.show()


if __name__ == '__main__':
    from Spectroscopy.SpecCore.SpecCoreSpectrum import Spectrum
    
    data_path = r""
    data = TransientSpectrum.from_file(data_path)
    data.eliminate_repetition()
    data.eliminate_idx(t_idx=[188, 187, 185])
    
    parent = Spectrum.calculate_from_files(r"", r"")
    parent.interpolate_to(data.x)
    
    matrix = (
        "[[-k0, 0, 0, 0, 0],"            # LMCT
        " [ k0, -k1 -k2, 0, 0, 0],"      # ES1
        " [ 0, k2, -k3 -k4, 0, 0],"      # ISC
        " [ 0, 0, k3, 0, 0],"            # Product
        " [ 0, k1, k4, 0, 0]]"           # Parent
    )

    c = (1, 0, 0, 0, -1)
    
    model = KineticModel(matrix, c, solve_symbol=False)

    py = Spectrum.convolve_gaussian(parent.x.array, parent.y.array, 6)
    py = py / py.max()
    pconv = Spectrum(parent.x.array, np.nan_to_num(py))
    pconv.interpolate_to(data.x)
    
    pconv = np.nan_to_num(pconv.y.array)

    f = PartialTargetFit(data, model)
    f.parent = pconv
    f.t0 = 1

    init_vals = [1/val for val in [3.5, 13, 36, 4600, 10000]]
    init_vals.append(11)

    f.fit(init_vals, 'powell')
    print(f.fit_report)

    f.plot()
