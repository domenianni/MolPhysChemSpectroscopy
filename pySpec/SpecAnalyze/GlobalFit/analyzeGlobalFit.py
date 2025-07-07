import matplotlib.pyplot as plt

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecCore.SpecCoreSpectrum.coreTransient import Transient
from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum

from .analyzeKineticModel import KineticModel

from copy import deepcopy
import numpy as np
from scipy.linalg import lstsq
from lmfit import minimize, Parameters, fit_report
from matplotlib.colors import Normalize

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


class GlobalFit:

    _optimizer = {'nelder' : {'method': 'nelder',
                              'tol'   : 1e-10},
                  'ampgo': {'method': 'ampgo',
                            'local' : 'Nelder-Mead',
                            'disp'  : True},
                  'dual_annealing': {'method': 'dual_annealing'},
                  'slsqp': {'method': 'slsqp'},
                  'powell': {'method': 'powell'},
                  'bfgs': {'method': 'bfgs'},
                  'least_squares': {'method': 'least_squares'},
                  'lm': {'method': 'leastsq'}
                  }
    
    def __init__(self, data: TransientSpectrum, model: KineticModel):
        """
        Base class for the global fit algorithm and capable of executing a global fit based on a KINETIC model,
        resulting in spectral lineshapes. A fit is conducted after instatiation using the member-fucntion `fit`.
        Parameters are generally set using properties. This class posesses though only the initial fit-timestep `t0` as
        a modifyable parameter.

        Fit results, so `lineshapes`, `concentrations`, `calculated_data` and `residuals`can be accessed after the fit
        has concluded, as well as a `fit_report` supplied by the lmfit module.

        Since a global fit often produces contaminated lineshapes, these artifacts can be removed using the
        `remove_fit_artefact` function, though this modifies the calculated lineshapes.

        A prebuilt plot can be created and returned by the method `.plot()`

        :param data: The data to be fitted as an instance of :class:`TransientSpectrum`.
        :param model: The kinetic model used for the fit as an instance of :class:`KineticModel`.
        """

        self.model = model
        self.data = deepcopy(data)

        self.data.y = np.nan_to_num(self.data.y.array)
        self.data.eliminate_repetition()
        self.data.orient_data('t')

        self._backup_data = deepcopy(self.data)

        self.t0 = 0
        self._lineshapes = None
        self._concentrations = None
        self._parameter = None
        self.result = None

    def fit(self, init_values: list[float] or None = None, vary_values=None, optimizer='nelder'):
        self._parameter = self._prepare_parameter(init_values, vary_values)

        self.result = minimize(self._target_function,
                               params=self._parameter,
                               nan_policy='omit',
                               **self._optimizer[optimizer]
                               )

        return self

    def evaluate(self, values: list[float] or None = None):
        self._parameter = self._prepare_parameter(values, None)

        self._target_function(self._parameter)

        return self

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = value
        self.data = deepcopy(self._backup_data)
        self.data = self.data.truncate_to(t_range=[value, np.max(self._backup_data.t)])

    @property
    def lineshapes(self):
        return [Spectrum(self.data.x,
                         ls,
                         x_unit=self.data.x.unit,
                         data_unit=self.data.y.unit,
                         time=i) for i, ls in enumerate(self._lineshapes)]

    @property
    def concentrations(self):
        return [Transient(self.data.t,
                          conc,
                          t_unit=self.data.x.unit,
                          data_unit=self.data.y.unit,
                          position=i) for i, conc in enumerate(self._concentrations.T)]

    @property
    def _calculated_data(self):
        if self._concentrations is not None and self._lineshapes is not None:
            return self._concentrations @ self._lineshapes
        else:
            return None

    @property
    def calculated_data(self):
        if (data := self._calculated_data) is not None:
            return TransientSpectrum(self.data.x,
                                     self.data.t,
                                     data,
                                     x_unit=self.data.x.unit,
                                     t_unit=self.data.t.unit,
                                     data_unit=self.data.y.unit)
        else:
            return None

    @property
    def _residuals(self):
        if (calculated_data := self._calculated_data) is not None:
            return self.data.y.array - calculated_data
        else:
            return None

    @property
    def residuals(self):
        if (residuals := self._residuals) is not None:
            return TransientSpectrum(self.data.x,
                                     self.data.t,
                                     residuals,
                                     x_unit=self.data.x.unit,
                                     t_unit=self.data.t.unit,
                                     data_unit=self.data.y.unit)
        else:
            return None

    @property
    def fit_report(self):
        return fit_report(self.result)

    def _target_function(self, params: Parameters):
        self._lineshapes, self._concentrations = self._kernel(params, self.model, self.data)

        return np.sum(np.square(self._residuals))

    @staticmethod
    def _kernel(params: Parameters,
                model: KineticModel,
                data: TransientSpectrum):

        k_parameter = [1/params[x].value for x in model.parameter if "f" not in x]
        concentrations = model.calculate_concentrations(data.t.array, k_parameter).T

        lineshapes, resid, rank, singul_val = lstsq(concentrations, np.matrix(data.y.array))

        return lineshapes, concentrations

    def _prepare_parameter(self, init_values, vary_values):
        if init_values is None:
            init_values = [1 for x in self.model.parameter if "f" not in x]

        if vary_values is None:
            vary_values = [1 for x in self.model.parameter if "f" not in x]
        
        return self._prepare_tau_parameter(init_values, vary_values)

    def _prepare_tau_parameter(self, init_values, vary_values):
        parameter = Parameters()
        for p, i_val, vary in zip(self.model.parameter, init_values, vary_values):
            parameter.add(p, value=i_val, vary=vary, expr=None, brute_step=None, min=0, max=1e10)

            if not "f" in p:
                parameter.add('k_'+p, 1/i_val, expr=f'1/{p}', vary=False)

        return parameter

    def _plot_setup(self):
        fig, axs = plt.subplots(2, 3, gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

        max_val = np.max(np.abs(self._calculated_data))
        norm = Normalize(- max_val, max_val)
        axs[0, 0].pcolormesh(self.data.y, norm=norm)
        axs[0, 0].set_title('Data', size=14, weight='bold')
        axs[0, 0].tick_params(labelbottom=False, labeltop=True)
        axs[0, 0].set_ylabel(f"Time / samples", size=14)

        axs[0, 1].pcolormesh(self._calculated_data, norm=norm)
        axs[0, 1].set_title('Fit', size=14, weight='bold')
        axs[0, 1].tick_params(labelbottom=False, labelleft=False, labeltop=True)

        axs[1, 0].pcolormesh(self._residuals, norm=norm)
        axs[1, 0].set_xlabel(f"Energy / samples\nResiduals (norm.)", size=14)
        axs[1, 0].set_ylabel(f"Time / samples", size=14)

        axs[1, 1].pcolormesh(self._residuals)
        axs[1, 1].tick_params(labelleft=False)
        axs[1, 1].set_xlabel(f"Energy / samples\nResiduals", size=14)

        axs[0, 2].plot(self.data.x, self._lineshapes.T)
        axs[0, 2].tick_params(labelbottom=False, labelleft=False, labelright=True, labeltop=True)
        axs[0, 2].set_xlabel(f"Energy / {self.data.x.unit}", size=14)
        axs[0, 2].set_ylabel(f"Intensity / {self.data.y.unit}", size=14)
        axs[0, 2].xaxis.set_label_position('top')
        axs[0, 2].yaxis.set_label_position('right')

        axs[1, 2].plot(self.data.t, self._concentrations)
        axs[1, 2].set_xscale('symlog')
        axs[1, 2].tick_params(labelright=True, labelleft=False)
        axs[1, 2].set_xlabel(f"Time / {self.data.t.unit}", size=14)
        axs[1, 2].set_ylabel(f"Fraction", size=14)
        axs[1, 2].yaxis.set_label_position('right')

        return fig, axs

    def plot(self):
        fig, axs = self._plot_setup()
        return fig, axs

    def remove_fit_artefact(self, reference_idx):
        if reference_idx == 'average':
            reference = np.mean(self._lineshapes, axis=0)
        else:
            reference = self._lineshapes[reference_idx]

        self._lineshapes = self._lineshapes - reference


if __name__ == '__main__':

    data_path = r""

    data = TransientSpectrum.from_file(data_path)

    matrix = (
        "[[-k0, 0, 0, 0, 0],"            # LMCT
        " [ k0, -k1 -k2, 0, 0, 0],"      # ES1
        " [ 0, k2, -k3 -k4, 0, 0],"      # ISC
        " [ 0, 0, k3, 0, 0],"            # Product
        " [ 0, k1, k4, 0, 0]]"           # Parent
    )

    c = (1, 0, 0, 0, -1)
    
    kinetic_model = KineticModel(matrix, c, solve_symbol=False)
    #kinetic_model = KineticModel.decay_associated_model(4)

    target_fit = GlobalFit(data=data, model=kinetic_model)
    target_fit.t0 = 0.1
    target_fit.fit(init_values = [1 / val for val in [3.5, 14.6, 30, 4000, 1e7]])

    print(target_fit.fit_report)

    target_fit.plot()
    plt.show()
