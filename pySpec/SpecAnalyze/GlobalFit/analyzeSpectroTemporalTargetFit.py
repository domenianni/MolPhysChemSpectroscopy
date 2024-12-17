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

from .analyzeGlobalFit import GlobalFit
from .analyzeKineticModel import KineticModel

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum

from lmfit import Parameters
import numpy as np


class SpectroTemporalTargetFit(GlobalFit):

    @property
    def lineshapes(self):
        return [Spectrum(self.data.x,
                         ls,
                         x_unit=self.data.x.unit,
                         data_unit=self.data.y.unit,
                         time=i) for i, ls in enumerate(self._lineshapes.T)]

    @lineshapes.setter
    def lineshapes(self, value):
        if not np.shape(value) == np.shape(self._lineshapes):
            raise ValueError("")

        self._base_lineshapes = value
        self._lineshapes = value

    def __init__(self, data, model):
        super().__init__(data, model)

        self._lineshapes = np.zeros((len(model.start_conc), len(data.x)))
        self._base_lineshapes = self._lineshapes.copy()

    def _kernel(self,
                params: Parameters,
                model: KineticModel,
                data: TransientSpectrum):

        concentrations = model.calculate_concentrations(data.t.array, [params[x] for x in model.parameter]).T
        lineshapes = params['A'] * self._base_lineshapes

        return lineshapes, concentrations

    def _prepare_parameter(self, init_values):
        parameter = self._prepare_k_parameter(init_values[:-1])
        return self._add_amplitude_param(init_values[-1], parameter)

    @staticmethod
    def _add_amplitude_param(init_amp, parameter):
        if init_amp is None:
            init_amp = 1
    
        parameter.add("A", value=init_amp, min=0.001, max=100, expr=None, brute_step=None)
    
        return parameter


if __name__ == '__main__':
    from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
    import scipy.integrate as itg
    import pySpec.SpecCore.coreLineShapes as ls

    data_path = r""
    data = TransientSpectrum.from_file(data_path)
    
    data.eliminate_idx(t_idx=[185, 186, 187])
    parent = Spectrum.calculate_from_files(r"",r"")
    parent.truncate_like(data.x)

    free = Spectrum.calculate_from_files(r"", r"")
    
    free.truncate_like(data.x)

    matrix = (
        "[[-k0 -k1 -k2, 0, 0, 0, 0]," 
        " [ k1, -k4, 0, 0, 0],"       
        " [ k2, 0, -k3, 0, 0],"       
        " [ 0, 0, k3, 0, 0],"         
        " [ k0, k4, 0, 0, 0]]"
    )

    c = (1, 0, 0, 0, -1)
    t = [24, 325, 57, 1e7, 1805]
    init_values = [1 / val for val in t]
    init_values.append(20)
    
    model = KineticModel(matrix, c, False)

    sgp = {
        "int": 1,
        "amp": 1.0,
        "pos": 2044,
        "sigma": 95,
        "skew": 10,
    }
    g1p = {"int": 1, "amp": 0.3, "pos": 2036, "sigma": 10}
    g2p = {"int": 1, "amp": 0.25, "pos": 2042, "sigma": 11}
    
    py = Spectrum.convolve_gaussian(parent.x.array, parent.y.array, 6)
    py = (py / py.max()) - 0.02
    pconv = Spectrum(parent.x.array, np.nan_to_num(py))
    pconv.interpolate_to(data.x.array)
    
    fy = Spectrum.convolve_gaussian(free.x.array, np.nan_to_num(free.y.array), 6)
    fy = fy / fy.max()
    fconv = Spectrum(free.x.array, np.nan_to_num(fy))
    fconv.interpolate_to(data.x.array)
    
    ls5 = np.nan_to_num(pconv.y.array)
    ls5_a = itg.trapezoid(np.nan_to_num(ls5), data.x.array)
    
    ls2 = np.nan_to_num(fconv.y.array)
    ls2_a = itg.trapezoid(ls2, data.x.array)
    ls2 = ls2 * (ls5_a / ls2_a) * 0.5
    
    ls1 = ls.skewed_gaussian(
        data.x.array, sgp["int"], sgp["pos"], sgp["sigma"], sgp["skew"]
    )
    ls1_a = itg.trapezoid(ls1, data.x.array)
    ls1 = ls1 * (ls5_a / ls1_a) * sgp["amp"]
    
    ls3 = ls.voigt(data.x.array, g1p["int"], g1p["pos"], g1p["sigma"], g2p["sigma"])
    ls3_a = itg.trapezoid(ls3, data.x.array)
    ls3 = ls3 * (ls5_a / ls3_a) * g1p["amp"]
    
    ls4 = ls.voigt(data.x.array, g2p["int"], g2p["pos"], g2p["sigma"], g2p["sigma"])
    ls4_a = itg.trapezoid(ls4, data.x.array)
    ls4 = ls4 * (ls5_a / ls4_a) * g2p["amp"]
    
    lineshapes = np.vstack((ls1, ls2, ls3, ls4, ls5))

    f = SpectroTemporalTargetFit(data, model)
    f.lineshapes = lineshapes
    f.t0 = 15

    f.fit(init_values=init_values, optimizer='nelder')
    print(f.fit_report)

    f.plot()
