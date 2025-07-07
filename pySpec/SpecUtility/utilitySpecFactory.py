import numpy as np
from random import random

from pySpec import ImportUVmIR
from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecCore.coreLineShapes import gaussian, skewed_gaussian, lorentzian, voigt, pseudo_voigt
from pySpec.SpecAnalyze.GlobalFit.analyzeKineticModel import KineticModel


class SpecSynthesizer(TransientSpectrum):

    _fntypes = {
        'gauss': gaussian,
        'lorentzian': lorentzian
    }

    def __init__(self,
                 model: KineticModel,
                 k_parameter = None,
                 lineshape_params = None,
                 lineshapes = None,
                 x_range = (1500, 1700),
                 x_num = 128):
        self._model = model
        self._lineshapes = lineshapes

        self._stitching_offsets = None
        self._noise = None

        x = np.linspace(x_range[0], x_range[1], x_num)
        t = np.append(np.linspace(0.1, 1, 30, endpoint=False),
                      np.logspace(0, 3, 60)
            )

        if self._lineshapes is None:
            self._lineshapes = [
               self._fntypes[val['fntype']](x, val['s0'], val['nu0'], val['fwhm']) for val in lineshape_params
            ]

        self._concentrations = self._model.calculate_concentrations(t, k_parameter).T

        y = self._concentrations @ self._lineshapes

        t = np.append(np.linspace(-1, 0, 30), t)
        t = np.append(np.linspace(-100, -1, 10, endpoint=False), t)
        y = np.append(np.zeros([40, len(x)]), y, axis=0)

        super().__init__(
            x, t, y, x_unit='wl'
        )

    def add_noise(self, amplitude):
        self._noise = 2 * amplitude * np.random.rand(*np.shape(self._data.array)) - amplitude
        self._data.array += self._noise

        return self

    def add_stitching_error(self, block_amount=2, amplitude=1):
        if len(self._x_axis) % block_amount != 0:
            raise ValueError()

        block_offsets = np.array([
            [2 * amplitude * random() - amplitude for _ in range(block_amount)] for _ in range(len(self._t_axis))
        ])

        self._stitching_offsets = np.tile(block_offsets, int(len(self._x_axis) / block_amount))

        self._data.array += self._stitching_offsets

        return self

    def add_shift(self, x_shift, t_shift):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    matrix = (
        "[[-k0, 0, 0, 0, 0],"  # LMCT
        " [ k0, -k1 -k2, 0, 0, 0],"  # ES1
        " [ 0, k2, -k3 -k4, 0, 0],"  # ISC
        " [ 0, 0, k3, 0, 0],"  # Product
        " [ 0, k1, k4, 0, 0]]"  # Parent
    )

    c = (1, 0, 0, 0, -1)

    kinetic_model = KineticModel(matrix, c, solve_symbol=False)

    sim_specs = [
        SpecSynthesizer(kinetic_model,
                               [1/10, 1/200, 1/50, 1/100, 1/500],
                               [
                                   {'fntype': 'gauss', 'nu0': 1590, 's0': 1, 'fwhm': 50},
                                   {'fntype': 'gauss', 'nu0': 1570, 's0': 1, 'fwhm': 20},
                                   {'fntype': 'gauss', 'nu0': 1600, 's0': 1, 'fwhm': 20},
                                   {'fntype': 'gauss', 'nu0': 1550, 's0': 1, 'fwhm': 20},
                                   {'fntype': 'gauss', 'nu0': 1680, 's0': 1, 'fwhm': 20}
                               ])#.add_noise(0.01).add_stitching_error(block_amount=8, amplitude=0.01)
        for _ in range(2)
    ]

    for spec in sim_specs[1:]:
        spec.t += 12
        spec.x -= 1.5

        spec.interpolate_to(sim_specs[0].x, sim_specs[0].t)
        spec.y = np.nan_to_num(spec.y.array)

    test = ImportUVmIR(sim_specs)
    #test.correct_stitch(block_amount=8)
    test.cross_correlate(reference_idx=0, par=False, x_region=[10000000/1500, 10000000/1700], t_region=[-2, 20])

    #a = test.average
    plt.pcolormesh(test._cc._correlation_matrix.x, test._cc._correlation_matrix.t, test._cc._correlation_matrix.y)
    #plt.pcolormesh(a.t, a.x, a.y, norm=TwoSlopeNorm(0, np.min(a.y), np.max(a.y)))

    plt.figure()
    plt.pcolormesh(test[0].x, test[0].t, test[0].y)#, norm=TwoSlopeNorm(0, np.min(a.y), np.max(a.y)))
    plt.figure()
    plt.pcolormesh(test[1].x, test[1].t, test[1].y)#, norm=TwoSlopeNorm(0, np.min(a.y), np.max(a.y)))

    plt.show()
