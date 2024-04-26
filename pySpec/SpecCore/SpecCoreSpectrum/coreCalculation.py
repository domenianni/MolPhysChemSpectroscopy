import numpy as np
from scipy.signal import convolve
from copy import deepcopy

from ..coreCalculationParser import CalculationParser
from ..SpecCoreData import OneDimensionalData
from ..SpecCoreAxis import EnergyAxis, WavelengthAxis
from ..SpecCoreSpectrum import Spectrum
from ..coreLineShapes import gaussian, lorentzian, skewed_gaussian, voigt, pseudo_voigt
from ..SpecCoreData.coreAbstractData import AbstractData


class Calculation(Spectrum):

    __ENV_FN = {
        'gaussian': gaussian,
        'lorentz': lorentzian,
        'skewed-gaussian': skewed_gaussian,
        'voigt': voigt,
    }

    __ENVELOPE = {
        'gaussian': {'fwhm': 10},
        'lorentz': {'fwhm': 10},
        'skewed-gaussian': {'fwhm': 10, 'alpha': 1},
        'voigt': {'fwhm_gauss': 10, 'fwhm_lorentz': 10}
    }

    __UVVIS = {
        'min' : 0,
        'max' : 1000,
        'step' : 0.001,
        'kw': ('uvvis', 'UVVis', 'TDDFT', 'tddft')
        }

    __IR = {
        'min': 0,
        'max': 4000,
        'step': 0.1,
        'kw': ('ir', 'IR', 'freq')
        }

    @property
    def roots(self):
        return {i + 1: {'position': p, 'intensity': ints} for i, (p, ints) in enumerate(zip(self.pos, self.int))}

    @property
    def lineshape(self):
        return self._lineshape

    @lineshape.setter
    def lineshape(self, value: str):
        if not value in self.__ENVELOPE.keys():
            raise KeyError(f"Lineshape {value} not recognized!")

        self._lineshape = value
        self._lineshape_params = self.__ENVELOPE.get(value)

        self._recalculate = True
        
    @property
    def shape_params(self):
        return self._lineshape_params
    
    @shape_params.setter
    def shape_params(self, value):
        self._lineshape_params.update(value)
        self._recalculate = True

    @property
    def pos(self):
        return self._pos_axis

    @property
    def int(self):
        return self._int_axis

    @property
    def y(self):
        if self._recalculate:
            self.y = self._calc_envelope(self.x)
            self._recalculate = False
        return self._data

    @y.setter
    def y(self, array: np.ndarray):
        if not isinstance(array, np.ndarray) and not isinstance(array, AbstractData):
            raise ValueError(f"data can only be ndarray or AbstractData but is {type(array)}!")

        if isinstance(array, AbstractData):
            self._data = deepcopy(array)
        else:
            self._data.array = array.copy()

    def sort(self):
        pass

    def save(self, path):
        pass

    def eliminate_repetition(self):
        pass

    @classmethod
    def from_file(cls, path, params=None):
        if params is None:
            params = {'lineshape': 'lorentz'}

        file = CalculationParser(path)

        return cls(file.pos, file.int, file.x_unit, file.y_unit, calc_type=file.calc_type, **params)

    def __init__(self,
                 positions:     np.ndarray,
                 intensities:   np.ndarray,
                 pos_unit:      str = 'wn',
                 int_unit:      str = 'f_osc',
                 calc_type:     str = 'ir',
                 lineshape:     str = 'lorentz'):

        self._int_axis = OneDimensionalData(intensities, int_unit)
        self._lineshape_params = self.__ENVELOPE.get(lineshape)
        self._lineshape = lineshape
        self._recalculate = False

        if len(np.shape(intensities)) != 1:
            raise ValueError("Data Object only accepts one-dimensional arrays.")

        if pos_unit in ('wn', 'ev'):
            self._pos_axis = EnergyAxis(positions, pos_unit)
        if pos_unit == 'wl':
            self._pos_axis = WavelengthAxis(positions, pos_unit)

        if calc_type in self.__IR.get('kw'):
            self._params = self.__IR
        elif calc_type in self.__UVVIS.get('kw'):
            self._params = self.__UVVIS
        else:
            ValueError(f'calc_type: {calc_type} not recognized!')

        x = np.arange(self._params['min'], self._params['max'] + self._params['step'], self._params['step'])
        y = self._calc_envelope(x)
        super().__init__(x, y, x_unit=pos_unit, data_unit=int_unit)

    def _calc_envelope(self, x):
        shape = self._calc_lineshape()

        envelope = np.zeros_like(x)

        for p, i in zip(self._pos_axis, self._int_axis):
            envelope[np.argmin(np.abs(x - p))] = i

        envelope = convolve(envelope, shape, mode='same')
        envelope = np.max(self._int_axis.array) * envelope / np.max(envelope)

        return envelope

    def _calc_lineshape(self):
        x = np.arange(-self._params['max'], self._params['max'] + self._params['step'], self._params['step'])

        if (fn := self.__ENV_FN.get(self.lineshape)) is None:
            raise Exception('Your function type has to be either:\n'
                            '-    gaussian\n'
                            '-    lorentz\n'
                            '-    pseudo-voigt.\n'
                            'Leaving it empty returns gauss-type function')

        shape = fn(nu=x, s0=1, nu0=0, **self._lineshape_params)

        return shape


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    irc = Calculation.from_file("")

    plt.bar(irc.pos, irc.int, 1)
    irc.shape_params = {'fwhm': 40}
    plt.plot(irc.x, irc.y)

    plt.xlim(200, 1000)
    plt.ylim(0.001, 1)

    plt.show()
