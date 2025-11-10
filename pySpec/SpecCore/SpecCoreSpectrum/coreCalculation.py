import numpy as np
from scipy.signal import convolve
from copy import deepcopy

from pySpec.SpecCore.coreCalculationParser import CalculationParser
from pySpec.SpecCore.SpecCoreData import OneDimensionalData
from pySpec.SpecCore.SpecCoreAxis import EnergyAxis, WavelengthAxis
from pySpec.SpecCore.SpecCoreSpectrum import Spectrum
from pySpec.SpecCore.coreLineShapes import gaussian, lorentzian, skewed_gaussian, voigt, pseudo_voigt, lognormal #, lognormal
from pySpec.SpecCore.SpecCoreData.coreAbstractData import AbstractData
from pySpec.SpecCore.coreFunctions import inPlaceOp


class Calculation(Spectrum):

    __ENV_FN = {
        'gaussian': gaussian,
        'lorentz': lorentzian,
        'skewed-gaussian': skewed_gaussian,
        'voigt': voigt,
        'lognormal': lognormal
    }

    __ENVELOPE = {
        'gaussian': {'fwhm': 10},
        'lorentz': {'fwhm': 10},
        'skewed-gaussian': {'fwhm': 10, 'alpha': 1},
        'voigt': {'fwhm_gauss': 10, 'fwhm_lorentz': 10},
        'lognormal': {'fwhm': 10, 'alpha': 1}
    }

    __UVVIS = {
        'min' : 0,
        'max' : 1000,
        'step' : 0.1,
        'kw': ('uvvis', 'UVVis', 'TDDFT', 'tddft')
        }

    __UVVIS_WN = {
        'min' : 0,
        'max' : 100000,
        'step': 10.0,
        'kw': ('uvvis', 'UVVis', 'TDDFT', 'tddft')
    }

    __UVVIS_EV = {
        'min': 0,
        'max': 10,
        'step': 0.001,
        'kw': ('uvvis', 'UVVis', 'TDDFT', 'tddft')
    }

    __IR = {
        'min': 0,
        'max': 4000,
        'step': 0.1,
        'kw': ('ir', 'IR', 'freq')
        }

    @property
    def fwhm(self):
        return self._lineshape_params.get('fwhm')

    @fwhm.setter
    def fwhm(self, value: float):
        self._lineshape_params['fwhm'] = value
        self._recalculate = True

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._pos_axis.array *= value / self._scaling
        self._scaling = value
        self._recalculate = True

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

        new = dict()
        for key, val in self.__ENVELOPE.get(value).items():
            if not key in self._lineshape_params:
                new[key] = val
            else:
                new[key] = self._lineshape_params.get(key)

        self._lineshape_params = new

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
            self._data.array = self._calc_envelope(self._x_axis.array)
            self._recalculate = False
        return self._data

    @y.setter
    def y(self, array: np.ndarray):
        raise NotImplementedError()

    def sort(self):
        pass

    def subtract(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        cp = deepcopy(self)
        cp._int_axis *= other

        return cp

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self._add(other)

    def _add(self, other):
        if not (
                (type(self._pos_axis) == type(other._pos_axis)) and
                (self._params.get('kw') == other._params.get('kw'))):
            return ValueError("Calculations not matching in type!")

        new_pos = np.append(self._pos_axis.array, other._pos_axis.array)
        new_int = np.append(self._int_axis.array, other._int_axis.array)

        idx = np.argsort(new_pos)

        new_pos = new_pos[idx]
        new_int = new_int[idx]

        if len(set(new_pos)) != len(new_pos):
            idx = np.where(np.diff(new_pos) == 0)

            new_pos = new_pos[np.diff(new_pos, append=new_pos[-1]+1) != 0]

            for i in idx:
                new_int[i+1] = new_int[i] + new_int[i+1]

            new_int = np.delete(new_int, idx)

        return Calculation(new_pos, new_int, self._pos_axis.unit, calc_type=self._params.get('kw')[0])

    def save(self, path):
        """
        :param path: The path to save at.

        Saves the spectrum as an ascii-formatted file.
        """
        self._save_one_dimension(self._x_axis.array, self._data.array, path)

        return self

    def save_stick(self, path):
        """
        :param path: The path to save at.

        Saves the spectrum as an ascii-formatted file.
        """

        self._save_one_dimension(self._pos_axis.array, self._int_axis.array, path)

        return self

    def eliminate_repetition(self):
        raise NotImplementedError()

    @inPlaceOp
    def truncate_to(self, x_range=None):
        #TODO truncating after setting a scaling factor shifts only the position, not the lineshape...?
        x_range = [self.x.closest_to(x)[0] for x in x_range]

        mask = np.where(self._pos_axis.array < self._x_axis[np.min(x_range)], False, True)
        mask *= np.where(self._pos_axis.array > self._x_axis[np.max(x_range)], False, True)

        self._pos_axis.array = self._pos_axis.array[mask]
        self._int_axis.array = self._int_axis.array[mask]

        self._x_axis.array, self._data.array = self._truncate_one_dimension(x_range, self._x_axis.array)
        self._recalculate = False

        return self

    @classmethod
    def from_file(cls, path, **kwargs):
        if kwargs.get('lineshape') is None:
            kwargs['lineshape'] = 'lorentz'

        file = CalculationParser(path)

        if not 'calc_type' in kwargs:
            kwargs['calc_type'] = file.calc_type

        return cls(file.pos, file.int, file.x_unit, file.y_unit, **kwargs)

    def __init__(self,
                 positions:     np.ndarray,
                 intensities:   np.ndarray,
                 pos_unit:      str = 'wn',
                 int_unit:      str = 'f_osc',
                 calc_type:     str = 'ir',
                 lineshape:     str = 'lorentz',
                 **kwargs):

        self._int_axis = OneDimensionalData(intensities, int_unit)
        self._lineshape_params = self.__ENVELOPE.get(lineshape).copy()
        self._lineshape_params.update(kwargs)
        self._lineshape = lineshape
        self._recalculate = False
        self._scaling = 1

        if len(np.shape(intensities)) != 1:
            raise ValueError("Data Object only accepts one-dimensional arrays.")

        if pos_unit in ('wn', 'ev'):
            self._pos_axis = EnergyAxis(positions, pos_unit)
        if pos_unit == 'wl':
            self._pos_axis = WavelengthAxis(positions, pos_unit)

        if calc_type in self.__IR.get('kw'):
            self._params = self.__IR
        elif calc_type in self.__UVVIS.get('kw'):
            if pos_unit == 'wn':
                self._params = self.__UVVIS_WN
            elif pos_unit == 'ev':
                self._params = self.__UVVIS_EV
            else:
                self._params = self.__UVVIS
        else:
            ValueError(f'calc_type: {calc_type} not recognized!')

        x = np.arange(self._params['min'], self._params['max'] + self._params['step'], self._params['step'])
        y = self._calc_envelope(x)
        super().__init__(x, y, x_unit=pos_unit, data_unit=int_unit)

    def _calc_envelope(self, x: np.ndarray):
        shape = self._calc_lineshape()

        envelope = np.zeros_like(x)

        for p, i in zip(self._pos_axis, self._int_axis):
            envelope[np.argmin(np.abs(x - p))] = i

        envelope = convolve(envelope, shape, mode='same')
        #envelope = np.max(self._int_axis.array) * envelope / np.max(envelope)

        return envelope

    def _calc_lineshape(self):
        x = np.arange(-self._params['max'], self._params['max'] + self._params['step'], self._params['step'])

        if (fn := self.__ENV_FN.get(self.lineshape)) is None:
            raise Exception('Your function type has to be either:\n'
                            '-    gaussian\n'
                            '-    lorentz\n'
                            '-    pseudo-voigt.\n'
                            '-    lognormal\n' 
                            'Leaving it empty returns gauss-type function')

        nu0 = 0
        if self.lineshape == 'lognormal':
            x = x + self._params['max']
            nu0 = self._params['max']

        shape = fn(nu=x, s0=1, nu0=nu0, **self._lineshape_params)

        if self.lineshape == 'lognormal':
            shape = shape[::-1]

        return shape


if __name__ == '__main__':
    name = r"N:\Calc\FcN3\S0\tddft\FcCOOtBuN3_S0.mpi.8.out"
    calc = Calculation.from_file(name)

    print(calc.pos.array)