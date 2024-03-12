import numpy as np
import math
import re
import scipy.constants as spc
from pathlib import Path

from ..SpecCore import coreLineShapes as cLS
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from ..SpecCore.SpecCoreSpectrum.coreStickSpectrum import StickSpectrum


class ImportCalculation:
    """
    :param file:
    :Param position:
    :param intensity:
    """


    __UVVIS = {
        'min' : 0,
        'max' : 1500,
        'step' : 0.001,
        'upper_border' : "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        'lower_border' : "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS",
        'value_format' : r"^\s+[0-9]+\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+",
        'pos_1' : 2,
        'pos_2' : 3,
        'current X': 'WL'
        }

    __IR = {
        'min': 0,
        'max': 4000,
        'step': 0.1,
        'upper_border': "IR SPECTRUM",
        'lower_border': r"^The first frequency considered to be a vibration is [0-9]*",
        'value_format': r"^\s*[0-9]+:", #r"^\s+[0-9]+:\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+\(",
        'pos_1': 1,
        'pos_2': 2,
        'current X': 'WN'
        }

    __TM_IR = {
        'pos': slice(27, 34),
        'int': slice(40, 50)
    }

    __TM_UV = {
        "pos": slice(62,  71),
        "int": slice(91, 102)
    }

    @property
    def stickSpectrum(self):
        return self._stick_spectrum

    @property
    def roots(self):
        return self.__root

    def __init__(self,
                 file: str or None = None,
                 position: list[float] or None = None,
                 intensity: list[float] or None = None
                 ) -> None:

        self.__file = Path(file)

        self.uvvis = False
        self.__root = None

        self.__pos = []
        self.__int = []

        if file is not None:
            self.__import_file()
        else:
            self.__params = self.__IR
            try:
                self.__pos.extend(position)
                self.__int.extend(intensity)
            except TypeError:
                self.__pos.append(position)
                self.__int.append(intensity)

        self._stick_spectrum, self.__root = self.__prepare_stick()

    def envelope(self, type='gauss', **kwargs):
        """
        :param type:
        :param fwhm:
        :param xi:
        """
        x = np.arange(- self.__params['max'], self.__params['max'] + self.__params['step'], self.__params['step'])

        if type == 'gauss':
            y  = cLS.gaussian(nu=x, s0=1, nu0=0, **kwargs)
        elif type == 'lorentz':
            y = cLS.lorentzian(nu=x, s0=1, nu0=0, **kwargs)
        elif type == 'skewed-gaussian':
            y = cLS.skewed_gaussian(nu=x, s0=1, nu0=0, **kwargs)
        elif type == 'voigt':
            y = cLS.voigt(nu=x, s0=1, nu0=0, **kwargs)
        else:
            raise Exception('Your function type has to be either:\n'
                            '-    gauss\n'
                            '-    lorentz\n'
                            '-    pseudo-voigt.\n'
                            'Leaving it empty returns gauss-type function')

        x_sim = np.arange(self.__params['min'], self.__params['max'] + self.__params['step'], self.__params['step'])
        envelope = np.zeros_like(x_sim)

        for p, i in zip(self.__pos, self.__int):
            cen = abs(x - p).argmin()
            low = (len(x)-cen-1)
            high = int((len(x)-1)*1.5-cen+1)
            envelope = np.add(i * y[low: high], envelope)

        return Spectrum(x_sim, envelope, x_unit=self._stick_spectrum.pos.unit, data_unit=self._stick_spectrum.int.unit)

    def __import_file(self):
        if re.search('ORCA', open(self.__file, 'r').read()):
            if re.search('ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR', open(self.__file, 'r').read()):
                self.__params = self.__UVVIS
            else:
                self.__params = self.__IR
            self.__read_file_orca()
        else:
            if re.search('# Excitation spectrum', open(self.__file, 'r').read()):
                self.__params = self.__UVVIS
                self.__params['current X'] = 'EV'
                self.__read_file_tm(self.__TM_UV)
                self.uvvis = True
            else:
                self.__params = self.__IR
                self.__read_file_tm(self.__TM_IR)

    def __read_file_tm(self, positions):
        file = open(self.__file, "r").readlines()
        for line in file:
            if line[0] == "#" or line[0] == "$":
                continue
            else:
                self.__pos.append(float(line[positions["pos"]]))
                self.__int.append(float(line[positions["int"]]))
        
        self.__pos = np.array(self.__pos)
        self.__int = np.array(self.__int)

    def __read_file_orca(self) -> None:
        i = 0
        with open(self.__file) as fh:
            for Search in fh:
                i += 1
                if re.search(self.__params['upper_border'], Search):
                    upperBound = i - 1
        # search for lower bound in text file
        i = 0
        with open(self.__file) as fh:
            for Search in fh:
                i += 1
                if re.search(self.__params['lower_border'], Search):
                    lowerBound = i - 1
        # cut out relevant section from data file
        fg = open(self.__file).readlines()
        fg = fg[upperBound:lowerBound]
        # extract relevant data points
        for line in fg:
            if re.search(self.__params['value_format'], line):
                line = line.split()
                self.__pos.append(float(line[self.__params['pos_1']]))
                self.__int.append(float(line[self.__params['pos_2']]))
        self.__pos = np.array(self.__pos)
        self.__int = np.array(self.__int)

    def __read_file_orca_old(self) -> None:
        i = 0
        with open(self.__file) as fh:
            for Search in fh:
                i += 1
                if re.search(self.__params['upper_border'], Search):
                    upperBound = i - 1
        # search for lower bound in text file
        i = 0
        with open(self.__file) as fh:
            for Search in fh:
                i += 1
                if re.search(self.__params['lower_border'], Search):
                    lowerBound = i - 1
        # cut out relevant section from data file
        fg = open(self.__file).readlines()
        fg = fg[upperBound:lowerBound]
        # extract relevant data points
        for line in fg:
            if re.search(self.__params['value_format'], line):
                line = line.split()
                self.__pos.append(float(line[self.__params['pos_1']]))
                self.__int.append(float(line[self.__params['pos_2']]))
        self.__pos = np.array(self.__pos)
        self.__int = np.array(self.__int)

    def __prepare_stick(self) -> tuple[StickSpectrum, list]:
        x = []
        y = []
        root = []
        x_unit = 'wn'

        for i, p in enumerate(self.__pos):
            x.append(p)
            y.append(self.__int[i])

            root.append([i+1, p, self.__int[i]])

        if self.uvvis:
            x_unit = 'wl'

        return StickSpectrum(np.array(x), np.array(y), pos_unit=x_unit, int_unit='osc.'), root
