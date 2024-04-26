import numpy as np
import re
from pathlib import Path


class CalculationParser:

    __UVVIS = {
        'type': 'uvvis',
        'upper_border': "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        'lower_border': "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS",
        'value_format': r"^\s+[0-9]+\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+",
        'pos_1': 2,
        'pos_2': 3,
        'x_unit': 'ev',
        'y_unit': 'f_osc'
    }

    __IR = {
        'type': 'ir',
        'upper_border': "IR SPECTRUM",
        'lower_border': r"^The first frequency considered to be a vibration is [0-9]*",
        'value_format': r"^\s*[0-9]+:",  # r"^\s+[0-9]+:\s+[0-9]+.[0-9]+\s+[0-9]+.[0-9]+\s+\(",
        'pos_1': 1,
        'pos_2': 2,
        'x_unit': 'wn',
        'y_unit': 'f_osc'
    }

    __TM_IR = {
        'type': 'ir',
        'pos': slice(27, 34),
        'int': slice(40, 50),
        'x_unit': 'wn',
        'y_unit': 'f_osc'
    }

    __TM_UV = {
        'type': 'uvvis',
        "pos": slice(62, 71),
        "int": slice(91, 102),
        'x_unit': 'wl',
        'y_unit': 'f_osc'
    }

    @property
    def pos(self):
        return self._pos

    @property
    def int(self):
        return self._int

    @property
    def x_unit(self):
        return self._params.get('x_unit')

    @property
    def y_unit(self):
        return self._params.get('y_unit')

    @property
    def calc_type(self):
        return self._params.get('type')

    def __init__(self,
                 file: str or None = None
                 ) -> None:

        self._file = Path(file)

        self._pos, self._int = self._import_file()

    def _import_file(self):
        if re.search('ORCA', open(self._file, 'r').read()):
            if re.search('ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR', open(self._file, 'r').read()):
                self._params = self.__UVVIS
            else:
                self._params = self.__IR
            pos, ins = self._read_file_orca()
        else:
            if re.search('# Excitation spectrum', open(self.__file, 'r').read()):
                self._params = self.__TM_UV
            else:
                self._params = self.__TM_IR
            pos, ins = self._read_file_tm()

        return pos, ins

    def _read_file_tm(self):
        pos = []
        ins = []

        file = open(self._file, "r").readlines()
        for line in file:
            if line[0] == "#" or line[0] == "$":
                continue
            else:
                pos.append(float(line[self._params["pos"]]))
                ins.append(float(line[self._params["int"]]))

        return np.array(pos), np.array(ins)

    def _read_file_orca(self):
        i = 0
        with open(self._file) as fh:
            for Search in fh:
                i += 1
                if re.search(self._params['upper_border'], Search):
                    upperBound = i - 1
        # search for lower bound in text file
        i = 0
        with open(self._file) as fh:
            for Search in fh:
                i += 1
                if re.search(self._params['lower_border'], Search):
                    lowerBound = i - 1
        # cut out relevant section from data file
        fg = open(self._file).readlines()
        fg = fg[upperBound:lowerBound]
        # extract relevant data points

        pos = []
        ins = []
        for line in fg:
            if re.search(self._params['value_format'], line):
                line = line.split()
                pos.append(float(line[self._params['pos_1']]))
                ins.append(float(line[self._params['pos_2']]))

        return np.array(pos), np.array(ins)
