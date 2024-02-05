from .coreAbstractAxis import AbstractAxis

import numpy as np


class TimeAxis(AbstractAxis):
    """
    Represents a time axis. Additional creation routines are provided if the time axis is not recorded in the data file,
    like for RapidScan or StepScan measurements.
    """

    __magnitude = {
        'fs': 15,
        'ps': 12,
        'ns': 9,
        'us': 6,
        'ms': 3,
        's': 0
    }

    ndim = 1

    def shift_by(self, amount, anchor=None):
        """"""
        self._array += amount

    def convert_to(self, axis_type=None):
        """"""
        self._array *= 10 ** (self.__magnitude[axis_type] - self.__magnitude[self.unit])

    @classmethod
    def from_file(cls, path, unit='s', sep='\t'):

        with open(path, 'r') as file:
            fl = file.readlines()

        t = []
        for line in fl:
            t.append(float(line.split(sep)[0]))

        return cls(np.array(t), unit)

    @classmethod
    def from_parameters(cls, steps: int, step_size: float, time_zero_step: 1, unit='s'):
        """
        :param steps: The amount of steps the time axis has.
        :param step_size: the step size for a regularly spaced time axis.
        :param time_zero_step: The (zero-based) index of the time-zero step.
        :param unit: The unit of the time axis.

        Creates a regularly spaced timeaxis.
        """

        return cls(
            np.array([(s - time_zero_step) * step_size for s in range(steps)]),
            unit=unit
        )

    @classmethod
    def from_average(cls, list):
        pass
