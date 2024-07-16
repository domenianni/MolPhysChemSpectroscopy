import numpy as np
from scipy.integrate import trapezoid

from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum


class Integral(TransientSpectrum):

    def __init__(self, data: TransientSpectrum, supersampling_rate=4):
        self._supersampling_rate = supersampling_rate
        x, y = self.__supersampling(data.x.array, data.t.array, data.y.array)

        data = TransientSpectrum(x, data.t, y, data.x.unit, data.t.unit, data.y.unit)
        data.orient_data('x')

        y = self._calculate_integrals(data)

        super().__init__(np.array([-1, 1]), data.t, y,
                         x_unit=data.x.unit, t_unit=data.t.unit, data_unit=data.y.unit)

    def __supersampling(self, x, t, y):
        x_steps = [i * self._supersampling_rate for i in range(len(x))]
        x_gen = [i for i in range(self._supersampling_rate * len(x))]

        x_new = np.interp(x_gen, x_steps, x, left=0, right=0)

        y_new = []
        for i, _ in enumerate(t):
            y_new.append(np.array(
            np.interp(x_gen, x_steps, y[:, i], left=0, right=0)
        ))

        return x_new, np.array(y_new)

    def _calculate_integrals(self, data: TransientSpectrum):
        pos = np.where(data.y.array > 0, data.y.array, 0)
        neg = np.where(data.y.array < 0, data.y.array, 0)

        area_neg = - trapezoid(neg, data.x.array, axis=0)
        area_pos = trapezoid(pos, data.x.array, axis=0)

        return np.array([area_neg, area_pos])
