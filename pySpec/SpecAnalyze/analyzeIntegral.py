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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum


class Integral(TransientSpectrum):

    def __init__(self, data: TransientSpectrum):
        data.orient_data('x')

        y = self._calculate_integrals(data)

        super().__init__(np.array([-1, 1]), data.t, y,
                         x_unit=data.x.unit, t_unit=data.t.unit, data_unit=data.y.unit)

    def _calculate_integrals(self, data: TransientSpectrum):
        pos = np.where(data.y.array > 0, data.y.array, 0)
        neg = np.where(data.y.array < 0, data.y.array, 0)

        area_neg = - trapezoid(neg, data.x.array, axis=0)
        area_pos = trapezoid(pos, data.x.array, axis=0)

        return np.array([area_neg, area_pos])

    def plot(self):
        from ..SpecPlot.plotStaticMethods import lighten_color
        from ..SpecPlot.plotMPCFigure import MPCFigure

        fig, ax = plt.subplots(FigureClass=MPCFigure)
        ax2 = ax.twinx()

        ax.plot(
            self.t, self.transient['-1'].y,
            marker='o', ls='', color='tab:blue', label='Bleach', markerfacecolor=lighten_color('tab:blue', 0.5)
        )
        ax.plot(
            self.t, self.transient['1'].y,
            marker='o', ls='', color='tab:red', label='Absorption', markerfacecolor=lighten_color('tab:red', 0.5)
        )
        ax2.plot(
            self.t, 100 * self.transient['1'].y / self.transient['-1'].y,
            marker='o', ls='', color='tab:green', markerfacecolor=lighten_color('tab:green', 0.5)
        )

        ax.set_xscale('symlog', linscale=2)
        ax.set_xlim(0, 3e8)
        ax.set_ylim(0)

        ax.set_xlabel(self.t.label)

        ax2.spines['right'].set_color('tab:green')
        ax2.set_ylim(0, 150)
        ax2.tick_params(which='both', color='tab:green', labelcolor='tab:green')

        ax.set_ylabel(r'$\int \Delta$ mOD')
        ax2.set_ylabel(r'% Absorption / Bleach', color='tab:green')
        ax.legend()

        return fig, (ax, ax2)
