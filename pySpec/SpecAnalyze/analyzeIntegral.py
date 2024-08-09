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
            marker='o', ls='', color='tab:blue', markersize=6, label='Bleach', markerfacecolor=lighten_color('tab:blue', 0.5)
        )
        ax.plot(
            self.t, self.transient['1'].y,
            marker='o', ls='', color='tab:red', markersize=6, label='Absorption', markerfacecolor=lighten_color('tab:red', 0.5)
        )
        ax2.plot(
            self.t, 100 * self.transient['1'].y / self.transient['-1'].y,
            marker='o', ls='', color='tab:green', markersize=6, markerfacecolor=lighten_color('tab:green', 0.5)
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
