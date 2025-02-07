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
from numpy.polynomial.polynomial import Polynomial

from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum


class Integral(TransientSpectrum):
    """
    This class calculates and plots the integral of a transient spectrum, either through numerical integration
    or polynomial fitting methods. It inherits from the TransientSpectrum class.
    """

    degree = 40

    def __init__(self, data: TransientSpectrum, integral_type='numerical'):
        """
        Initializes the Integral object by calculating the integral of the provided TransientSpectrum data.

        :param data: The transient spectrum data to calculate integrals from.
        :param integral_type: The method of integration ('numerical' or 'polynomial').

        :raise ValueError: If the integral_type is neither 'numerical' nor 'polynomial'.
        """

        data.orient_data('x')

        if integral_type == 'numerical':
            y = self._calculate_integrals(data)
        elif integral_type == 'polynomial':
            y = self._calculate_polynomial(data)
        else:
            raise ValueError("integral_type has to be either:\n"
                             "- numerical\n"
                             "- polynomial\n")

        super().__init__(np.array([-1, 1]), data.t, y,
                         x_unit=data.x.unit, t_unit=data.t.unit, data_unit=data.y.unit)

    def _calculate_integrals(self, data: TransientSpectrum):
        """
        Calculates the integral of the spectrum using numerical trapezoidal integration.

        This method computes the area under the curve for both the positive and negative values of the spectrum.

        :param data: The transient spectrum data to integrate.

        :return: An array containing two values, the area of the negative and positive parts of the spectrum.
        """

        pos = np.where(data.y.array > 0, data.y.array, 0)
        neg = np.where(data.y.array < 0, data.y.array, 0)

        area_neg = - trapezoid(neg, data.x.array, axis=0)
        area_pos = trapezoid(pos, data.x.array, axis=0)

        return np.array([area_neg, area_pos])

    def _calculate_polynomial(self, data):
        """
        Calculates the integral of the spectrum using polynomial fitting.

        This method fits a polynomial to the spectrum data, evaluates it, and then integrates
        the positive and negative areas under the curve.

        :param data: The transient spectrum data to integrate.

        :return: An array containing the integrated areas for the negative and positive values of each spectrum.
        """

        val = []

        for s in data.spectrum:
            p = Polynomial.fit(s.x.array, s.y.array, self.degree)
            v = p.linspace(10000, [p.domain[0], p.domain[1]])

            val.append(
                    (
                    - trapezoid(np.where(v[1] <= 0, v[1], 0), v[0]),
                      trapezoid(np.where(v[1] >= 0, v[1], 0), v[0])
                )
            )

        return np.array(val)

    def plot(self):
        """
        Plots the integral results of the transient spectrum, including the bleach, absorption,
        and their ratio as a percentage.

        Uses the 'MPCFigure' class for custom plotting and visualizes the transient spectrum
        with three different axes: one for the integral of the bleach, one for the absorption,
        and a third for their ratio.

        Returns:
        tuple: The figure and axes used for plotting.
        """

        from pySpec.SpecPlot.plotStaticMethods import lighten_color
        from pySpec.SpecPlot.plotMPCFigure import MPCFigure

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


if __name__ == '__main__':
    pass