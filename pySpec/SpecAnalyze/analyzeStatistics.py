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
import scipy.stats as sp
from copy import deepcopy
import matplotlib.pyplot as plt

from Spectroscopy.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum


class StatisticalAnalysisMeasurements:

    def __init__(self,
                 data_list: list[TransientSpectrum],
                 t0: float = 0
                 ) -> None:

        self._data_list = [deepcopy(data) for data in data_list]

        for data in self._data_list:
            data.sort()
            data.truncate_to(t_range=[t0, np.max(data.t)])

        self.m_time = self._create_timeaxis()

        self.stats, self.spm_p, self.delta = self._regress_data(self.m_time)
        self.variance, self.standard_deviation, self.sem = self._calculate_var(
            np.array([data.y.array for data in self._data_list])
        )

    def _create_timeaxis(self) -> np.ndarray:
        return np.arange(0, len(self._data_list))

    @staticmethod
    def _calculate_var(y: np.ndarray) -> (float, float, float):
        return np.nanvar(y, axis=(1, 2)), np.nanstd(y, axis=(1, 2)), np.nanstd(y, axis=(1, 2))/len(y)

    def _calculate_delta(self) -> list:
        return [np.nanmean(abs(data.y.array - self._data_list[0].y.array)) for data in self._data_list[1:]]

    def _regress_data(self, m_time: np.ndarray) -> (tuple, float, list):
        delta = self._calculate_delta()

        reg = sp.linregress(m_time[1:], delta, alternative='greater')
        _, spm_p = sp.spearmanr(m_time[1:], delta)
        return reg, spm_p, delta

    @property
    def output_message(self) -> str:
        message = f"""File No.                            = {[f'{x:.3f}' + ' ' for x in self.m_time]}
Variance of the data sets           = {[f'{x:.3f}' + ' ' for x in self.variance]}
Standard Deviation of the data sets = {[f'{x:.3f}' + ' ' for x in self.standard_deviation]}
Standard Error of the Mean (SEM)    = {[f'{x:.3f}' + ' ' for x in self.sem]}\n\n"""

        message += f"""Regression Results Total Difference
slope            = {self.stats.slope:.3f}
intercept        = {self.stats.intercept:.3f}
r^2              = {self.stats.rvalue**2:.3f}
p-Value          = {self.stats.pvalue:.3f}
Spearman p-Value = {self.spm_p:.3f}
Std. Error       = {self.stats.stderr:.3f}
Itcpt. Error     = {self.stats.intercept_stderr:.3f}
"""

        if self.stats.pvalue >= 0.05 and self.spm_p >= 0.05:
            message += f"""
Hypothesis of linear dependency rejected!
(p-Value            = {self.stats.pvalue:.3f})
(Spearman p-Value   = {self.spm_p:.3f})
"""

        elif self.stats.pvalue < 0.05 or self.spm_p < 0.05:
            message += f"""
Statistically relevant linear dependency possible!
(p-Value            = {self.stats.pvalue:.3f})
(Spearman p-Value   = {self.spm_p:.3f})
Please check your Data!
"""
        else:
            raise ValueError("p-Values are faulty!")

        return message

    @property
    def plot(self):
        fig = plt.figure( figsize = (6,4) )
        grid = plt.GridSpec(2,4, hspace=0.05, wspace=0.05)

        std = fig.add_subplot(grid[0,:3])
        std.plot(self.m_time, self.standard_deviation, marker='o', ls='')
        std.text(0.05, 0.75, 'Standard\nDeviation', transform = std.transAxes)

        std_bxplt = fig.add_subplot(grid[0, 3])
        std_bxplt.boxplot(self.standard_deviation)

        bxplt = fig.add_subplot(grid[1, 3])
        bxplt.boxplot(self.delta)

        linreg = fig.add_subplot(grid[1, :3], sharex=std, sharey=bxplt)
        linreg.plot(self.m_time[1:], self.delta, marker='o', ls='')
        linreg.text(0.05, 0.75, 'Absolute\nDeviation', transform = linreg.transAxes)
        linreg.plot(self.m_time[1:], self.stats.slope*self.m_time[1:]+self.stats.intercept, ls='--')

        std.tick_params(
            direction="in", top=True, right=True, left=True, bottom=True, labelbottom=False, labelleft=True,
            labelright=False, labeltop=True, width=1.5, labelsize=12
        )
        std_bxplt.tick_params(
            direction="in", top=False, right=True, left=True, bottom=False, labelbottom=False, labelleft=False,
            labelright=True, labeltop=False, width=1.5, labelsize=12
        )
        bxplt.tick_params(
            direction="in", top=False, right=True, left=True, bottom=False, labelbottom=False, labelleft=False,
            labelright=True, labeltop=False, width=1.5, labelsize=12
        )
        linreg.tick_params(
            direction="in", top=True, right=True, left=True, bottom=True, labelbottom=True, labelleft=True,
            labelright=False, labeltop=False, width=1.5, labelsize=12
        )

        return fig


if __name__ == '__main__':
    pass
