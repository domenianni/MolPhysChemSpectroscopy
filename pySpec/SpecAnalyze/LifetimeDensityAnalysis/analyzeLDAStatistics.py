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

Based on the work of Gabriel F.Dorlhiac : https://github.com/gadorlhiac/PyLDM
                                          Access: 13.06.2025
                                          DOI: 10.1371/journal.pcbi.1005528
And David Ehrenberg: https://github.com/deef128/trtoolbox
                    Access: 13.06.2025
"""

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


class LdaStatistics:

    @property
    def gcv(self):
        return self._gcv

    @property
    def cp(self):
        return self._cp

    @property
    def k(self):
        return self._k

    @property
    def l_curve(self):
        return self._l_curve

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        if value not in self._idx.keys():
            ValueError(f"The chosen statistic has to be one of {self._idx.keys()}!")

        self._current = value

    @property
    def best_idx(self):
        return self._idx.get(self._current)

    def __init__(self, LDA):
        self._LDA = LDA

        self._gcv = np.zeros(len(self._LDA._alphas))
        self._cp = np.zeros([len(self._LDA._alphas)])

        for i, alpha in enumerate(self._LDA._alphas):
            h_matrix, s_matrix = self._calc_H_and_S(alpha)
            self._gcv[i] = self._calc_gcv(i, h_matrix)
            self._cp[i] = self._calc_cp(i, s_matrix)

        l_curve_x, l_curve_y, self._k = self._calc_lcurve()
        self._l_curve = [l_curve_x, l_curve_y]

        self._current = 'lcurve'

        self._idx = {
            'lcurve': np.argmax(self._k),
            'gcv': np.argmin(self._gcv),
            'cp': np.argmin(self._cp)
        }

    @staticmethod
    def _calc_k(lx, ly, alphas):
        da = np.gradient(alphas)
        dx = np.gradient(lx, da)
        dy = np.gradient(ly, da)
        d2x = np.gradient(dx, da)
        d2y = np.gradient(dy, da)
        k = (dx*d2y - d2x*dy)/(dx**2 + dy**2)**1.5

        return k

    def _calc_H_and_S(self, alpha):
        x = np.transpose(self._LDA.d_matrix) @ self._LDA.d_matrix + alpha * (np.transpose(self._LDA.l_matrix) @ self._LDA.l_matrix)

        # noinspection PyTupleAssignmentBalance
        u, s, vt = svd(x, full_matrices=False)

        x_inv = np.transpose(vt) @ np.diag(1/s) @ np.transpose(u)
        h_matrix = self._LDA.d_matrix @ x_inv @ np.transpose(self._LDA.d_matrix)
        s_matrix = x_inv @ (np.transpose(self._LDA.d_matrix) @ self._LDA.d_matrix)

        return h_matrix, s_matrix

    def _calc_residual(self, alpha_idx, x_idx=None):
        if x_idx is None:
            return np.sum(
                ((self._LDA.d_matrix @ self._LDA._result[:, :, alpha_idx]).T - self._LDA._spec.y.array) ** 2
            )

        return np.sum(
            ((self._LDA.d_matrix @ self._LDA._result[:, x_idx, alpha_idx]).T - self._LDA._spec.y.array[:, x_idx]) ** 2
        )

    def _calc_smoothNorm(self, alpha_idx):
        return np.sum((self._LDA.l_matrix @ self._LDA._result[:, :, alpha_idx]) ** 2) ** 0.5

    def _calc_variance(self):
        return np.sum(
            ((self._LDA.d_matrix @ self._LDA._result[:, :, 0]).T - self._LDA._spec.y.array)**2
        ) / len(self._LDA._spec.t)

    def _calc_gcv(self, alpha_idx, h_matrix):
        n = len(self._LDA._spec.t.array)
        I = np.identity(len(h_matrix))
        tr = (np.trace(I - h_matrix) / n) ** 2

        return self._calc_residual(alpha_idx) / tr

    def _calc_cp(self, alpha_idx, s_matrix):
        res = self._calc_residual(alpha_idx)
        df = np.trace(s_matrix)

        return res + 2 * self._calc_variance() * df

    def _calc_lcurve(self):
        l_curve_x = np.array([self._calc_residual(a) ** 0.5 for a in range(len(self._LDA._alphas))])
        l_curve_y = np.array([self._calc_smoothNorm(a) for a in range(len(self._LDA._alphas))])
        k = self._calc_k(l_curve_x, l_curve_y, self._LDA._alphas.array)

        return l_curve_x, l_curve_y, k

    def plot_statistics(self):
        fig, ax = plt.subplots(1,1)

        ax.plot(self._cp, color='tab:blue')
        ax.axvline(np.argmin(self._cp), color='tab:blue')
        ax.plot(self._gcv, color='tab:red')
        ax.axvline(np.argmin(self._gcv), color='tab:red')
        ax.plot(self._k, color='tab:green')
        ax.axvline(np.argmax(self._k), color='tab:green')

        ax.set_xlabel("Alpha")

        return fig, ax
