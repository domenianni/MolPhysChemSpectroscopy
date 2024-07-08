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
from scipy import linalg
from scipy.optimize import minimize
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

from pySpec.SpecCore.SpecCoreSpectrum import TransientSpectrum


class DiscontinuityRemoval:

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: TransientSpectrum):
        if not isinstance(value, TransientSpectrum):
            raise ValueError(f"{value} is not of type {type(TransientSpectrum)}!")

        self._data = value

    def __init__(self,
                 data: TransientSpectrum,
                 stitching_blocks: int = 4):

        self._data_raw = deepcopy(data)
        self._data_raw.orient_data('x')
        self._data = deepcopy(self._data_raw)

        self._stitching_blocks = stitching_blocks

    def _get_data_idx(self, side, pixel_idx, intact_pixel):
        if side == 'lower':
            discontinuity_idx = [pixel_idx + i for i in range(self._stitching_blocks)]
            reference_idx = [pixel_idx - i for i in range(1, self._stitching_blocks * intact_pixel + 1)]
            reference_idx.sort()
        elif side == 'upper':
            discontinuity_idx = [pixel_idx + i for i in range(self._stitching_blocks)]
            reference_idx = [pixel_idx + self._stitching_blocks + i for i in range(1, self._stitching_blocks * intact_pixel + 1)]
        else:
            raise ValueError()

        return discontinuity_idx, reference_idx

    def _calc_mean_position(self, discontinuity_x, reference_x):
        if np.min(discontinuity_x) > np.max(reference_x):
            return (np.min(discontinuity_x) + np.max(reference_x)) / 2
        else:
            return (np.min(reference_x) + np.max(discontinuity_x)) / 2

    def _prepare_data(self, discontinuity_idx, reference_idx):
        discontinuity_x = np.atleast_2d(self._data.x.array[discontinuity_idx]).T
        discontinuity_y = np.atleast_2d(self._data.y.array[discontinuity_idx])
        reference_x = np.atleast_2d(self._data.x.array[reference_idx]).T
        reference_y = np.atleast_2d(self._data.y.array[reference_idx])

        discontinuity_x -= self._calc_mean_position(discontinuity_x, reference_x)
        reference_x -= self._calc_mean_position(discontinuity_x, reference_x)

        return reference_x, reference_y, discontinuity_x, discontinuity_y

    def one_sided(self,
                  pixel_idx,
                  side='lower',
                  intact_pixel=1,
                  rank_pixel=2,
                  rank_ref=2,
                  derivative=1,
                  start_params=[1.3, 0.04]):

        discontinuity_idx, reference_idx = self._get_data_idx(side, pixel_idx, intact_pixel)
        reference_x, reference_y, discontinuity_x, discontinuity_y = self._prepare_data(discontinuity_idx, reference_idx)

        def kernel(beta):
            return self.ninevski_oleary(reference_x,
                                        reference_y,
                                        rank_ref,
                                        discontinuity_x,
                                        beta[0] * discontinuity_y + beta[1],
                                        rank_pixel,
                                        derivative)[2]

        res = minimize(kernel, x0=np.array(start_params), method='cobyla', bounds=[(1.0, np.inf), (None, None)])
        print(res)

        return self.manipulate_pixel(pixel_idx, res.x[0], res.x[1]), res.x

    def two_sided(self,
                  pixel_idx,
                  intact_pixel=1,
                  rank_pixel=2,
                  rank_ref=2,
                  derivative=1,
                  start_params=[1.3, 0.04]):

        d_idx_lower, r_idx_lower = self._get_data_idx('lower', pixel_idx, intact_pixel)
        r_x_lower, r_y_lower, d_x_lower, d_y_lower = self._prepare_data(d_idx_lower, r_idx_lower)

        d_idx_upper, r_idx_upper = self._get_data_idx('upper', pixel_idx, intact_pixel)
        r_x_upper, r_y_upper, d_x_upper, d_y_upper = self._prepare_data(d_idx_upper, r_idx_upper)

        def kernel(beta):
            return np.sqrt(self.ninevski_oleary(r_x_lower, r_y_lower, rank_ref,
                                                d_x_lower, beta[0] * d_y_lower + beta[1], rank_pixel,
                                                derivative)[2]**2 +
                           self.ninevski_oleary(r_x_upper, r_y_upper, rank_ref,
                                                d_x_upper, beta[0] * d_y_upper + beta[1], rank_pixel,
                                                derivative)[2]**2)

        res = minimize(kernel, x0=np.array(start_params), method='cobyla', bounds=[(0.5, np.inf), (None, None)])
        print(res)

        return self.manipulate_pixel(pixel_idx, res.x[0], res.x[1]), res.x

    def manipulate_pixel(self, pixel: int, f_sc, offset):
        sb = self._stitching_blocks
        y = self._data.y.array

        diag = np.diag(
            np.array([f_sc if i in range(pixel * sb, (pixel+1) * sb) else 1 for i in range(len(y))]))

        left = diag @ y
        right = np.array([offset if i in range(pixel * sb, (pixel+1) * sb) else 0 for i in range(len(y))])

        return left + right[:, np.newaxis]

    @staticmethod
    def ninevski_oleary(x_l, y_l, d_l, x_r, y_r, d_r, n, write=False):  # f_sc, offset, return_arg="tn_fg"
        """
        An algorithm to detect Cn discontinuities based on the publication by Ninevski and O'Leary
        https://doi.org/10.48550/arXiv.1911.12724

        :param x_l:         the left-sided x values
        :param y_l:         the left-sided y values
        :param d_l:         the polynomial degree of the Taylor approximation to the left-sided data
        :param x_r:         the right-sided x values
        :param y_r:         the right-sided y values
        :param d_r:         the polynomial degree of the Taylor approximation to the right-sided data
        :param n:           check for discontinuity in C^n
                            The Taylor polynomials are constructed such tha C^n-1 and lower derivatives are forced to be
                            continuous!
        :return:            tn_fg: the difference in the Taylor coefficients (alpha_n - beta_n)
                            E_a: the error of approximation (deviation between fits and corresponding actual data)
                            E_e: the error of extrapolation (deviation between extrapolated fits and actual data)
        """

        # build Vandermonde matrices
        V_L = np.vander(x_l.squeeze(), d_l + 1)
        V_R = np.vander(x_r.squeeze(), d_r + 1)

        # y_r = np.vander(np.squeeze(y_r), 2) @ np.array([[f_sc], [offset]])

        y = np.vstack((y_l, y_r))

        # build V matrix from Vandermonde matrices
        V = linalg.block_diag(V_L, V_R)

        # build C matrix
        C = np.hstack((np.zeros((n, d_l + 1 - n)), np.identity(n), np.zeros((n, d_r + 1 - n)), -np.identity(n)))
        C = np.atleast_2d(C)

        # calculate orthonormal vector basis of null(C),
        # i.e. basis vector for the vector space whose elements multiplied with C always return the zero vector
        N = linalg.null_space(C)

        gamma = N @ linalg.pinv(V @ N) @ y

        # split to extract alpha and beta
        alpha = np.atleast_2d(gamma[:d_l + 1])
        beta = np.atleast_2d(gamma[d_l + 1:])

        tn_fg = abs(alpha[-(n + 1)][0] - beta[-(n + 1)][0])

        # calculate errors
        # c.f. Section 4.1 (approximation error)
        E_a = (y_l - V_L @ alpha).T @ (y_l - V_L @ alpha) + (y_r - V_R @ beta).T @ (y_r - V_R @ beta)

        # c.f. Section 4.3 (extrapolation error)
        E_e = (y_l - V_L @ beta).T @ (y_l - V_L @ beta) + (y_r - V_R @ alpha).T @ (y_r - V_R @ alpha)

        if write:
            print(f"Taylor coefficients (alpha):\n{alpha}")
            print(f"Taylor coefficients (beta):\n{beta}")

            print(f"tn_fg = {tn_fg}")
            print(f"approximation error (E_a) = {E_a[0][0]}")
            print(f"extrapolation error (E_e) = {E_e[0][0]}")

        return tn_fg, E_a[0][0], E_e[0][0]

    @staticmethod
    def plot(d, vmin=-3, vmax=3):
        data = deepcopy(d)
        data.sort()
        data.orient_data('x')

        fig, ax = plt.subplots(1, 1)

        ax.pcolormesh(data.t, data.x, data.y, norm=TwoSlopeNorm(0, vmin, vmax))

        for i in range(0, len(data.x)-3, 4):
            x_low = data.x[i]
            height = abs(x_low - data.x[i + 3])

            rect = patches.Rectangle((-2, x_low - 0.1*height), 1, 1.1*height)
            ax.add_patch(rect)
            ax.annotate(int(i / 4), (-1.5, x_low + height / 2), color='w', weight='bold',
                        fontsize=6, ha='center', va='center')

        ax.set_xscale('symlog')

        return fig

    def plot_correct(self):
        return self.plot(self._data)

    def plot_raw(self):
        return self.plot(self._data_raw)


if __name__ == '__main__':
    from pySpec import *
    from glob import glob

    files = glob(r"")

    f = ImportUVmIR.from_files(files)
    f.correct_stitch()
    f.cross_correlate()
    f.subtract_prescans(-2)
    f.convert_all_to('wn')
    f.apply_baseline([2100, 3000])

    g = f.average

    d = DiscontinuityRemoval(g, 4)

    p = d.plot_raw()
    plt.show()

    d.two_sided(19,
               intact_pixel=3,
               rank_pixel=6,
               rank_ref=6,
               derivative=2,
               start_params=[1.3, 0.04])

    d.two_sided(15,
                intact_pixel=3,
                rank_pixel=6,
                rank_ref=6,
                derivative=2,
                start_params=[1.3, 0.04])

    p = d.plot_correct()
    plt.show()

    plt.plot(g.x, g.spectrum['1'].y)
    plt.plot(d.data.x, d.data.spectrum['1'].y)
    plt.show()

