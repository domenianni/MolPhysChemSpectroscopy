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
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from pySpec.SpecAnalyze.LifetimeDensityAnalysis.analyzeLDAAxes import TauAxis, AlphaAxis
from pySpec.SpecAnalyze.LifetimeDensityAnalysis.analyzeLDAStatistics import LdaStatistics

from copy import deepcopy


class LifetimeDensity:

    @property
    def alphas(self):
        return self._alphas

    @property
    def tau(self):
        return self._tau

    @property
    def x(self):
        return self._spec.x

    @property
    def y(self):
        self._check_for_updates()

        if self._result is None:
            self.tiks()

        return self._result[:, :, self.stats.best_idx]

    def _check_for_updates(self):
        if self._alphas.updated:
            self._alphas.updated = False
            self._result   = None
            self._fit      = None
            self._stats    = None

        if self._tau.updated:
            self._tau.updated = False
            self._d_matrix = None
            self._l_matrix = None
            self._result   = None
            self._fit      = None
            self._stats    = None

    @property
    def d_matrix(self):
        self._check_for_updates()

        if self._d_matrix is None:
            def model(s, time, ks):
                arr = [-ks[0] * s[0], ks[0] * s[0] - ks[1] * s[1]]
                return arr

            self._d_matrix = np.zeros([self._spec.t.size, self._tau.size])

            for i, t in enumerate(self._tau):
                if i > 0 and self._sequential_model is True:
                    ks = 1. / self._tau[np.array([i - 1, i])]
                    res = odeint(model, [1, 0], self._spec.t.array, (ks,))
                    self._d_matrix[:, i] = res[:, 1]
                else:
                    self._d_matrix[:, i] = (np.exp(-self._spec.t.array / t)).reshape(-1)

        return self._d_matrix

    @property
    def l_matrix(self):
        self._check_for_updates()

        if self._l_matrix is None:
            self._l_matrix = np.identity(np.shape(self.d_matrix)[1])
            b = np.ones(np.shape(self.d_matrix)[1])
            np.fill_diagonal(self._l_matrix[:, 1:], -b)

        return self._l_matrix
    
    @property
    def fit(self):
        self._check_for_updates()

        if self._fit is None:
            self._fit = np.empty(self._spec.y.shape + (np.shape(self._result)[2],))
            for i in range(np.shape(self._result)[2]):
                self._fit[:, :, i] = np.transpose(self.d_matrix.dot(self._result[:, :, i]))

        return self._fit[:, :, self.stats.best_idx]

    @property
    def results(self):
        self._check_for_updates()

        if self._result is None:
            self.tiks()

        return self._result

    @property
    def stats(self):
        self._check_for_updates()

        if self._result is None:
            self.tiks()

        if self._stats is None:
            self._stats = LdaStatistics(self)

        return self._stats

    def __init__(self, spec,
                 tau_limit=None, tau_n=100,
                 alpha_limit=(0.1, 5), alpha_n=100,
                 method='tik', alpha_scaling='log', seq_model=False):

        self._method           = method
        self._alpha_spacing    = alpha_scaling
        self._sequential_model = seq_model

        self._alphas = AlphaAxis(alpha_n, alpha_limit, alpha_scaling)

        if tau_limit is None:
            tau_limit = (spec.t[np.argmin(np.abs(spec.t))+1], np.max(spec.t) * 1000)
        self._tau  = TauAxis(tau_n, tau_limit)

        self._spec = deepcopy(spec)
        self._spec.orient_data('x')
        self._spec.y = np.nan_to_num(self._spec.y)

        self._d_matrix  = None
        self._l_matrix  = None

        self._result = None
        self._fit    = None
        self._stats  = None

    @staticmethod
    def inversesvd(d_matrix, k=-1):
        """ Returns the inverse of matrix computed via SVD.

        Parameters
        ----------
        d_matrix : np.array
            Matrix to be inverted
        k : int
            Point of truncation. If *-1* then all singular values are used.

        Returns
        -------
        v : np.array
            Inverse of input matrix.
        """

        # noinspection PyTupleAssignmentBalance
        u, s, vt = svd(d_matrix, full_matrices=False)

        if k == -1:
            k = len(s)

        s = 1 / s
        sig = np.array([s[i] if i < k else 0 for i in range(len(s))])
        sig = np.diag(sig)

        ut = np.transpose(u)
        v = np.transpose(vt)

        return v.dot(sig).dot(ut)

    def _tik(self, alpha):
        """ Function for Tikhonov regularization:
            min_x ||Dx - A|| + alpha*||Lx||
            D-matrix contains exponential profiles,
            x are prefactors/amplitudes,
            A is the dataset,
            alpha is the regularization factor and
            L is the identity matrix.

            Details can be found in
            Dorlhiac, Gabriel F. et al.
            "PyLDM-An open source package for lifetime density analysis
            of time-resolved spectroscopic data."
            PLoS computational biology 13.5 (2017)

        Parameters
        ----------
        alpha : float
            Regularization factor

        Returns
        -------
        x_k : np.array
            Expontential prefactors/amplitudes.
        """

        # constructing augmented D- and A-matrices.
        # d_aug = (D, sqrt(alpha)*L)
        # a_aug = (A, zeros)
        if alpha != 0:
            d_aug = np.concatenate((self.d_matrix, alpha ** 0.5 * self.l_matrix))
            a_aug = np.concatenate(
                (self._spec.y.array, np.zeros([np.shape(self._spec.y.array)[0], len(self.l_matrix)])),
                axis=1)
        else:
            d_aug = self.d_matrix
            a_aug = self._spec.y.array

        d_tilde = self.inversesvd(d_aug)
        x_k = d_tilde.dot(np.transpose(a_aug))
        return x_k

    def tiks(self):
        """ Wrapper for computing LDA for various alpha values.
            Parallelization makes execution actually slower. I suspect that the svd numpy method already optimizes
            CPU usage.

        Returns
        -------
        x_k : np.array
            3D matrix of expontential prefactors/amplitudes.
        """
        self._result = np.empty([np.shape(self.d_matrix)[1], np.shape(self._spec.y.array)[0], len(self._alphas)])
        for i, alpha in enumerate(self._alphas):
            x_k = self._tik(alpha)
            self._result[:, :, i] = x_k

        return self._result

    def tsvd(self, k):
        """ Truncated SVD for LDA. Similar to Tikhonov regularization
            but here we have a clear cut-off after a specified singular value.

            Details can be found in
            Hansen PC.
            The truncated SVD as a method for regularization.
            Bit. 1987

        Parameters
        ----------
        k : int
            Cut-off for singular values.
        """

        d_tilde = self.inversesvd(self.d_matrix, k)
        self._result = d_tilde.dot(np.transpose(self._spec.y.array))


if __name__ == '__main__':
    from pySpec import *
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    data = TransientSpectrum.from_file(r"N:\DataProcessing\Stuttgart\BTC-6-PtN3\UVmIR/BTC-6-PtN3_UVmIR_266nm.dat", x_unit='wn', t_unit='ps', data_unit='mdod')

    data.y = np.nan_to_num(data.y)
    data.truncate_to(t_range=[1, 3e8])

    l = LifetimeDensity(data, alpha_limit=[1e-5, 100], alpha_n=100, tau_n=150)

    plt.figure()
    plt.pcolormesh(l.tau, l.x, l.y.T,
                   norm=TwoSlopeNorm(0, -np.max(np.abs(l.y)), np.max(np.abs(l.y))))
    plt.xscale('log')
    plt.xlim(1, 1e8)

    l.stats.plot_statistics()

    plt.figure()
    plt.pcolormesh(data.t, data.x, l.fit, norm=TwoSlopeNorm(0, -1, 1))
    plt.xscale('log')
    plt.xlim(1, 1e8)

    plt.show()

    print('')
