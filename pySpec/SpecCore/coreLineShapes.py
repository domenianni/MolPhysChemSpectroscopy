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
from scipy.special import erf
from scipy.signal import convolve
from functools import cache
from scipy.interpolate import interp1d

def wl_wn(x):
    return 1e7 / x

@cache
def sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def gaussian(nu, s0, nu0, fwhm):
    # Adapted from >> https://mathworld.wolfram.com/GaussianFunction.html << 28.09.2023
    return (s0 / (sigma(fwhm) * np.sqrt(2 * np.pi))) * np.exp(-1/2 * ((nu - nu0) / sigma(fwhm)) ** 2)


def skewed_gaussian(nu, s0, nu0, fwhm, alpha):
    return gaussian(nu, s0, nu0, fwhm) / 2 * ( 1 + erf(- alpha * ((nu-nu0)/sigma(fwhm))))


def lorentzian(nu, s0, nu0, fwhm):
    # Adapted from >> https://mathworld.wolfram.com/LorentzianFunction.html << 28.09.2023
    gamma = fwhm / 2
    return (s0 / np.pi) * (gamma / ( (nu - nu0)**2 + gamma**2 ))


def pseudo_voigt(nu, s0, nu0, fwhm, xi):
    return xi * gaussian(nu, s0, nu0, fwhm) + (1 - xi) * lorentzian(nu, s0, nu0, fwhm)


def voigt(nu, s0, nu0, fwhm_gauss=10, fwhm_lorentz=10):
    OVERSAMPLING = 3
    mean = np.mean(nu)
    xnew = np.linspace(np.min(nu) - mean, np.max(nu) - mean, num=OVERSAMPLING * np.size(nu))

    gauss = gaussian(xnew, s0, nu0 - mean, fwhm_gauss)
    lorentz = lorentzian(xnew, s0, 0, fwhm_lorentz)

    conv = convolve(gauss, lorentz, mode='same')

    voigt = interp1d(xnew + mean, conv)
    return voigt(nu)


if __name__ == '__main__':
    pass
