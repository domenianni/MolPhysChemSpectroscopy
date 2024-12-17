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

from pySpec.SpecCore.coreLineShapes import gaussian, skewed_gaussian, lorentzian, voigt, pseudo_voigt

from lmfit import Model


class Gaussian(Model):

    parameter = {
        's0': 1,
        'nu0': 2000,
        'fwhm': 10
    }

    @staticmethod
    def function(nu, s0, nu0, fwhm):
        return gaussian(nu, s0, nu0, fwhm)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='Gaussian',
                         **kws)

        self.set_param_hint('s0', value=self.parameter['s0'])
        self.set_param_hint('nu0', value=self.parameter['nu0'])
        self.set_param_hint('fwhm', min=1e-7, value=self.parameter['fwhm'])


class Lorentzian(Model):
    parameter = {
        's0': 1,
        'nu0': 2000,
        'fwhm': 10
    }

    @staticmethod
    def function(nu, s0, nu0, fwhm):
        return lorentzian(nu, s0, nu0, fwhm)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='Lorentzian',
                         **kws)

        self.set_param_hint('s0', value=self.parameter['s0'])
        self.set_param_hint('nu0', value=self.parameter['nu0'])
        self.set_param_hint('fwhm', min=1e-7, value=self.parameter['fwhm'])


class Voigt(Model):
    parameter = {
        's0': 1,
        'nu0': 2000,
        'fwhm': 10
    }

    @staticmethod
    def function(nu, s0, nu0, fwhm_g, fwhm_l):
        return voigt(nu, s0, nu0, fwhm_g, fwhm_l)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='Lorentzian',
                         **kws)

        self.set_param_hint('s0', value=self.parameter['s0'])
        self.set_param_hint('nu0', value=self.parameter['nu0'])
        self.set_param_hint('fwhm_g', min=1e-7, value=self.parameter['fwhm'])
        self.set_param_hint('fwhm_l', min=1e-7, value=self.parameter['fwhm'])


class PseudoVoigt(Model):
    parameter = {
        's0': 1,
        'nu0': 2000,
        'fwhm': 10,
        'xi': 0.5
    }

    @staticmethod
    def function(nu, s0, nu0, fwhm, xi):
        return pseudo_voigt(nu, s0, nu0, fwhm, xi)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='Lorentzian',
                         **kws)

        self.set_param_hint('s0', value=self.parameter['s0'])
        self.set_param_hint('nu0', value=self.parameter['nu0'])
        self.set_param_hint('fwhm', min=1e-7, value=self.parameter['fwhm'])
        self.set_param_hint('xi', max=1, min=0, value=self.parameter['xi'])
