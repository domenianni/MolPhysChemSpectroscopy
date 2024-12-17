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
import scipy.special as sp

from lmfit import Model


class ExponentialFall(Model):

    parameter = {
        'amplitude': 1,
        'tau': 10,
        't0': 0
    }

    @staticmethod
    def function(t, amplitude, tau, t0):
        return amplitude * np.exp(-(t - t0) / tau) * np.heaviside(t - t0, 1)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='ExponentialFall',
                         **kws)

        self.set_param_hint('amplitude', value=self.parameter['amplitude'])
        self.set_param_hint('tau', min=1e-4, value=self.parameter['tau'])
        self.set_param_hint('t0', vary=False, value=self.parameter['t0'])


class ExponentialRise(Model):

    parameter = {
        'amplitude': 1,
        'tau': 10,
        't0': 0
    }

    @staticmethod
    def function(t, amplitude, tau, t0):
        return amplitude * (1 - np.exp(-(t - t0) / tau)) * np.heaviside(t - t0, 1)

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='ExponentialRise',
                         **kws)

        self.set_param_hint('amplitude', value=self.parameter['amplitude'])
        self.set_param_hint('tau', min=1e-4, value=self.parameter['tau'])
        self.set_param_hint('t0', vary=False, value=self.parameter['t0'])


class ConvolvedExponentialRise(Model):

    parameter = {
        'amplitude': 1,
        'tau': 10,
        't0': 0,
        'sigma': 8
    }

    @staticmethod
    def function(t, amplitude, tau, t0, sigma):
        return amplitude * (np.exp(t / tau) * sp.erf(np.sqrt(2) * (t - 2*t0)/(2*sigma)) +
                            np.exp(t/tau) +
                            np.exp((2*t0 + sigma**2/(2*tau))/tau) *
                            sp.erf(np.sqrt(2)*(-tau*t + 2*tau*t0 + sigma**2)/(2*tau*sigma)) -
                            np.exp((2*t0 + sigma**2/(2*tau))/tau)
                            ) * np.exp(-t/tau)/2

    def __init__(self, nan_policy='raise', prefix='', **kws):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='ConvolvedExponentialRise',
                         **kws)

        self.set_param_hint('amplitude', value=self.parameter['amplitude'])
        self.set_param_hint('tau', min=1e-4, value=self.parameter['tau'])
        self.set_param_hint('t0', vary=False, value=self.parameter['t0'])
        self.set_param_hint('sigma', vary=False, value=self.parameter['sigma'])


class ConvolvedExponentialFall(Model):

    parameter = {
        'amplitude': 1,
        'tau': 10,
        't0': 0,
        'sigma': 8
    }

    @staticmethod
    def function(t, amplitude, tau, t0, sigma):
        return (amplitude * (1 - sp.erf( np.sqrt(2)*(-tau*t + 2*tau*t0 + sigma**2)/(2*tau*sigma)) ) *
                np.exp((-t + 2*t0 + sigma**2/(2*tau))/tau)/2)

    def __init__(self, nan_policy='raise', prefix='', **kwargs):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='ConvolvedExponentialFall',
                         **kwargs)

        self.set_param_hint('amplitude', value=self.parameter['amplitude'])
        self.set_param_hint('tau', min=1e-4, value=self.parameter['tau'])
        self.set_param_hint('t0', vary=False, value=self.parameter['t0'])
        self.set_param_hint('sigma', vary=False, value=self.parameter['sigma'])


class Offset(Model):

    parameter = {
        'amplitude': 1,
        't0': 0
    }

    @staticmethod
    def function(t, amplitude, t0):
        return amplitude * np.heaviside(t - t0, 1)

    def __init__(self, nan_policy='raise', prefix='', **kwargs):
        super().__init__(self.function,
                         independent_vars=None,
                         param_names=None,
                         nan_policy=nan_policy,
                         prefix=prefix,
                         name='Offset',
                         **kwargs)

        self.set_param_hint('amplitude', value=self.parameter['amplitude'])
        self.set_param_hint('t0', vary=False, value=self.parameter['t0'])


if __name__ == '__main__':
    pass
