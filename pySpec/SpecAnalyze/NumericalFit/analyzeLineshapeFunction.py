from pySpec.SpecCore.coreLineShapes import gaussian, skewed_gaussian, lorentzian, voigt, pseudo_voigt

from lmfit import Model


class Gaussian(Model):

    parameter = {
        's0': 1,
        'nu0': 2000,
        'fwhm': 10
    }

    function = gaussian

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

    function = lorentzian

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
