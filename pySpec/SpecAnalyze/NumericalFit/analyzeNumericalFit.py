import numpy as np

from .analyzeExponentialFunction import (ExponentialFall, ExponentialRise,
                                         ConvolvedExponentialRise, ConvolvedExponentialFall)
from .analyzeLineshapeFunction import Gaussian, Lorentzian

from pySpec.SpecCore.SpecCoreSpectrum.coreTransient import Transient
from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum

from lmfit import Model, Parameter, CompositeModel
from copy import deepcopy


class NumericalFit:

    __method = 'leastsq'

    __fntypes = {
        'fall': ExponentialFall,
        'rise': ExponentialRise,
        'fall_cnv': ConvolvedExponentialFall,
        'rise_cnv': ConvolvedExponentialRise,
        'gauss': Gaussian,
        'lorentzian': Lorentzian
    }

    def __init__(self, data: Transient or Spectrum):
        self._fncount = 0

        self._models = {}
        self._result = None

        self._composit_model: CompositeModel = None
        self._composit_params: Parameter = None

        self._data = deepcopy(data)
        self._data.y.array = np.nan_to_num(self._data.y.array)

    def __getitem__(self, item):
        return self._models[item]

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, method):
        self.__method = method

    @property
    def result(self):
        if self._result is None:
            self._fit()

        return self._result

    @property
    def fit(self):
        if self._result is None:
            self._fit()

        return Transient(self._data.t, self._result.best_fit, self._data.t.unit, self._data.y.unit, self._data.position)

    def add_fn(self, fntype: str or Model = 'fall', **kwargs):
        if isinstance(fntype, str):
            model: Model = self.__fntypes.get(fntype)(prefix=f'f{self._fncount}_')
        elif isinstance(fntype, Model):
            model: Model = fntype
            model.prefix = f'f{self._fncount}_'
        elif issubclass(fntype, Model):
            model: Model = fntype(prefix=f'f{self._fncount}_')
        else:
            raise ValueError("Möp")

        params: Parameter = model.make_params()

        for key, val in kwargs.items():
            params[f'f{self._fncount}_{key}'].value = val

        self._models[self._fncount] = (model, params)
        self._create_composit_model()

        self._fncount += 1

        return self

    def _create_composit_model(self):
        self._composit_model = self._models[0][0]

        for i in range(0, self._fncount, 1):
            self._composit_model += self._models[i+1][0]

        self._composit_model.nan_policy = 'propagate'

        self._composit_params = self._composit_model.make_params()

        for key, vals in self._models.items():
            for pname, params in vals[1].items():
                self._composit_params[pname] = params

        self._result = None

    def _fit(self):
        if self._composit_model is None:
            raise ValueError("No model present!")

        self._result = self._composit_model.fit(data=self._data.y.array,
                                                t=self._data.t.array,
                                                params=self._composit_params,
                                                nan_policy='omit',
                                                method=self.__method)

    def save(self, path: str = "./"):
        if self._result is None:
            raise ValueError("No Result present!")

        with open(path + f"{self._data.position['value']:.1f}_{self._data.position['unit']}_fit_report.txt", 'w') as file:
            file.write(self._result.fit_report())

        self.fit.save(path + f"{self._data.position['value']:.1f}_{self._data.position['unit']}_fit_function.dat")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Spectroscopy import TransientSpectrum

    d = TransientSpectrum.from_file(r"")

    t = (NumericalFit(d.transient['']).add_fn(ExponentialFall, amplitude=-15, tau=10)
                                          .add_fn(ExponentialFall, amplitude=-10, tau=30)
                                          .add_fn(ExponentialRise, amplitude=-5, tau=100))
    f = t.result

    print(t.fit.fit_report())

    plt.plot(d.t, d.transient[''].y)
    plt.plot(f.t, f.y)
    plt.show()
