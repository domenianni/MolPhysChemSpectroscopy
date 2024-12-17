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
from abc import ABC, abstractmethod

from pySpec.SpecCore.SpecCoreSpectrum.coreTransient import Transient
from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum

from lmfit import Model, Parameter, CompositeModel
from copy import deepcopy


class NumericalFit(ABC):

    _method = 'nelder'

    _fntypes = {}

    def __init__(self, data: Transient or Spectrum):
        self._fncount = 0

        self._models = {}
        self._result = None
        self._components = None

        self._composit_model: CompositeModel = None
        self._composit_params: Parameter = None

        self._data = deepcopy(data)
        self._data.y.array = np.nan_to_num(self._data.y.array)

        self._prefix = ""

    def __getitem__(self, item):
        return self._models[item]

    @property
    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def _fit(self):
        ...

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        self._method = method

    @property
    def result(self):
        if self._result is None:
            self._fit()

        return self._result

    def add_fn(self, fntype: str or Model = 'fall', vary=None, **kwargs):
        if isinstance(fntype, str):
            model: Model = self._fntypes.get(fntype)(prefix=f'f{self._fncount}_')
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

        if vary is not None:
            for key, val in vary.items():
                params[f'f{self._fncount}_{key}'].vary = val

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

    def save(self, path: str = "./", print_params: bool = True, print_report: bool = True):
        if self._result is None:
            raise ValueError("No Result present!")

        if print_report:
            with open(path + f"{self._prefix}_fit_report.txt", 'w') as file:
                file.write(self._result.fit_report())

        if print_params:
            with open(path + f"{self._prefix}_params.txt", 'w') as file:
                for key, value in self._result.params.valuesdict():
                    file.write(f"{key:12} = {value:10.4f}\n")

        self.fit.save(path + f"{self._prefix}_fit_function.dat")

        return self


if __name__ == '__main__':
    pass
