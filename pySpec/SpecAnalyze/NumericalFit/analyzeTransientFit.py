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

from pySpec.SpecAnalyze.NumericalFit.analyzeNumericalFit import NumericalFit
from pySpec.SpecCore.SpecCoreSpectrum import Transient

from pySpec.SpecAnalyze.NumericalFit.analyzeExponentialFunction import (ExponentialFall, ExponentialRise,
                                         ConvolvedExponentialRise, ConvolvedExponentialFall)


class TransientFit(NumericalFit):

    _fntypes = {
        'fall': ExponentialFall,
        'rise': ExponentialRise,
        'fall_cnv': ConvolvedExponentialFall,
        'rise_cnv': ConvolvedExponentialRise
    }

    def __init__(self, data: Transient):
        super().__init__(data)

        self._prefix = f"{self._data.position['value']:.1f}_{self._data.position['unit']}"

    def __getitem__(self, item):
        if self._result is None:
            self._fit()

        if self._components is None:
            self._components = []
            for key, y in self._result.eval_components(t=self._data.t).items():
                self._components.append(
                    Transient(self._data.t,
                             y,
                             self._data.t.unit, self._data.y.unit,
                             key)
                )

        return self._components[item]

    @property
    def fit(self):
        if self._result is None:
            self._fit()

        return Transient(self._data.t, self._result.best_fit, self._data.t.unit, self._data.y.unit, self._data.position)

    def _fit(self):
        if self._composit_model is None:
            raise ValueError("No model present!")

        self._result = self._composit_model.fit(data=self._data.y.array,
                                                t=self._data.t.array,
                                                params=self._composit_params,
                                                nan_policy='omit',
                                                method=self._method)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pySpec import TransientSpectrum

    d = TransientSpectrum.from_file(
        r"N:\DataProcessing\Stuttgart\BTC-6-NiN3\Output\BTC-6-NiN3_UVmIR_2000cm_340nm_complete.dat"
    )

    t = (TransientFit(d.transient['2052']).add_fn(ExponentialFall, amplitude=-15, tau=10)
                                          .add_fn(ExponentialFall, amplitude=-10, tau=30)
                                          .add_fn(ExponentialRise, amplitude=-5, tau=1000))
    f = t.fit

    print(t.result.fit_report())

    for key, value in t.result.params.valuesdict().items():
        print(f"{key:12} = {value:10.4f}")

    plt.plot(d.t, d.transient['2052'].y)
    plt.plot(f.t, f.y)
    plt.xscale('symlog')
    plt.show()
