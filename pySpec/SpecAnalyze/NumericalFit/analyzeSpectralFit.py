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
from pySpec.SpecCore.SpecCoreSpectrum import Spectrum

from pySpec.SpecAnalyze.NumericalFit.analyzeLineshapeFunction import Gaussian, Lorentzian


class SpectralFit(NumericalFit):

    _fntypes = {
        'gauss': Gaussian,
        'lorentzian': Lorentzian
    }

    def __init__(self, data: Spectrum):
        super().__init__(data)

        try:
            self._prefix = f"{self._data.time['value']:.1f}_{self._data.time['unit']}"
        except TypeError:
            self._prefix = ''

    def __getitem__(self, item):
        if self._result is None:
            self._fit()

        if self._components is None:
            self._components = []
            for key, y in self._result.eval_components(x=self._data.x).items():
                self._components.append(
                    Spectrum(self._data.x,
                    y,
                    self._data.x.unit, self._data.y.unit,
                    key)
                )

        return self._components[item]

    @property
    def fit(self):
        if self._result is None:
            self._fit()

        return Spectrum(self._data.x,
                        self._result.best_fit,
                        self._data.x.unit, self._data.y.unit,
                        self._data.time)

    def _fit(self):
        if self._composit_model is None:
            raise ValueError("No model present!")

        self._result = self._composit_model.fit(data=self._data.y.array,
                                                nu=self._data.x.array,
                                                params=self._composit_params,
                                                nan_policy='omit',
                                                method=self._method)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pySpec import TransientSpectrum

    d = TransientSpectrum.from_file(
        r"N:\DataProcessing\Stuttgart\BTC-6-NiN3\Output\BTC-6-NiN3_UVmIR_2000cm_340nm_complete.dat"
    )

    t = (
        SpectralFit(d.spectrum['1']).add_fn(Gaussian, s0=-10, nu0=2040, fwhm=10)
                                    .add_fn(Gaussian, s0= 10, nu0=2000, fwhm=10)
    )
    f = t.fit

    print(t.result.fit_report())

    for key, value in t.result.params.valuesdict().items():
        print(f"{key:12} = {value:10.4f}")

    plt.plot(d.x, d.spectrum['1'].y)
    plt.plot(f.x, f.y)
    plt.xscale('symlog')
    plt.show()
