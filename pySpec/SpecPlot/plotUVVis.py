from .plotMPCFigure import MPCFigure
from .plotStaticMethods import lighten_color
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from ..SpecCore.SpecCoreSpectrum.coreCalculation import Calculation

import matplotlib.pyplot as plt
import numpy as np

from ..SpecCore.enums.enumUnit import EnergyUnit, DataUnit


def plotUVVis(spec: Spectrum,
              calc: Calculation or None = None,
              cfac=0.9,
              c1='k', c2='tab:blue',
              xlim=(15, 45), ylim=(1e2, 1e5)):

    def f(x):
        return 1e4 / x

    fig, ax = plt.subplots(1, 1, FigureClass=MPCFigure)

    ax.plot(f(spec.x), spec.y, color=c1, zorder=-5)
    ax.fill_between(f(spec.x), spec.y, 0, color=lighten_color(c1, 0.25), zorder=-6)

    ax.set_xlim(xlim)
    ax.set_yscale('log')
    ax.set_ylim(ylim)

    ax.set_ylabel(spec.y.label)
    ax.set_xlabel(EnergyUnit.WAVENUMBER.value)

    if calc is not None:
        calc_fac = cfac * np.max(spec.y) / np.max(calc.y)

        ax.fill_between(f(calc.x),   calc_fac * calc.y, color=lighten_color(c2, 0.25), zorder=-4)
        ax.plot(f(calc.x),   calc_fac * calc.y, color=c2, zorder=-3)
        ax.bar(f(calc.pos), calc_fac * calc.int, 0.1, color=lighten_color(c2, 0.75), zorder=-2)

        rax = ax.twinx()
        rax.set_yscale('log')
        rax.set_ylim([x / calc_fac for x in ax.get_ylim()])
        rax.spines['right'].set_color(lighten_color(c2, 1.3))
        rax.tick_params(which='both', color=lighten_color(c2, 1.3), labelcolor=c2)
        rax.set_ylabel(DataUnit.FOSC.value, color=c2)

    tax = ax.twiny()
    newlabel = [1000, 600, 400, 300, 250, 200]
    newpos = [f(x) for x in newlabel]
    minorlabel = np.arange(200, 1000, 25)
    minorpos = [f(x) for x in minorlabel]
    tax.set_xticks(newpos)
    tax.set_xticks(minorpos, minor=True)
    tax.set_xticklabels(newlabel)
    tax.set_xlim(ax.get_xlim())
    tax.set_xlabel(EnergyUnit.WAVELENGTH.value)
    tax.spines['right'].set_visible(False)

    return fig, ax
