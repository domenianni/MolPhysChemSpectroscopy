import matplotlib.pyplot as plt
import numpy as np
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from .plotStaticMethods import lighten_color
from .plotMPCFigure import MPCFigure


def plotFluorescence(uv: Spectrum, fl: Spectrum,
                     amp: float = 10000,
                     c1: str = 'k', c2: str = 'tab:blue',
                     fl_first: bool = False,
                     xlim: tuple[float, float] = (220, 700), ylim: tuple[float, float] = (50, 1e5)):

    fig, ax = plt.subplots(FigureClass=MPCFigure)

    ax.plot(uv.x, uv.y, color=c1, zorder=1)
    ax.fill_between(uv.x, 0, uv.y, color=lighten_color(c1, 0.25), zorder=0)

    ax.plot(fl.x, fl.y / np.max(fl.y) * amp,
            color=c2, zorder=-1 + fl_first * 3)
    ax.fill_between(fl.x, fl.y / np.max(fl.y) * amp, 0,
                    color=lighten_color(c2, 0.25), zorder=-2 + fl_first * 3)

    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(uv.x.label)
    ax.set_ylabel(uv.y.label, color=c1)
    ax.tick_params(which='both', axis='y', colors=c1)
    ax.spines['left'].set_color(c1)
    ax.spines['right'].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylim(50/amp, ax.get_ylim()[1]/amp)
    ax2.set_yscale('log')
    ax2.set_ylabel('Fluorescence Intensity / A.U.', color=c2)
    ax2.spines['right'].set_color(c2)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(which='both', axis='y', colors=c2)

    return fig, ax, ax2