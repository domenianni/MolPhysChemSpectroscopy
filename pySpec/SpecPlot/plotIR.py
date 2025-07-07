from .plotMPCFigure import MPCFigure
from .plotStaticMethods import lighten_color
from ..SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from ..SpecCore.SpecCoreSpectrum.coreCalculation import Calculation
from ..SpecCore.enums.enumUnit import DataUnit

import matplotlib.pyplot as plt
import numpy as np


def plotIR(
        spec: Spectrum,
        calc: Calculation,
        xlim: list[list[float]] = ([1100, 1700], [1950, 2250], [2750, 3100]),
        scale: list[list[float]] or None = (1, 1, 1),
        ylim: list[float] = (-2, 2),
        calc_ylim: float = -0.3,
        c1: str = 'k',
        c2: str = 'tab:blue',
        wspace: float = 0.1
):
    if scale is None:
        scale = [1 for _ in xlim]

    if not len(xlim) == len(scale):
        raise ValueError("Scaling factor amount needs to be the same as xlim segments.")

    width_ratios = [l[1] - l[0] for l in xlim]

    fig, axs = plt.subplots(1, len(xlim),
                            gridspec_kw={'width_ratios': width_ratios, 'wspace': wspace},
                            FigureClass=MPCFigure)
    y_fac = 1
    if spec.y.unit == DataUnit.EPSILON:
        y_fac = 0.001

    axs2 = []
    for ax in axs:
        axs2.append(ax.twinx())

    for i, (ax, ax2) in enumerate(zip(axs, axs2)):
        ax.plot(spec.x, scale[i] * y_fac * spec.y, color=c1, zorder=-4)
        ax.fill_between(spec.x, 0, scale[i] * y_fac * spec.y, color=lighten_color(c1, 0.25), zorder=-5)

        ax2.bar(calc.pos, - scale[i] * calc.int, 5, color=c2, zorder=-2)
        ax2.plot(calc.x, - scale[i] * calc.y, color=c2, zorder=-1)
        ax2.fill_between(calc.x, 0, - scale[i] * calc.y, zorder=-3, color=lighten_color(c2, 0.1))

        ax.set_xlim(xlim[i])

        ax.set_ylim(ylim)
        ax2.set_ylim([x * calc_ylim / ylim[0] for x in ylim])

    for ax in (axs, axs2):
        fig.format_axis_break(ax)

    fig.add_axis_break_lines(ax)

    axs2[-1].spines['right'].set_color(c2)
    axs2[-1].tick_params(which='both', color=c2, labelcolor=c2)

    axs[0].set_ylabel(spec.y.label)
    axs2[-1].set_ylabel(calc.y.label, color=c2)

    if spec.y.unit == DataUnit.EPSILON:
        axs[0].set_ylabel(r'$\epsilon$ / 10$^{3}$ M$^{-1}$cm$^{-1}$')
    else:
        axs[0].set_ylabel(axs, spec.y.label)

    fig.place_xlabel(axs, spec.x.label, wspace=wspace)

    return fig, axs, axs2
