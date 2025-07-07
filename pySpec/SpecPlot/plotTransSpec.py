import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from pySpec.SpecPlot.plotStaticMethods import lighten_color, factor_string
from pySpec.SpecCore.SpecCoreSpectrum.coreSpectrum import Spectrum
from pySpec.SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from pySpec.SpecAnalyze.analyzeProductSpectrum import ProductSpectrum

from pySpec.SpecPlot.plotMPCFigure import MPCFigure


mapping = {
    r'Ts'   :  12,
    r'Gs'   :   9,
    r'Ms'   :   6,
    r'ks'   :   3,
    r's'    :   0,
    r'ms'   :  -3,
    r'us'   :  -6,
    r'ns'   :  -9,
    r'ps'   : -12,
    r'fs'   : -15
}

cm = 1/2.54


def plotTransSpec(transspec: TransientSpectrum,
                  parent: Spectrum or None = None,
                  times: list[float] = (1, 5, 10),
                  average_width: int = 1,
                  offset_size: float = 2.5,
                  start_idx: int = 3,
                  factor: float = 4,
                  parent_scale: float or None = None):

    parent = deepcopy(parent)
    transspec = deepcopy(transspec)

    scale = {'start_idx': start_idx, 'factor': factor}

    fig = plt.figure(FigureClass=MPCFigure, height_factor=2)
    grid = plt.GridSpec(1, 4, figure=fig, hspace=0.05, wspace=0.05)

    ts = fig.add_subplot(grid[0, :3])
    textbox = fig.add_subplot(grid[0, 3], sharey=ts)
    textbox.spines[:].set_visible(False)
    textbox.set_xticks([])
    textbox.tick_params(which='both', left=False, right=False, labelleft=False)

    scale_factor = 1
    flag = False
    for i, t in enumerate(times):
        if i >= scale.get('start_idx'):
            scale_factor = scale.get('factor')

            if not flag:
                ts.text(0.1 * (ts.get_xlim()[1] - ts.get_xlim()[0]) + ts.get_xlim()[0],
                        offset_size * i,
                        fr'$\times$ {scale_factor}', size=12)
                flag = True

            i += 1

        spec = transspec.spectrum[str(t), average_width]
        offset = offset_size * i
        y = scale_factor * spec.y.array + offset

        ts.plot(spec.x, y, color='k')
        ts.fill_between(spec.x, offset, y, where=y>=offset, interpolate=True, color=lighten_color('tab:red', 0.75))
        ts.fill_between(spec.x, offset, y, where=y<offset, interpolate=True, color=lighten_color('tab:blue', 0.75))
        textbox.text(0,
                     offset,
                     f"${factor_string(spec.time.get('value'), 1, mapping.get(spec.time.get('unit')), 's')}$",
                     va='center'
                     )

    textbox.text(0.1, offset_size * (len(times)+flag), 'Delay')

    ts.set_xlabel(f"{transspec.x.label}")
    ts.set_ylabel(f"{transspec.y.label}")

    if parent is not None:
        parent.truncate_like(transspec.x)

        if parent_scale is None:
            parent.y *= -np.min(transspec.spectrum[str(times[0])].y) / np.max(parent.y)
        else:
            parent.y *= parent_scale

        ts.plot(parent.x, -parent.y - offset_size, color='k', zorder=-1)
        ts.fill_between(parent.x, -parent.y - offset_size, -offset_size, color=lighten_color('grey', 0.5), zorder=-2)
        textbox.text(0.1, -offset_size, 'Parent', va='center')

    ts.set_xlim(np.min(transspec.x), np.max(transspec.x))
    ts.set_ylim(ts.get_ylim()[0], offset_size * (len(times)+flag+0.5))

    return fig, ts, textbox


def plotGlobalFit(transspec: TransientSpectrum,
                  fitspec: TransientSpectrum,
                  times: list[float] = (1, 5, 10),
                  average_width: int = 1,
                  offset_size: float = 2.5,
                  start_idx: int = 3,
                  factor: float = 4):

    fitspec = deepcopy(fitspec)
    transspec = deepcopy(transspec)

    scale = {'start_idx': start_idx, 'factor': factor}

    fig = plt.figure(FigureClass=MPCFigure, height_factor=2)
    grid = plt.GridSpec(1, 4, figure=fig, hspace=0.05, wspace=0.05)

    ts = fig.add_subplot(grid[0, :3])
    textbox = fig.add_subplot(grid[0, 3], sharey=ts)
    textbox.spines[:].set_visible(False)
    textbox.set_xticks([])
    textbox.tick_params(which='both', left=False, right=False, labelleft=False)

    scale_factor = 1
    flag = False
    for i, t in enumerate(times):
        if i >= scale.get('start_idx'):
            scale_factor = scale.get('factor')

            if not flag:
                ts.text(0.1 * (ts.get_xlim()[1] - ts.get_xlim()[0]) + ts.get_xlim()[0],
                        offset_size * i,
                        fr'$\times$ {scale_factor}', size=12)
                flag = True

            i += 1

        spec = transspec.spectrum[str(t), average_width]
        fit = fitspec.spectrum[str(t)]
        offset = offset_size * i
        y = scale_factor * spec.y.array + offset
        yfit = scale_factor * fit.y.array + offset

        ts.plot(spec.x, y, color='tab:red', marker='o', fillstyle='none', ls='')
        ts.plot(fit.x, yfit, color='k')
        textbox.text(0,
                     offset,
                     f"${factor_string(spec.time.get('value'), 1, mapping.get(spec.time.get('unit')), 's')}$",
                     va='center'
                     )

    textbox.text(0.1, offset_size * (len(times)+flag), 'Delay')

    ts.set_xlabel(f"{transspec.x.label}")
    ts.set_ylabel(f"{transspec.y.label}")

    ts.set_xlim(np.min(transspec.x), np.max(transspec.x))
    ts.set_ylim(ts.get_ylim()[0], offset_size * (len(times)+flag+0.5))

    return fig, ts, textbox


def plotProdSpec(product: ProductSpectrum,
                 times: list[float] = (1, 5, 10),
                 average_width: int = 1,
                 offset_size: float = 2.5,
                 start_idx: int = 3,
                 factor: float = 4,
                 parent_scale: float = 1):

    product = deepcopy(product)

    scale = {'start_idx': start_idx, 'factor': factor}

    fig = plt.figure(FigureClass=MPCFigure, height_factor=2)
    grid = plt.GridSpec(1, 4, figure=fig, hspace=0.05, wspace=0.05)

    ts = fig.add_subplot(grid[0, :3])
    textbox = fig.add_subplot(grid[0, 3], sharey=ts)
    textbox.spines[:].set_visible(False)
    textbox.set_xticks([])
    textbox.tick_params(which='both', left=False, right=False, labelleft=False)

    scale_factor = 1
    flag = False
    for i, t in enumerate(times):
        if i >= scale.get('start_idx'):
            scale_factor = scale.get('factor')

            if not flag:
                ts.text(0.1 * (ts.get_xlim()[1] - ts.get_xlim()[0]) + ts.get_xlim()[0],
                        offset_size * i,
                        fr'$\times$ {scale_factor}', size=12)
                flag = True

            i += 1

        spec = product.spectrum[str(t), average_width]
        offset = offset_size * i
        y = scale_factor * spec.y.array + offset

        ts.plot(spec.x, y, color='k')
        ts.fill_between(spec.x, offset, y, interpolate=True, where=y >= offset, color=lighten_color('tab:green', 0.75))
        textbox.text(0.1,
                     offset,
                     f"${factor_string(spec.time.get('value'), 1, mapping.get(spec.time.get('unit')), 's')}$",
                     va='center'
                     )

    textbox.text(0.1, offset_size * (len(times)+1+flag), 'Delay')

    ts.set_xlabel(f"{product.x.label}")
    ts.set_ylabel(f"{product.y.label}")

    parent = product.static_data
    ts.plot(parent.x, parent_scale * parent.y + offset_size * (i+1), color='k', zorder=-1)
    ts.fill_between(parent.x, parent.y + parent_scale * offset_size * (i+1), offset_size * (i+1), color=lighten_color('grey', 0.5), zorder=-2)
    textbox.text(0.1, offset_size * (i+1), 'Parent', va='center')

    ts.set_xlim(np.min(product.x), np.max(product.x))
    ts.set_ylim(0)

    return fig, ts, textbox


if __name__ == '__main__':
    times = [1, 5, 10, 100, 200, 500, 1000, 5000, 1e4, 5e4, 1e5, 6e5, 1e7, 3e8]

    pd = TransientSpectrum.from_file("", t_unit='ps', data_unit='mdod')
    par = Spectrum.from_file("")

    fig, ax, _ = plotTransSpec(pd, par, times, 1, 2.5, 5, 4)

    fig.savefig(r"")
    fig.savefig(r"")
