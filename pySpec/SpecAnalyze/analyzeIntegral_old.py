from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy

#from ..SpecPlot.plotMinorSymLogLocator import MinorSymLogLocator
from ..SpecCore import *


class IntegralAnalysis:

    linthresh = 10

    def __init__(self, data: TransientSpectrum, path='./'):
        plt.close()

        self.path = os.path.join(path, 'Integral_Analysis')
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.data = deepcopy(data)
        self.data.sort()
        self.data.orient_data('x')

        self.y_neg_int, self.y_pos_int, self.y_rel, self.y_dif, self.y_cum = self.calculate_integral(data)

    def create_output(self):
        self.plot_abs()
        self.plot_rel()
        self.write_file()

    @staticmethod
    def calculate_integral(data: TransientSpectrum, threshhold=0) -> (float, float, float, float, float):
        y_pos = np.where(data.y.array > threshhold, data.y.array, 0)
        y_neg = np.where(data.y.array < - threshhold, data.y.array, 0)

        y_neg_int = - trapezoid(y_neg, data.x.array, axis=0)
        y_pos_int = trapezoid(y_pos,   data.x.array, axis=0)

        integral_rel = 100 * (y_pos_int / y_neg_int)

        integral_dif = y_pos_int - y_neg_int
        integral_cum = y_pos_int + y_neg_int

        return y_neg_int, y_pos_int, integral_rel, integral_dif, integral_cum

    def plot_abs(self) -> None:
        fig, ax = plt.subplots(1, figsize=(7, 3))
        t_0, _ = self.data.t.closest_to(0.5)

        normalization = self.y_pos_int + self.y_neg_int

        ax.plot(self.data.t, self.y_neg_int/normalization, color='tab:blue', marker='o', fillstyle='none', ls='', label='Bleach')
        ax.plot(self.data.t, self.y_pos_int/normalization, color='tab:red', marker='o', fillstyle='none', ls='', label='Absorption')

        ax.tick_params(which='both',
                       direction="in", top=True, right=True, left=True, labelbottom=True, labelleft=True,
                       labelright=False, labeltop=False, width=1.5, labelsize=10
                       )

        ax.set_ylabel('Area', weight='bold', size=12)

        ax.set_xlim(-1, np.max(self.data.t))

        ax.set_xscale('symlog')
        #ax.xaxis.set_minor_locator(MinorSymLogLocator(self.linthresh))

        plt.savefig(os.path.join(self.path, 'integralAnalysis_Abs.png'))
        plt.savefig(os.path.join(self.path, 'integralAnalysis_Abs.svg'))
        plt.close()

    def plot_rel(self):
        fig, axs = plt.subplots(3, sharex='all', figsize=(7, 8), gridspec_kw={'hspace': 0.1, 'wspace': 0.1, 'height_ratios': [2, 1, 1]})

        t_0, _ = self.data.t.closest_to(0.5)
        axs[0].set_ylim(-10, 130)

        axs[1].set_ylim(- 1.1 * np.max(abs(self.y_cum[t_0:])), 1.1 * np.max(abs(self.y_cum[t_0:])))
        axs[2].set_ylim(0, 1.1 * np.max(abs(self.y_dif[t_0:])))

        for i, (ax, y, label, color, ylabel) in enumerate(zip(
                axs,
                [self.y_rel, self.y_cum, self.y_dif],
                ['Absorption / Bleach', 'Absorption - Bleach', 'Absorption + Bleach'],
                ['tab:green', 'tab:blue', 'tab:red'],
                ['%A', '\u0394A', '\u03A3A']

        )
        ):
            ax.tick_params(which='both',
                           direction="in",
                           top=True, right=True, left=True,
                           labelbottom=False, labelleft=True, labelright=False, labeltop=False,
                           width=1.5, labelsize=10
                           )

            if i == 2:
                ax.tick_params(labelbottom=True)
                ax.tick_params(labelbottom=True)
                ax.set_xlabel('Time / ps', weight='bold', size=12)

            ax.plot(self.data.t, y, label=label, marker='o', ls='', fillstyle='none', color=color)

            if i == 1:
                ax.axhline(0, 0, 1, color='k', ls='--')

            ax.set_xscale('symlog', linthresh=self.linthresh)
            #ax.xaxis.set_minor_locator(MinorSymLogLocator(self.linthresh))
            ax.legend()
            ax.set_xlim(-0.2, 2000) #ax.set_xlim(-1, np.max(self.data.t))
            ax.set_ylabel(ylabel, weight='bold', size=12)

        plt.savefig(os.path.join(self.path, 'integralAnalysis_Rel.png'))
        plt.savefig(os.path.join(self.path, 'integralAnalysis_Rel.svg'))
        plt.close()

    def write_file(self) -> None:
        with open(os.path.join(self.path, 'integralAnalysis.txt'), 'w') as file:
            file.write('t/ps:')
            for t in self.data.t:
                file.write(f' {t:.3f}')
            file.write('\nAbsorptionArea:')
            for y in self.y_pos_int:
                file.write(f' {y:.3f}')
            file.write('\nBleachArea: ')
            for y in self.y_neg_int:
                file.write(f' {y:.3f}')

            file.write('\n%Area:')
            for y in self.y_rel:
                file.write(f' {y:.3f}')
            file.write('\nDeltaArea:')
            for y in self.y_cum:
                file.write(f' {y:.3f}')
            file.write('\nSumArea:')
            for y in self.y_dif:
                file.write(f' {y:.3f}')

    @staticmethod
    def read_file(path: str) -> list:
        with open(path, 'r') as file:
            lines = file.readlines()

        values = []
        for line in lines:
            line = [float(x) for x in line.split()[1:]]
            values.append(line)

        return values

    @staticmethod
    def monoexponential(t: float, a: float, tau: float) -> float:
        return a * np.exp(-t / tau)

    @staticmethod
    def biexponential(t: float, a1: float, a2: float, tau1: float, tau2: float) -> float:
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
