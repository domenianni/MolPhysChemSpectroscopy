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

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cm = 1/2.54


class MPCFigure(Figure):

    unicode_chars = {
        'epsilon': u"\u03B5",
        'Delta': u"\u0394",
        'micro': u"\u00B5",
        '^-': u"\u207B",
        '^1': u"\u00B9",
        '^2': u"\u00B2",
    }

    __presets = {
        'angewandte' : {
            'width' : {
               'one_column': 8.6 * cm,
               'two_column': 17.4 * cm
            } ,
            'aspect_ratio': 0.7,
            'margin': 0.15, # 0.3 / 2
            'hspace': 0.05,
            'wspace': 0.05
        }
    }

    def __init__(self, *args, height_factor=1, width='one_column', preset='angewandte', **kwargs):
        path = Path(__file__).parent.resolve()

        plt.style.use(path.joinpath('publication.mplstyle'))

        kwargs.update({
            'figsize': (
                self.__presets.get(preset).get('width').get(width),
                self.__presets.get(preset).get('width').get(width) *
                self.__presets.get(preset).get('aspect_ratio') * height_factor
            )
        })

        super().__init__(
            *args,
            **kwargs
        )

        self.subplots_adjust(left  = self.__presets.get(preset).get('margin'),
                             bottom= self.__presets.get(preset).get('margin'),
                             right = 1 - self.__presets.get(preset).get('margin'),
                             top   = 1 - self.__presets.get(preset).get('margin'),
                             hspace= self.__presets.get(preset).get('hspace'),
                             wspace= self.__presets.get(preset).get('wspace'))

    @classmethod
    def format_axis_break(cls, axes, which='x'):
        if which == 'x':
            cls._format_x_axis_break(axes)
        elif which == 'y':
            cls._format_y_axis_break(axes)

    @staticmethod
    def _format_x_axis_break(axes):
        if len(axes) < 2:
            raise ValueError("There need to be at least two axes!")

        axes[0].tick_params(which='both', right=False, labelright=False)
        axes[0].spines['right'].set_visible(False)

        if len(axes) > 2:
            for ax in axes[1:-1]:
                ax.tick_params(which='both', left=False, right=False, labelleft=False, labelright=False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)

        axes[-1].tick_params(which='both', left=False, labelleft=False)
        axes[-1].spines['left'].set_visible(False)

    @staticmethod
    def _format_y_axis_break(axes):
        if len(axes) < 2:
            raise ValueError("There need to be at least two axes!")

        axes[0].tick_params(which='both', bottom=False, labelbottom=False)
        axes[0].spines['bottom'].set_visible(False)

        if len(axes) > 2:
            for ax in axes[1:-1]:
                ax.tick_params(which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)

        axes[-1].tick_params(which='both', top=False, labeltop=False)
        axes[-1].spines['top'].set_visible(False)

    @classmethod
    def add_axis_break_lines(cls, axes, length: float = 0.04, which='x'):
        if which == 'x':
            cls._add_x_axis_break_lines(axes, length)
        elif which == 'y':
            raise NotImplementedError()

    @staticmethod
    def _add_x_axis_break_lines(axes, length):
        for ax in axes:
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1, ls=':')
            for i in (-0.01, 1.01):
                if (i == -0.01) & (ax == axes[0]):
                    continue

                if (i == 1.01) & (ax == axes[-1]):
                    continue

                bk, = ax.plot((i, i), (-length, length), **kwargs)
                bk.set_dashes([1, 1])
                bk, = ax.plot((i, i), (1 - length, 1 + length), **kwargs)
                bk.set_dashes([1, 1])

    @staticmethod
    def place_xlabel(axes, label, wspace: float = 0.1):
        width_ratios = [ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes]

        axes[0].set_xlabel(label, ha='center')
        # calculates the complete axis length, without spacings
        axis_len = sum(width_ratios)
        # spacings are prepared in fractions of the AVERAGE axis length, therefore add this
        axis_len += np.average(width_ratios) * (len(axes) - 1) * wspace
        # now transfer to axis coordinates of subaxis no. 0 and place at 0.5
        axes[0].xaxis.set_label_coords(0.5 / (width_ratios[0] / axis_len), -0.1)


if __name__ == '__main__':
    pass
