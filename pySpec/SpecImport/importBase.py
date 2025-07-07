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

from ..SpecCore.SpecCoreSpectrum.coreTransientSpectrum import TransientSpectrum
from ..SpecTransform.transformBaseline import Baseline
from ..SpecTransform.transformAlignment import CrossCorrelationAlignment as cca
from ..SpecAnalyze.analyzeStatistics import StatisticalAnalysisMeasurements
from ..SpecPlot.plotMPCFigure import MPCFigure

import concurrent.futures as ftr
from multiprocessing import cpu_count
import numpy as np


class ImportTimeResolvedBase:
    """Base class for import and pre-processing of the different time-resolved experiments. Acts on a list of
    :class:`TransientSpectrum`, aka data-sets by modifying them individually/ with respect to each other and finally
    averaged via the `average`-property. Subclasses can implement class methods to implement a file-parser and to create
    the necessary `TransientSpectrum`-Objects. This class also supports indexing to access the individual
    data sets."""

    def __init__(self, data_list: list[TransientSpectrum], file_list=None):
        """
        :param data_list: The list of data sets to process.
        :param file_list: A list of file-names associated with the data sets. Utilized for time-axis assignment.
        """
        self._data_list: list = data_list
        self._file_list: list = file_list
        self._ignore: set = set()
        self._pre_scans: list or None = None
        self._average = None

        self._cc = []

    @property
    def pre_scans(self):
        """
        :returns: The pre-scans that were subtracted from the data sets.
        """
        return self._pre_scans

    def subtract_prescans(self, until_time, from_time=None):
        """
        Subtracts pre-scans from the data sets, storing the subtracted values for further analysis.

        :param until_time: Time until which the pre-scan is subtracted.
        :param from_time: Optional time from which to start subtraction (if different from the start).
        """
        self._pre_scans = []

        for data in self._data_list:
            data.subtract_prescans(until_time, from_time)
            self._pre_scans.append(data.pre_scan)

    @property
    def data(self):
        """
        :returns: The list of data sets being processed.
        """
        return self._data_list

    def __getitem__(self, item):
        """
        Access a specific data set by index.

        :param item: The index of the data set to access.
        :returns: A specific TransientSpectrum object from the data list.
        """
        return self._data_list[item]

    def convert_all_to(self, x_type='wn'):
        """
        Converts all data sets' x-axis units to the specified type ('wn', 'wl', or 'ev').

        :param x_type: The desired x-axis unit ('wn', 'wl', or 'ev').
        :returns: The current instance of ImportTimeResolvedBase after conversion.
        """
        for data in self._data_list:
            data.x = data.x.convert_to(x_type)

        return self

    def sort_all(self):
        """
        Sorts all data sets in both x and t directions to ensure correct ordering.
        :returns: The current instance of ImportTimeResolvedBase after sorting.
        """
        for data in self._data_list:
            data.sort()

        return self

    def interpolate_all(self, reference: int = 0):
        """
        Interpolates all data sets to match the reference data set, aligning in both x and t directions.

        :param reference: The index of the reference data set to interpolate others to.
        :returns: The current instance of ImportTimeResolvedBase after interpolation.
        """
        for i, data in enumerate(self._data_list):
            if i == reference:
                continue

            data.interpolate_to(self._data_list[reference].x, self._data_list[reference].t)

        return self

    @property
    def average(self):
        """
        TODO: Repeated evaluation causes the average to change! Change to single time lazy evaluation?

        :returns: The averaged data from all data sets, after applying sorting and interpolation.
        """
        self.sort_all()
        self.interpolate_all()

        return TransientSpectrum.average_from([data for idx, data in enumerate(self._data_list) if idx not in self._ignore])

    def apply_baseline(self, region: list[float], t_region: list[float] or None = None, baseline_type: str = 'static', dimension: str = 'wn'):
        """
        Applies a baseline correction to each data set in the specified region. Can handle different baseline types.

        :param region: The region of the x-axis to apply the baseline correction.
        :param t_region: The time region for baseline correction, if any.
        :param baseline_type: The type of baseline correction ('static', 'one-point', or 'linear').
        :param dimension: The x-axis dimension (default is 'wn').
        :returns: A list of baseline objects created during the operation.
        """
        self._orient_all_data('t')
        self.convert_all_to(dimension)

        baselines = []

        for data in self._data_list:
            if baseline_type == 'static':
                bsl = Baseline(data).static_baseline(region, t_region=t_region)
                data.y -= bsl.y
            elif baseline_type == 'one-point':
                bsl = Baseline(data).one_point_baseline(x_region=region, t_region=t_region)
                data.y -= bsl
            elif baseline_type == 'linear': # WHY DOES THIS NOT WORK?????? TODO
                bsl = Baseline(data).linear_baseline(regions=region)
                data.y -= bsl.y
            else:
                raise ValueError("baseline_type has to be either of 'static', 'one-point' or 'linear'!")

            baselines.append(bsl)

        return baselines

    def cross_correlate(self, reference_idx=0, par=True, **kwargs):
        """
        Cross-correlates all data sets to the reference data set using the specified reference index.

        :param reference_idx: The index of the reference data set.
        :param par: Boolean flag to enable parallel processing.
        :returns: The current instance of ImportTimeResolvedBase after cross-correlation.
        """
        assert reference_idx <= len(self._data_list)

        unit = self._data_list[0].x.unit
        self.convert_all_to('wl')

        if par:
            workers = cpu_count() - 2
            if workers < 1:
                workers = 1

            with ftr.ProcessPoolExecutor(max_workers=workers) as executor:
                cca_ftr = []
                for idx, data in enumerate(self._data_list):
                    if idx == reference_idx:
                        continue
                    cca_ftr.append(executor.submit(cca, data, self._data_list[reference_idx], **kwargs))

            for idx in range(len(self._data_list)):
                if idx == reference_idx:
                    continue

                self._cc.append(cca_ftr.pop(0).result())

                data = self._data_list[idx]
                data.t = data.t + self._cc[-1].shift_vector['t']
                data.x = data.x + self._cc[-1].shift_vector['x']
        else:
            for idx in range(len(self._data_list)):
                if idx == reference_idx:
                    continue

                data = self._data_list[idx]

                self._cc.append(cca(data, self._data_list[reference_idx], **kwargs))

                data.t = data.t + self._cc[-1].shift_vector['t']
                data.x = data.x + self._cc[-1].shift_vector['x']

        self.convert_all_to(unit)

        return self

    @property
    def stat_analysis(self):
        """
        :returns: An instance of StatisticalAnalysisMeasurements for analyzing the statistical properties of the data.
        """
        return StatisticalAnalysisMeasurements(self._data_list)

    def _orient_all_data(self, direction):
        """
        Orients all data sets along the specified direction ('x' or 't').

        :param direction: The direction to orient the data ('x' or 't').
        :returns: The current instance of ImportTimeResolvedBase after orientation.
        """
        for data in self._data_list:
            data.orient_data(direction)

        return self

    def delete_run(self, idx):
        """
        Deletes one or more data sets by their index.

        :param idx: The index or list of indices of the data sets to delete.
        :returns: The current instance of ImportTimeResolvedBase after deletion.
        """
        if isinstance(idx, list):
            for i in reversed(sorted(idx)):
                self._data_list.pop(i)
        else:
            self._data_list.pop(idx)

        return self

    def ignore_run(self, idx: int):
        """
        Marks a data set for exclusion from averaging or further processing.

        :param idx: The index or list of indices to ignore.
        :returns: The current instance of ImportTimeResolvedBase.
        """
        self._ignore.update(idx)

        return self

    def delete_positions(self, x_pos: list[float] = None, t_pos: list[float] = None):
        """
        Deletes specific positions from the data sets based on x or t coordinates.

        :param x_pos: List of x-axis positions to delete.
        :param t_pos: List of t-axis positions to delete.
        :returns: The current instance of ImportTimeResolvedBase after deletion.
        """
        for data in self._data_list:
            data.eliminate_positions(x_pos=x_pos, t_pos=t_pos)

        return self

    def delete_steps(self, x_idx: list[int] = None, t_idx: list[int] = None):
        """
        Deletes specific time steps or x steps from the data sets based on indices.

        :param x_idx: List of x-axis indices to delete.
        :param t_idx: List of t-axis indices to delete.
        :returns: The current instance of ImportTimeResolvedBase after deletion.
        """
        for data in self._data_list:
            data.eliminate_idx(x_idx=x_idx, t_idx=t_idx)

        return self

    def plot_single_runs(self, vmin=-1, vmax=1):
        """
        Plots each individual data set in a grid of subplots.

        :param vmin: Minimum value for color normalization.
        :param vmax: Maximum value for color normalization.
        :returns: The figure and axes of the plot.
        """

        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        col = int(np.ceil(np.sqrt(len(self._data_list))))
        row = int(np.ceil(len(self._data_list) / col))

        fig, axs = plt.subplots(col, row,
                                gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
                                sharex=True, sharey=True,
                                FigureClass=MPCFigure,
                                width='two_column')

        for i, single in enumerate(self._data_list):
            single.orient_data('t')
            ax = axs[i % col, i // col]

            ax.pcolormesh(single.x,
                          single.t,
                          single.y,
                          norm=TwoSlopeNorm(0, vmin, vmax)
                          )

            ax.text(0.9, 0.1, f"{i}", transform=ax.transAxes, va='center', ha='center')

        for ax in axs:
            for a in ax:
                a.tick_params(labelleft=False, labelbottom=False)

        axs[0, 0].set_yscale('symlog')

        for i in range(col):
            fig.format_axis_break(axs[i, :], 'x')

        for i in range(row):
            fig.format_axis_break(axs[:, i], 'y')

        return fig, axs


if __name__ == '__main__':
    pass
