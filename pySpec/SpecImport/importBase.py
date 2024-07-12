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

import concurrent.futures as ftr
from multiprocessing import cpu_count


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

    @property
    def pre_scans(self):
        return self._pre_scans

    def subtract_prescans(self, until_time, from_time=None):
        self._orient_all_data('x')
        self._pre_scans = []

        for data in self._data_list:
            data.subtract_prescans(until_time, from_time)
            self._pre_scans.append(data.pre_scan)

    @property
    def data(self):
        return self._data_list

    def __getitem__(self, item):
        return self._data_list[item]

    def convert_all_to(self, x_type='wn'):
        """
        :param x_type:

        Converts all `TransientSpectrum`-Object x-axis types. These can be 'wn', 'wl' or 'ev'.
        """
        for data in self._data_list:
            data.x = data.x.convert_to(x_type)

        return self

    def sort_all(self):
        """
        Sorts all data sets in both x- and t-directions.
        """
        for data in self._data_list:
            data.sort()

        return self

    def interpolate_all(self, reference: int = 0):
        """
        Interpolates all data sets in both x- and t-directions.
        """
        for i, data in enumerate(self._data_list):
            if i == reference:
                continue

            data.interpolate_to(self._data_list[reference].x, self._data_list[reference].t)

        return self

    @property
    def average(self):
        """
        :returns: The averaged data, as they were processed before this property-call.
        """
        self.sort_all()
        self.interpolate_all()

        return TransientSpectrum.average_from([data for idx, data in enumerate(self._data_list) if idx not in self._ignore])

    def apply_baseline(self, region: list[float], t_region: list[float] or None = None, baseline_type: str = 'static', dimension: str = 'wn'):
        """
        Applies a static baseline correction in the specified region for each data set individually.

        :param region: The region over which the baseline correction has to be applied.
        :param baseline_type: TODO NOT YET FUNCTIONAL! ONLY 'static' and 'one-point' WORKS
        :param dimension: The x-axis dimension in which the region is defined. Defaults to 'wn'.
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
        Cross-correlates all data sets onto the reference selected with reference_idx.
        """
        assert reference_idx <= len(self._data_list)

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
                    # cca_ftr.append(executor.submit(cca.correlate_from, self._data_list[reference_idx], data, parameter))

            for idx in range(len(self._data_list)):
                if idx == reference_idx:
                    continue

                cc = cca_ftr.pop(0).result()

                data = self._data_list[idx]
                data.t = data.t + cc.shift_vector['t']
                data.x = data.x + cc.shift_vector['x']

        else:
            for idx in range(len(self._data_list)):
                if idx == reference_idx:
                    continue

                data = self._data_list[idx]

                cc = cca(data, self._data_list[reference_idx], **kwargs)
                data.t = data.t + cc.shift_vector['t']
                data.x = data.x + cc.shift_vector['x']
                # self._data_list[idx] = cca.correlate_from(self._data_list[reference_idx], data, parameter)[0]

        return self

    @property
    def stat_analysis(self):
        """
        :returns: An instance of StatisticalAnalysisMeasurements

        This can be used to investigate the variance, standard deviation and other statistical markers of the data sets
        with respect to each other to check for changes and irregularities.
        """
        return StatisticalAnalysisMeasurements(self._data_list)

    def _orient_all_data(self, direction):
        """
        Orients all data sets in either 'x' or 't' directions.
        """
        for data in self._data_list:
            data.orient_data(direction)

        return self

    def delete_run(self, idx):
        if isinstance(idx, list):
            for i in reversed(sorted(idx)):
                self._data_list.pop(i)
        else:
            self._data_list.pop(idx)

        return self

    def ignore_run(self, idx):
        self._ignore.update(idx)

    def delete_positions(self, x_pos: list[float] = None, t_pos: list[float] = None):
        """
        :param x_pos:
        :param t_pos:

        Deletes x-axis / time steps.
        """
        for data in self._data_list:
            data.eliminate_positions(x_pos=x_pos, t_pos=t_pos)

        return self

    def delete_steps(self, x_idx: list[int] = None, t_idx: list[int] = None):
        """
        :param x_idx:
        :param t_idx:

        Deletes x-axis / time steps.
        """
        for data in self._data_list:
            data.eliminate_idx(x_idx=x_idx, t_idx=t_idx)

        return self


if __name__ == '__main__':
    pass
