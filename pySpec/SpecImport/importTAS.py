import glob

import numpy as np
import scipy.ndimage as sn
from copy import deepcopy
import scipy.optimize as opt

import json

import matplotlib.pyplot as plt

from ..SpecCore.coreParser import Parser
from .importBase import ImportTimeResolvedBase
from ..SpecCore.SpecCoreSpectrum import TransientSpectrum
from ..SpecCore.SpecCoreSpectrum import Transient


class ImportTas(ImportTimeResolvedBase):

    def __init__(self, data_list):
        super().__init__(data_list)

        self._pre_scans = None

    @staticmethod
    def _normalize_transients(y: np.ndarray):
        y = y.copy()
        y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
        max_val = np.nanmax(y, axis=0)
        min_val = np.nanmin(y, axis=0)

        return (y - min_val) / (max_val - min_val)

    @staticmethod
    def _standard_deviation_mask(y: np.ndarray):
        kernel_x = 2
        kernel_y = 2
        std_dev_threshold = 2
        threshold = 0.3

        std = np.ones_like(y)

        for i in range(kernel_x, y.shape[0] - kernel_x):
            for j in range(kernel_y, y.shape[1] - kernel_y):
                std[i, j] = sn.standard_deviation(y[i-kernel_x:i+kernel_x, j-kernel_y:j+kernel_y])

        noise_cutoff = std_dev_threshold * np.nanmedian(std)
        noise_mask = np.where(np.nan_to_num(std, nan=1e308, posinf=1e308, neginf=1e308) > noise_cutoff, 1, 0)

        noise_score = np.mean(noise_mask, axis=0)
        noise_mask = np.where(noise_score > threshold, False, True)

        return noise_mask

    @staticmethod
    def _edge_detection(y):
        return sn.sobel(y, axis=0)

    @staticmethod
    def _sellmeier(x: np.ndarray, m: float = 1, b: float = 1) -> np.ndarray:
        # Source: https://refractiveindex.info/ 15.02.2024, 10:47
        return (
            m * np.sqrt(1 +
                (0.6961663 * (x**2)) / ((x**2) - 0.0684043 ** 2) +
                (0.4079426 * (x**2)) / ((x**2) - 0.1162414 ** 2) +
                (0.8974794 * (x**2)) / ((x**2) - 9.8961610 ** 2)
            ) + b
        )

    def _target_fn(self, param, x: np.ndarray, t_offset: np.ndarray):
        return np.sum(np.power(t_offset - self._sellmeier(x, param[0], param[1]), 2))

    def correct_dispersion(self,
                           data: TransientSpectrum,
                           x_range=(300, 500),
                           t_range = (-2, 2),
                           scaling_factor=1,
                           method='bfgs',
                           visualize=False):

        #  Make sure that x is in units of wavelengths
        data.x = data.x.convert_to('wl')

        # Copy data to work on.
        scratch_data = deepcopy(data)
        # Truncate to remove unwanted noisy regions
        scratch_data.truncate_to(t_range=t_range, x_range=x_range)

        # Make sure, that the time dimension is the first dimension in the data array
        scratch_data.orient_data('t')
        y = scratch_data.y.array

        # Calculate mask to remove too noisy transients TODO: ALSO REMOVE TIME SLICES!
        x_mask = self._standard_deviation_mask(y)
        normalized_data = self._normalize_transients(np.abs(y[:, x_mask]))

        # Detect horizontal edge via sobel filtering
        edge_data = self._edge_detection(normalized_data)
        # Find maximum (TODO: MAKE MORE SOPHISTICATED!)
        t_index = np.argmax(edge_data, axis=0)
        t_offset = [scratch_data.t[idx] for idx in t_index]

        # Fit Sellmeier Equation
        res = opt.minimize(self._target_fn, (-4.397e+03, 7.619e+03),
                           args=(scratch_data.x.array[x_mask], t_offset),
                           method=method,
                           tol=1e-12)
        print(res)

        # Calculate temporal offset per wavelength
        t_correction = self._sellmeier(data.x.array, res.x[0], res.x[1])

        data.orient_data('t')

        if visualize:
            plt.pcolormesh(scratch_data.x[x_mask], scratch_data.t, edge_data)
            plt.plot(data.x, t_correction)
            plt.show()

        new_y = np.zeros_like(data.y.array.T)
        for idx, t_corr in enumerate(t_correction):
            y_tr: Transient = data.transient[idx]
            y_tr.t -= scaling_factor * t_corr
            y_tr.interpolate_to(data.t.array)

            new_y[idx] = y_tr.y.array

        return TransientSpectrum(data.x,
                                 data.t,
                                 new_y,
                                 x_unit=data.x.unit,
                                 t_unit=data.t.unit,
                                 data_unit=data.y.unit)

    def flip_align(self, pos):
        # I HOPE THIS FUNCTION BECOMES UNNECESSARY SOON
        for data in self._data_list:
            sign = np.where(data.transient[pos].y.array < 0, -1, 1)
            sign *= np.where(data.t.array < 6, -1, 1)

            data.y *= sign

        return self

    def subtract_prescans(self, until_time, from_time=None):
        self._orient_all_data('x')

        self._pre_scans = []

        for data in self._data_list:
            data.subtract_prescans(until_time, from_time)

        return self

    @staticmethod
    def _parse_data(file_path):
        data = []

        with open(file_path, 'r') as f:
            for line in f.readlines():
                data.append([complex(x.replace('i', 'j')).real for x in line.split('\t')])

            return data

    @classmethod
    def from_files(cls, path, param_path):
        files = Parser.parse_path(path)

        with open(param_path, 'r') as f:
            param = json.load(f, strict=False)

        data_list = []
        for file in files:
            data = cls._parse_data(file)

            data_list.append(TransientSpectrum(np.array(param['wavelengthAxis']),
                                               np.array(param['delayAxis']),
                                               np.array(data),
                                               'wl',
                                               'ps',
                                               'dod'))

        return cls(data_list)

    @classmethod
    def from_elDelay_files(cls, path, param_path, t_zero_idx=6, step_size=11830):
        data = cls.from_files(path, param_path)

        for d in data._data_list:
            d.t = (d.t - t_zero_idx) * step_size

        return data


if __name__ == '__main__':
    pass
