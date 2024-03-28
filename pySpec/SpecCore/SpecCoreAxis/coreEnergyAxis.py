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

from .coreAbstractAxis import AbstractAxis
import numpy as np
import scipy.constants as spc


class EnergyAxis(AbstractAxis):
    """
    Class to represent an axis in energy units, either wavenumbers (in cm :sup:`-1`) or electron volt. Counterpart to
    wavelength axis.
    Inherits from :class:`AbstractAxis`.
    """

    __slots__ = ('_array', '_unit')

    ndim = 1

    def shift_by(self, amount: float, anchor: float = None):
        """
        :param amount: The amount to shift the axis by.
        :param anchor: The point, which is to be shifted by `amount`. Defaults to the center of the axis.

        Shifts the axis by `amount`. This is not a simple addition, since the axis is only linear in the wavelength
        domain, so before shifting the axis it has to be converted to the inverse. The amount is also specified in the
        energy unit and subsequently converted to a wavelength and added to the anchor position.
        Finally, the axis is converted back to the energy unit.
        """
        if anchor is None:
            anchor = np.nanmean(self._array)

        diff = 1 / (anchor + amount) - 1 / anchor
        x_temp = np.divide(1, self._array)
        x_temp += diff

        self._array = np.divide(1, x_temp)

    def convert_to(self, axis_type: str = None):
        """
        :param axis_type: The axis type to convert to. Possibilities are 'wl', 'ev' or 'wn'.
        :type axis_type: str

        Used to convert the energy axis to a different unit, or to a wavelength axis. The new axis is returned as a new
        instance.
        """
        if axis_type not in ('wl', 'ev', 'wn'):
            raise ValueError("Illegal Argument for axis_type.")

        if axis_type == self._unit:
            return self

        if axis_type == 'wl':
            from .coreWavelengthAxis import WavelengthAxis
            return WavelengthAxis(10_000_000 / self._array, 'wl')

        elif axis_type == 'ev':
            return EnergyAxis(
                self._array / spc.physical_constants['electron volt-inverse meter relationship'][0] / 100, 'ev'
            )
        elif axis_type == 'wn':
            return EnergyAxis(
                self._array * spc.physical_constants['electron volt-inverse meter relationship'][0] / 100, 'wn'
            )
        else:
            return


if __name__ == '__main__':
    pass
