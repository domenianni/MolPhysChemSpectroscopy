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
from ..coreFunctions import inPlaceOp


class WavelengthAxis(AbstractAxis):

    def shift_by(self, amount: float, anchor: None = None):
        """"""
        # Has to return only an array, if used in conjunction with Operator+ and assignment into a property for some
        # inexplicable reason
        return self._array + amount

    def convert_to(self, axis_type: str = None):
        """
        :param axis_type: The axis type to convert to. Possibilities are 'wl', 'ev' or 'wn'.

        Used to convert the energy axis to a different unit, or to a wavelength axis. The new axis is returned as a new
        instance.
        """
        if axis_type not in ('wl', 'ev', 'wn'):
            raise ValueError("Illegal Argument for axis_type.")

        if axis_type == self._unit:
            return self

        if axis_type == 'wn':
            from .coreEnergyAxis import EnergyAxis
            return EnergyAxis(10_000_000 / self._array, 'wn')

        else:
            return
