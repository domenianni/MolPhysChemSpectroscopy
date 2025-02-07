from enum import Enum, auto


class EnergyUnit(Enum):
    WAVENUMBER = 'Wavenumbers / cm$^{-1}$'
    wn = WAVENUMBER
    cm = WAVENUMBER

    WAVELENGTH = 'Wavelength / nm'
    wl = WAVELENGTH
    nm = WAVELENGTH

    EV = 'Energy / eV'
    ev = EV


class TimeUnit(Enum):
    SECOND      = 's',      1
    MILLISECOND = 'ms',     1e-3
    MICROSECOND = '$\mu$s', 1e-6
    NANOSECOND  = 'ns',     1e-9
    PICOSECOND  = 'ps',     1e-12
    FEMTOSECOND = 'fs',     1e-15


class DataUnit(Enum):
    OD = 'OD'
    od = OD

    MOD = 'mOD'
    mod = MOD

    DELTAOD = r'$\Delta OD$'
    dod = DELTAOD

    DELTAMOD = r'$\Delta mOD$'
    dmod = DELTAMOD
    mdod = DELTAMOD

    DELTADELTAOD = r'$\Delta\Delta$OD'
    ddod = DELTADELTAOD

    DELTADELTAMOD = r"$\Delta\Delta$mOD"
    ddmod = DELTADELTAMOD

    EPSILON = r"$\epsilon$ / M$^{-1}$cm$^{-1}$"
    eps = EPSILON
    epsilon = EPSILON

    FOSC = 'f$_{osc.}$'
    fosc = FOSC
    f_osc = FOSC


if __name__ == '__main__':
    pass
