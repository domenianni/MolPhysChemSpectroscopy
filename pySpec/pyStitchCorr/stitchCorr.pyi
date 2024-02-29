import numpy as np

def stitchCorr(x: np.ndarray, t: np.ndarray, y: np.ndarray, block_count: int, copy: bool) -> np.ndarray: ...
# :param x, t, y: numpy-arrays from python call. No changes necessary.
#
# :param blockCount: Amount of stitching blocks utilised.
#
# :param copy: flag, whether arrays should be copied inside the stitch corr class.
#
# :returns: numpy array of calculated offsets. y-values are changed also automatically, if copy=false.
#

def stitchCorrFunc(offset: np.ndarray, x: np.ndarray, y: np.ndarray) -> float: ...
# Pure target function to be called from python frontend and minimized within this context. Only one timestep at
# a time is processed.
#
# :param offset: Numpy array of offset values, which are added on top of DOD values.
#
# :param x, y: Numpy array of the respective wavelength and DOD values.
#
# :returns: Mean total variance (one double value).
#
# See documentation inside the class for more details on the evaluated function.
#