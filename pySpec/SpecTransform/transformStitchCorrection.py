import numpy as np
from scipy.optimize import minimize


class StitchCorrection:

    def treat_stitch(self,
                     x: list,
                     t: list,
                     y: np.ndarray,
                     # nan_col_no: list or None = None,
                     stitch_diagnostics: bool = False,
                     asymmetric_stitch: bool = True,
                     method: str = 'SLSQP'):
        """
        :Notes:
        To treat the systematic error of varying offsets between stitching blocks within timesteps by minimizing the
        slope between single points of all blocks. Can also include differing treatment between even and odd pixels.

        :param x: The wavelength/wavenumber axis of the measurements, without being sorted. Is also used to determine
        the position of the stitching blocks.
        :param t: The timeaxis.
        :param y: The DeltaOD values of the measurement with form y[x,t].
        nan_col_no: To eliminate single stitching blocks from the dataset. A list of lists which contains at
        position [i][0] the integer index of the timestep, and at positions [0][1:] strings of the index of the unwanted
        stitching block at that timestep.
        E.g.: [10, '2', '3']
        :param stitch_diagnostics: Boolean. If true, plots for every time step the stitching blocks in different
        colours.
        :param asymmetric_stitch: Boolean. Toggles if even and odd pixels are to be corrected independently of each
        other.
        :param method: The minimization algorithm utilized. Default is SLSQP; for all methods, see the documentation for
        scipy.optimize.minimize
        :return: Numpy array of the corrected stitching blocks with form y[x,t], though still NOT ordered along the
        wavelength axis.
        """

        block_x, block_y = self.separate_stitch_blocks(x, y)

        # block_y_uncorr = copy.copy(block_y)
        # nan_col_no_complete = copy.copy(nan_col_no)

        # if nan_col_no:
        #     for i in nan_col_no:
        #         if len(i) != 1:
        #             for sb in i[1:]:
        #                 block_y[int(sb)][i[0]][:] = float('NaN')

        # Actual stitch correction algorithm
        for i, time in enumerate(t):
            bool_block = []

            # First reduce block_y to one time delay
            block_y_red = np.array(block_y[:, i])

            # create the bool_block array, which shows, which stitching blocks are not eliminated
            for j, _ in enumerate(block_y):
                bool_block.append(True)

            # sets the eliminated stitching blocks
            # if nan_col_no and i == nan_col_no[0][0]:
            #     elim = nan_col_no.pop(0)
            #     for j in elim[1:]:
            #         bool_block[int(j)] = False
            off_block = block_y_red[bool_block]
            # Prepare starting values (all zeroes) for minimisation
            offset = np.zeros(np.count_nonzero(bool_block))
            # skip time step, if completely empty
            if not np.size(offset):
                continue

            if asymmetric_stitch:
                offset = np.tile(np.array([0, 0], dtype=np.float64), len(offset))

                off = minimize(self.asymmetric_offset_correction, offset, (off_block, block_x), method=method)
                # off = minimize(pySC.stitchCorrFunc, offset, (np.array(x), y[i]), method=method)

                # print(off)

                k = 0
                for j, block in enumerate(bool_block):
                    if block:
                        block_y[j, i, ::2] += off['x'][k]
                        block_y[j, i, 1::2] += off['x'][k+1]
                        k += 2
            else:
                off = minimize(self.offset_correction, offset, (off_block, block_x), method=method)
                # print(off)

                k = 0
                for j, block in enumerate(bool_block):
                    if block:
                        block_y[j, i, :] += off['x'][k]
                        k += 1

        # now re-assemble all stitching block into one array
        y = []
        for i in block_y:
            y.extend(np.transpose(i))

        # if stitch_diagnostics:
        #     cmap = mpl.cm.tab10(np.linspace(0, 1, 10))
        #     for i in nan_col_no_complete:
        #         fig = plt.figure(figsize=(10, 5))
        #         grid = plt.GridSpec(1, 2, hspace=0.05, wspace=0.05)
        #         fig_ = fig.add_subplot(grid[0, 0])
        #         fig_after = fig.add_subplot(grid[0, 1])
        #         for x in enumerate(block_x):
        #             fig_after.plot(x[1], block_y[x[0], i[0]], color=cmap[x[0]])
        #             fig_.plot(x[1], block_y_uncorr[x[0], i[0]], color=cmap[x[0]], ls='--')
        #             fig_.text(0.1, 0.9 - 0.05* x[0], 'Block: ' + str(x[0]) + ' Time: ' + str(t[i[0]]),
        #             transform=fig_.transAxes, color=cmap[x[0]])
        #         plt.show()

        return np.transpose(np.array(y))

    @staticmethod
    def asymmetric_offset_correction(offset: np.ndarray, block_y: np.ndarray, block_x: np.ndarray) -> float:
        """
        :param offset: The additive offset double in size of the amount of stitching blocks. Even and odd indices
        correspond to even and odd pixels respectively.
        :param block_y: The stitching blocks as numpy arrays with form y[stitching block, x].
        :param block_x: The corresponding x-values in the same structure as above.
        :return: The average total variation between the stitching blocks as a single float.
        """
        diff = []
        offset_0 = np.empty_like(block_y[-1])
        offset_1 = np.empty_like(block_y[-1])

        for i, (y, x) in enumerate(zip(block_y, block_x)):
            for j, (y_2, x_2) in enumerate(zip(block_y, block_x)):
                if j != i:
                    offset_0[::2] = offset[2*i]
                    offset_0[1::2] = offset[2*i+1]
                    offset_1[::2] = offset[2*j]
                    offset_1[1::2] = offset[2*j+1]
                    diff.append(np.mean(abs(y + offset_0 - y_2 - offset_1)/abs(x-x_2)))
                    diff.append(np.mean(abs(y[:-1] + offset_0[:-1] - y_2[1:] - offset_1[1:]) / abs(x[:-1] - x_2[1:])))

        return np.mean(diff, dtype=np.float64)

    @staticmethod
    def offset_correction(offset: np.ndarray, block_y: np.ndarray, block_x: np.ndarray) -> float:
        """
        :param offset: The additive offset in the same size as the amount of stitching blocks.
        :param block_y: The stitching blocks as numpy arrays with form y[stitching block, x].
        :param block_x: The corresponding x-values in the same structure as above.
        :return: The average total variation between the stitching blocks as a single float.
        """
        diff = []
        for i, (y, x) in enumerate(zip(block_y, block_x)):
            for j, (y_2, x_2) in enumerate(zip(block_y, block_x)):
                if j != i:
                    diff.append(np.mean(abs(y - y_2 + offset[i] - offset[j])/abs(x-x_2)))
                    diff.append(np.mean(abs(y[:-1] - y_2[1:] + offset[i] - offset[j]) / abs(x[:-1] - x_2[1:])))
        return np.mean(diff, dtype=np.float64)

    @staticmethod
    def derive_stitch_blocks(x: list) -> list:
        # Derive stitching blocks from wavenumber axis
        stitch = [0]
        j = 2
        while j < len(x) - 1:
            d1 = x[j - 1] - x[j - 2]
            d2 = x[j] - x[j - 1]
            d3 = x[j + 1] - x[j]
            dd1 = int(d1) - int(d2)
            dd2 = int(d2) - int(d3)
            if dd2 != 0 and dd2 == -dd1:
                stitch.append(j)
            j += 1
        stitch.append(-1)
        return stitch

    def separate_stitch_blocks(self, x: list, y: np.ndarray) -> (np.ndarray, np.ndarray):
        stitch = self.derive_stitch_blocks(x)

        block_x = []
        block_y = []

        i = 0
        for end in stitch[1:]:
            if end != -1:
                block_x.append(x[stitch[i]:end])
                block_y.append(y[:, stitch[i]:end])
                i += 1
            else:
                block_x.append(x[stitch[i]:])
                block_y.append(y[:, stitch[i]:])

        return np.array(block_x), np.array(block_y)
