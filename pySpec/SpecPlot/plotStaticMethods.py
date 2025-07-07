import matplotlib as mpl
import matplotlib.colors as mcol
import math
import numpy as np


def lighten_color(color, amount=0.5):
    """
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Answer by: Ian Hincks, Accessed April 3rd 2024

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def create_colormap(axis: np.ndarray or list[float], c1='r', c2='b'):
    cmap = mcol.LinearSegmentedColormap.from_list("", [c1, c2])
    norm = mcol.Normalize(vmin=min(axis), vmax=max(axis))

    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colormap.set_array([])
    return colormap


def exponential_string(value, sign_digits=2):
    if value == 0.0:
        return "0"

    e = math.floor(math.log10(abs(value)))
    value = round(value / 10 ** e, sign_digits)

    significant = f"{value:.2}"
    exponent = "10^{" + str(e) + "}"
    if e == 0:
        return significant
    elif significant == 1:
        return exponent
    else:
        return fr"{significant} \times {exponent}"


def factor_string(value, sign_digits=2, base_exponent=0, unit='s'):
    mapping = {
         12: r'T',
          9: r'G',
          6: r'M',
          3: r'k',
          0: r'',
         -3: r'm',
         -6: r'\mu ',
         -9: r'n',
        -12: r'p',
        -15: r'f'
    }

    if value == 0.0:
        return f"0~{unit}"

    e = math.floor(math.log10(abs(value)))
    value = round(value / 10 ** e, sign_digits)
    e += base_exponent

    if ((e % 3) == 1):
        value *= 10
        e -= 1
    elif ((e % 3 == 2)):
        value *= 100
        e -= 2

    significant = f"{value:.4}"
    if e == 0:
        sign = f"{significant}"
        unt = f"{unit}"
    else:
        sign = fr"{significant}"
        unt = f"{mapping[e]}{unit}"

    return sign + "~\mathrm{" + unt + "}"


if __name__ == '__main__':
    pass
