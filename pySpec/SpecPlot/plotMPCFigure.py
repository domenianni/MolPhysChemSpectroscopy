from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
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
