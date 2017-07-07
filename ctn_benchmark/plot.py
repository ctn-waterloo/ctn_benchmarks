# import matplotlib
import numpy as np

from .data import Data
from . import stats

class Plot(object):
    def __init__(self, data):
        if isinstance(data, Data):
            data = [data]
        self.data = data

    colors = ["#1c73b3","#039f74","#d65e00","#cd79a7","#f0e542","#56b4ea"]

    def color(self, index):
        return self.colors[index % len(self.colors)]

    def measures(self, measures, plt=None, width=0.8, outlier_cutoff=3.0,
                 show_outliers=True, ylim=None):
        if plt is None:
            plt = matplotlib.pyplot
            plt.figure()

        if outlier_cutoff is not None:
            outliers = [data.find_outliers(measures, 
                                           cutoff=outlier_cutoff)
                        for data in self.data]

        for i, m in enumerate(measures):
            plt.subplot(1, len(measures), i+1)

            means = []
            error_bars = []
            for j, data in enumerate(self.data):
                d = data.get(m)
                if outlier_cutoff is not None:
                    out = d[outliers[j]]
                    d = np.delete(d, outliers[j])

                if len(d) > 0:
                    mean = np.mean(d)
                    sd = np.std(d)
                    ci = stats.bootstrapci(d, np.mean)
                else:
                    mean = 0
                    sd = 0
                    ci = [0, 0]
                means.append(mean)
                error_bars.append([mean-ci[0], ci[1]-mean])


                c = self.color(j)

                plt.fill_between([j-width/2, j+width/2], mean+sd, mean-sd, 
                                 color=c)
                plt.scatter(np.linspace(j-width/4, j+width/4, len(d)),
                              d, marker='x', color='k', s=30)
                if show_outliers and outlier_cutoff is not None:
                    plt.scatter(np.linspace(j-width/4, j+width/4, len(out)),
                              out, marker='.', color='k', s=60)
            plt.errorbar(range(len(means)), means, yerr=np.array(error_bars).T,
                         color='k', lw=2)
            plt.xticks(range(len(self.data)), [d.label for d in self.data],
                         rotation='vertical')
            plt.xlabel(m)
            if ylim is not None:
                plt.ylim(*ylim)
        plt.tight_layout()

    def vary(self, x, measures, plt=None):
        if plt is None:
            plt = matplotlib.pyplot
            plt.figure()

        for i, m in enumerate(measures):
            plt.subplot(1, len(measures), i+1)

            for j, data in enumerate(self.data):
                c = self.color(j)

                xx = data.get(x)
                d = data.get(m)

                plt.scatter(xx, d, color=c)
            plt.xlabel(m)
        plt.tight_layout()


    def lines(self, x, measures, plt=None, x_offset=0):
        if plt is None:
            plt = matplotlib.pyplot
            plt.figure()

        for i, m in enumerate(measures):
            plt.subplot(1, len(measures), i+1)

            for j, data in enumerate(self.data):
                c = self.color(j)

                xx = data.get(x)
                d = data.get(m)

                bins = {}
                for k, xxx in enumerate(xx):
                    if xxx not in bins:
                        bins[xxx] = [d[k]]
                    else:
                        bins[xxx].append(d[k])
                means = []
                error_bars = []
                x_values = []
                for x_value, y_vals in sorted(bins.items()):
                    x_values.append(x_value + x_offset*j)
                    mean = np.mean(y_vals)
                    means.append(mean)
                    ci = stats.bootstrapci(y_vals, np.mean)
                    error_bars.append([mean-ci[0], ci[1]-mean])

                plt.errorbar(x_values, means, yerr=np.array(error_bars).T,
                         color=c, lw=2)
            plt.ylabel(m)
            plt.xlabel(x)
        plt.tight_layout()


