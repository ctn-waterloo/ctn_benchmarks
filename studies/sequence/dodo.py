
import ctn_benchmark

def run_sims(backend, data_dir):
    for i in range(30):
        ctn_benchmark.nengo.SPASequence().run(seed=i, data_dir=data_dir,
                                              backend=backend, debug=True)

def task_nengo():
    return dict(actions=[(run_sims, ['nengo', 'nengo'])])
def task_ocl():
    return dict(actions=[(run_sims, ['nengo_ocl', 'ocl'])])
def task_spinn():
    return dict(actions=[(run_sims, ['nengo_spinnaker', 'spinn'])])
def task_sp_rmv():
    return dict(actions=[(run_sims, ['nengo_spinnaker', 'sp_rmv'])])

import os
import numpy as np

class Data(object):
    def __init__(self, path, label=None):
        self.path = path
        if label is None:
            label = path
        self.label = label
        self.data = []
        for fn in os.listdir(path):
            if fn.endswith('.txt'):
                with open(os.path.join(path, fn)) as f:
                    text = f.read()
                d = dict()
                try:
                    exec(text, d)
                except:
                    continue
                self.data.append(d)

    def get(self, key):
        return np.array([d[key] for d in self.data])

import matplotlib
import pylab
class Plot(object):
    def __init__(self, data):
        if isinstance(data, Data):
            data = [data]
        self.data = data

    colors = ["#1c73b3","#039f74","#d65e00","#cd79a7","#f0e542","#56b4ea"]

    def color(self, index):
        return self.colors[index % len(self.colors)]

    def measures(self, measures, plt=None, width=0.8):
        if plt is None:
            plt = matplotlib.pyplot

        for i, m in enumerate(measures):
            plt.subplot(1, len(measures), i+1)

            means = []
            error_bars = []
            for j, data in enumerate(self.data):
                d = data.get(m)
                mean = np.mean(d)
                sd = np.std(d)
                ci = ctn_benchmark.stats.bootstrapci(d, np.mean)
                means.append(mean)
                error_bars.append([mean-ci[0], ci[1]-mean])


                c = self.color(j)

                pylab.fill_between([j-width/2, j+width/2], mean+sd, mean-sd, color=c)
                pylab.scatter(np.random.uniform(j-width/2, j+width/2,d.shape),
                              d, marker='x', color='k', s=30)
            plt.errorbar(range(len(means)), means, yerr=np.array(error_bars).T,
                         color='k', lw=2)
            pylab.xticks(range(len(self.data)), [d.label for d in self.data],
                         rotation='vertical')
            pylab.xlabel(m)
        pylab.tight_layout()



def task_plot():
    """plot results"""

    def plot():
        import pylab

        plot = Plot([Data('nengo'),
                     Data('ocl'),
                     Data('spinn'),
                     Data('sp_rmv')])

        plot.measures(['period', 'period_sd', 'peak', 'peak_sd'])

        pylab.show()

    return dict(actions=[plot])

