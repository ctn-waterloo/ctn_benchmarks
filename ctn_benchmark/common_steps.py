"""Common processing steps."""
# TODO unit tests
from __future__ import absolute_import, print_function

import importlib
import os
import os.path
import shelve
import time

import matplotlib.pyplot as plt
import nengo
import numpy as np

from ctn_benchmark import parameters
from ctn_benchmark.procstep import (
    Connector, MappedStep, ParametrizedMixin, Step)


class GatherStep(Step):
    items = Connector('items')

    def process(self):
        yield list(self.items)


class GenFilenameStep(MappedStep, ParametrizedMixin):
    """Generates a new filename for each input item.

    Parameters
    ----------
    basename : str
        Prefix for filenames
    """

    items = Connector('items')

    def __init__(self, basename, **kwargs):
        self.basename = basename
        super(GenFilenameStep, self).__init__(**kwargs)

    def process_item(self, items, **kwargs):
        name = self.p.basename
        uid = np.random.randint(0x7FFFFFFF)
        filename = name + '#' + time.strftime('%Y%m%d-%H%M%S')+('-%08x' % uid)

        if not os.path.exists(self.p.data_dir):
            os.mkdir(self.p.data_dir)

        return os.path.join(self.p.data_dir, filename)

    def params(self):
        self.p.add_default("Filename prefix.", basename=self.basename)
        self.p.add_default("Data directory.", data_dir='data')


class DictToTextStep(MappedStep):
    """Convert dictionaries to strings."""

    dictionary = Connector('dictionary')

    def process_item(self, dictionary, **kwargs):
        text = []
        for k, v in sorted(dictionary.items()):
            text.append('%s = %s\n' % (k, repr(v)))
        return ''.join(text)


class ParamsToDictStep(Step):
    """Provides the parameters as dictionary."""

    def __init__(self, hidden_params=None, **kwargs):
        super(ParamsToDictStep, self).__init__(**kwargs)
        self.params = parameters.ParameterSet()
        if hidden_params is None:
            hidden_params = []
        self.hidden_params = hidden_params

    def process(self):
        yield {k: v for k, v in self.params.flatten().items()
               if k not in self.hidden_params}


def AppendTextStep(*args):
    """Appends texts to each other."""
    named = {'a' + str(i): a for i, a in enumerate(args)}
    d = {a: Connector(a) for a in named}
    d['process_item'] = lambda self, **kwargs: ''.join(
        kwargs['a' + str(i)] for i in range(len(args)))
    return type('AppendTextStep', (MappedStep,), d)(**named)


class WriteToTextFileStep(MappedStep):
    """Write strings to files."""

    text = Connector('text')
    filename = Connector('filename')

    def process_item(self, filename, text, **kwargs):
        with open(filename + '.txt', 'w') as f:
            f.write(text)
        return text


class PrintTextStep(MappedStep):
    """Prints text."""

    text = Connector('text')

    def process_item(self, text, **kwargs):
        print(text)
        return text


class AddFigureOverlayStep(MappedStep, ParametrizedMixin):
    """Adds an overlay with parameters values to figures."""

    figs = Connector('figs')
    title = Connector('title')
    text = Connector('text')

    def process_item(self, figs, title, text, **kwargs):
        if figs is None:
            figs = (plt.figure(num=i) for i in plt.get_fignums())
        for fig in figs:
            fig.suptitle(title)
            if self.p.overlay:
                fig.text(0.13, 0.12, text)
        return figs

    def params(self):
        self.p.add_default('show overlay on figures', overlay=True)


class ShowAllFigsStep(Step):
    """Shows all matplotlib figures."""

    figs = Connector('figs')

    def process(self):
        self.figs.process_all()
        plt.show()
        yield


class SaveAllFigsStep(MappedStep, ParametrizedMixin):
    """Saves all matplotlib figures."""

    figs = Connector('figs')
    filename = Connector('filename')

    def process_item(self, filename, figs, **kwargs):
        if figs is None:
            figs = (plt.figure(num=i) for i in plt.get_fignums())
        for i, fig in enumerate(figs):
            fig.savefig(filename + '-' + str(i) + self.p.ext, dpi=self.p.dpi)

    def params(self):
        self.p.add_default("File extension of saved figures.", ext='.png')
        self.p.add_default("Resolution of saved figures.", dpi=300)


class BuildNengoSimStep(MappedStep, ParametrizedMixin):
    """Builds a Nengo model and returns the simulator."""

    model = Connector('model')

    def process_item(self, model, **kwargs):
        module = importlib.import_module(self.p.backend)
        Simulator = module.Simulator

        if self.p.backend == 'nengo_spinnaker':
            if not hasattr(model.config[nengo.Node], 'function_of_time'):
                import nengo_spinnaker
                nengo_spinnaker.add_spinnaker_params(model.config)
            for node in model.all_nodes:
                if (node.size_in == 0 and node.size_out > 0 and
                        callable(node.output)):
                    model.config[node].function_of_time = True

        return Simulator(model, dt=self.p.dt)

    def params(self):
        self.p.add_default("backend to use", backend='nengo')
        self.p.add_default("time step", dt=0.001)


class StartNengoGuiStep(MappedStep):
    """Starts the Nengo GUI."""

    model = Connector('model')

    def process_item(self, model, **kwargs):
        import nengo_gui
        nengo_gui.GUI(
            model=model, filename=self.__class__.__name__,
            locals=dict(model=model), interactive=False,
            allow_file_change=False).start()


class SaveNengoRawStep(MappedStep):
    """Saves raw Nengo simulation data."""

    sim = Connector('sim')
    run_result = Connector('run_result')
    filename = Connector('filename')

    def __init__(self, **kwargs):
        self.probes = {}
        super(SaveNengoRawStep, self).__init__(**kwargs)

    def add_probes(self, **kwargs):
        self.probes.update(kwargs)

    def process_item(self, sim, filename, **kwargs):
        db = shelve.open(filename + '.db')
        try:
            db['trange'] = sim.trange()
            for k, v in self.probes.items():
                db[k] = sim.data[v]
        finally:
            db.close()
