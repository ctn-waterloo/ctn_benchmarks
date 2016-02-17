from __future__ import absolute_import

import argparse
import importlib
import inspect
import logging
import os
import shelve
import time

import matplotlib.pyplot
import nengo
import numpy as np

from ctn_benchmark.parameters import ParameterSet, to_argparser


class Benchmark(object):
    def __init__(self):
        self.parameters = ParameterSet()
        self.hidden_params = []
        self.fixed_params()
        self.params()

    def default(self, description, **kwargs):
        self.parameters.add_default(description, **kwargs)

    def fixed_params(self):
        self.default('backend to use', backend='nengo')
        self.default('time step', dt=0.001)
        self.default('random number seed', seed=1)
        self.default('data directory', data_dir='data')
        self.default('display figures', show_figs=False)
        self.default('enable debug messages', debug=False)
        self.default('save raw data', save_raw=False)
        self.default('save figures', save_figs=False)
        self.default('hide overlay on figures', hide_overlay=False)
        self.default('save results', save_results=False)
        self.default('use nengo_gui', gui=False)
        self.hidden_params.extend(['data_dir', 'show_figs', 'debug',
                                   'save_raw', 'save_figs', 'save_results'])

    def process_args(self, allow_cmdline=True, **kwargs):
        param_parser = argparse.ArgumentParser(
                parents=[to_argparser(self.parameters)],
                description="Nengo benchmark: " + self.__class__.__name__)

        if len(kwargs) == 0 and allow_cmdline:
            args = param_parser.parse_args()
        else:
            args = argparse.Namespace()
            for k in self.parameters:
                v = kwargs.get(k, param_parser.get_default(k))
                setattr(args, k, v)

        name = self.__class__.__name__
        self.args_text = []
        for k in self.parameters:
            if k not in self.hidden_params:
                self.args_text.append('_%s = %r' % (k, getattr(args, k)))

        uid = np.random.randint(0x7FFFFFFF)
        filename = name + '#' + time.strftime('%Y%m%d-%H%M%S')+('-%08x' % uid)

        return args, filename

    def make_model(self, **kwargs):
        p, fn = self.process_args(allow_cmdline=False, **kwargs)
        np.random.seed(p.seed)
        model = self.model(p)
        return model

    def record_speed(self, t):
        now = time.time()
        self.sim_speed = t / (now - self.start_time)

    def run(self, **kwargs):
        p, fn = self.process_args(**kwargs)
        if p.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.ERROR)
        print('running %s' % fn)
        np.random.seed(p.seed)

        model = self.model(p)
        if p.gui:
            import nengo_gui
            nengo_gui.GUI(model=model, filename=self.__class__.__name__,
                          locals=dict(model=model), interactive=False,
                          allow_file_change=False).start()
            return
        module = importlib.import_module(p.backend)
        Simulator = module.Simulator

        if p.backend == 'nengo_spinnaker':
            try:
                _ = model.config[nengo.Node].function_of_time
            except AttributeError:
                import nengo_spinnaker
                nengo_spinnaker.add_spinnaker_params(model.config)
            for node in model.all_nodes:
                if (node.size_in == 0 and
                    node.size_out > 0 and
                    callable(node.output)):
                        model.config[node].function_of_time = True

        if p.save_figs or p.show_figs:
            plt = matplotlib.pyplot
            plt.figure()
        else:
            plt = None
        sim = Simulator(model, dt=p.dt)
        self.start_time = time.time()
        self.sim_speed = None
        result = self.evaluate(p, sim, plt)

        if p.backend == 'nengo_spinnaker':
            sim.close()

        if self.sim_speed is not None and 'sim_speed' not in result:
            result['sim_speed'] = self.sim_speed

        text = []
        for k, v in sorted(result.items()):
            text.append('%s = %s' % (k, repr(v)))


        if plt is not None and not p.hide_overlay:
            plt.suptitle(fn +'\n' + '\n'.join(text),
                         fontsize=8)
            plt.figtext(0.13,0.12,'\n'.join(self.args_text))

        text = self.args_text + text
        text = '\n'.join(text)

        if not os.path.exists(p.data_dir):
            os.mkdir(p.data_dir)
        fn = os.path.join(p.data_dir, fn)
        if p.save_figs:
            plt.savefig(fn + '.png', dpi=300)

        with open(fn + '.txt', 'w') as f:
            f.write(text)
        print(text)

        if p.save_raw:
            db = shelve.open(fn + '.db')
            db['trange'] = sim.trange()
            for k, v in inspect.getmembers(self):
                if isinstance(v, nengo.Probe):
                    db[k] = sim.data[v]
            db.close()

        if p.show_figs:
            plt.show()

        return result
