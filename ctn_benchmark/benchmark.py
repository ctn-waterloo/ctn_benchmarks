from __future__ import absolute_import

import argparse
import importlib
import inspect
import logging
import os
import shelve
import sys
import time

import matplotlib.pyplot
import nengo
import numpy as np

from ctn_benchmark.parameters import ParameterSet, to_argparser


class _ActionInvoker(object):
    def __init__(self, instance, action):
        self.instance = instance
        self.action = action

    def __call__(self, p):
        return self.action.f_action(
            self.instance, p,
            **{k: v(p) for k, v in self.dependencies.items()})

    @property
    def dependencies(self):
        args = inspect.getargspec(self.action.f_action).args
        dependencies = {}
        for name in args[2:]:  # first two arguments are self and p
            dependencies[name] = getattr(self.instance, name)
        return dependencies

    @property
    def params(self):
        ps = ParameterSet()
        self.action.f_params(self.instance, ps)
        return ps

    @property
    def all_params(self):
        ps = ParameterSet()
        for dependency in self.dependencies.values():
            ps.add_parameter_set(dependency.all_params)
        ps.add_parameter_set(self.params)
        return ps


class Action(object):
    def __init__(self, f_action, f_params=None, name=None):
        if name is None:
            name = f_action.__name__
        self.f_action = f_action
        if f_params is None:
            f_params = lambda inst: ParameterSet()
        self.f_params = f_params
        self.name = name

    def params(self, f_params):
        self.f_params = f_params
        return self

    def __get__(self, instance, owner):
        return _ActionInvoker(instance, self)


def gather_actions(action_class):
    for attr_name in dir(action_class):
        attr = getattr(action_class, attr_name)
        if isinstance(attr, _ActionInvoker):
            yield attr_name, attr


def cmd_run_actions(action_class, default=None, argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # TODO provide useful description to parser
    parser = argparse.ArgumentParser()
    action_parsers = parser.add_subparsers(dest='action')
    actions = gather_actions(action_class)

    if default is not None:
        if len(argv) < 1 or argv[1] not in [a[0] for a in actions]:
            argv.insert(0, default)

    for attr_name in dir(action_class):
        attr = getattr(action_class, attr_name)
        if isinstance(attr, _ActionInvoker):
            action_parser = action_parsers.add_parser(attr_name)
            to_argparser(attr.all_params, action_parser)

    args = parser.parse_args(args=argv)
    return getattr(action_class, args.action)(args)


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
