from __future__ import absolute_import, print_function

import argparse
import importlib
import inspect
import logging
import os
import shelve
import sys
import time
from weakref import WeakKeyDictionary

import matplotlib.pyplot
import matplotlib.pyplot as plt
import nengo
import numpy as np

from ctn_benchmark.parameters import ParameterSet, to_argparser


class _ActionInvoker(object):
    def __init__(self, instance, action):
        self.instance = instance
        self.action = action

    def __call__(self, p):
        if self.instance in self.action.results:
            return self.action.results[self.instance]
        self.action.f_pre(
            self.instance, p, **self.eval_dependencies(self.action.f_pre, p))
        result = self.action.f_action(
            self.instance, p,
            **self.eval_dependencies(self.action.f_action, p))
        self.action.results[self.instance] = result
        self.action.f_post(
            self.instance, p, **self.eval_dependencies(self.action.f_post, p))
        return result

    @property
    def name(self):
        return self.action.name

    def get_dependencies(self, fn):
        args = inspect.getargspec(fn).args
        dependencies = {}
        for name in args[2:]:  # first two arguments are self and p
            dependencies[name] = getattr(self.instance, name)
        return dependencies

    def eval_dependencies(self, fn, p):
        return {k: v(p) for k, v in self.get_dependencies(fn).items()}

    @property
    def params(self):
        ps = ParameterSet()
        self.action.f_params(self.instance, ps)
        return ps

    @property
    def all_params(self):
        ps = ParameterSet()
        for dependency in self.get_dependencies(self.action.f_action).values():
            ps.add_parameter_set(dependency.all_params)
        for dependency in self.get_dependencies(self.action.f_pre).values():
            ps.add_parameter_set(dependency.all_params)
        for dependency in self.get_dependencies(self.action.f_post).values():
            ps.add_parameter_set(dependency.all_params)
        ps.add_parameter_set(self.params)
        return ps


class Action(object):
    def __init__(
            self, f_action, f_params=None, pre=None, post=None, name=None):
        if name is None:
            name = f_action.__name__
        self.f_action = f_action
        if f_params is None:
            f_params = lambda inst, ps: ps
        if pre is None:
            pre = lambda inst, p: None
        if post is None:
            post = lambda inst, p: None
        self.f_params = f_params
        self.f_pre = pre
        self.f_post = post
        self.name = name
        self.results = WeakKeyDictionary()

    def params(self, f_params):
        self.f_params = f_params
        return self

    def pre(self, f_pre):
        self.f_pre = f_pre
        return self

    def post(self, f_post):
        self.f_post = f_post
        return self

    def __get__(self, instance, owner):
        return _ActionInvoker(instance, self)


def gather_actions(action_class):
    for attr_name in dir(action_class):
        attr = getattr(action_class, attr_name)
        if isinstance(attr, _ActionInvoker) and not attr_name.startswith('_'):
            yield attr_name, attr


def parse_args(action_class, default=None, argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # TODO provide useful description to parser
    parser = argparse.ArgumentParser()
    action_parsers = parser.add_subparsers(dest='action')
    actions = list(gather_actions(action_class))

    help_flag = '-h' in argv or '--help' in argv
    if default is not None and not help_flag:
        if len(argv) < 1 or argv[0] not in [a[0] for a in actions]:
            argv.insert(0, default)

    for attr_name, attr in actions:
        action_parser = action_parsers.add_parser(attr_name)
        to_argparser(attr.all_params, action_parser)

    args = parser.parse_args(args=argv)
    p = getattr(action_class, args.action).all_params
    for k in p:
        p[k] = getattr(args, k)
    return getattr(action_class, args.action), p


class Benchmark2(object):
    def __init__(self):
        self.hidden_params = ['data_dir', 'debug']
        self.probes = {}
        self.start_time = None
        self.sim_speed = None

    def add_probes(self, **kwargs):
        self.probes.update(kwargs)

    def model(self, p):
        raise NotImplementedError()

    def model_params(self, ps):
        pass

    @Action
    def _model(self, p):
        return self.model(p)

    @_model.params
    def _model(self, ps):
        self.model_params(ps)

    def evaluate(self, p, sim):
        raise NotImplementedError()

    def evaluate_params(self, ps):
        pass

    @Action
    def _evaluate(self, p, _sim):
        self.start_time = time.time()
        results = self.evaluate(p, _sim)
        if hasattr(_sim, 'close'):
            _sim.close()
        return results

    @_evaluate.params
    def _evaluate(self, ps):
        self.evaluate_params(ps)

    def record_speed(self, t):
        now = time.time()
        self.sim_speed = t / (now - self.start_time)

    def plot(self, p, sim, results):
        pass

    def plot_params(self, ps):
        pass

    @Action
    def _plot(self, p, _sim, _evaluate):
        return self.plot(p, _sim, _evaluate)

    @_plot.params
    def _plot(self, ps):
        self.plot_params(ps)

    @Action
    def _sim(self, p, _model):
        module = importlib.import_module(p.backend)
        Simulator = module.Simulator

        if p.backend == 'nengo_spinnaker':
            try:
                _ = _model.config[nengo.Node].function_of_time
            except AttributeError:
                import nengo_spinnaker
                nengo_spinnaker.add_spinnaker_params(_model.config)
            for node in _model.all_nodes:
                if (node.size_in == 0 and node.size_out > 0 and
                        callable(node.output)):
                    _model.config[node].function_of_time = True

        return Simulator(_model, dt=p.dt)

    @_sim.params
    def _sim(self, ps):
        ps.add_default('backend to use', backend='nengo')
        ps.add_default('time step', dt=0.001)

    @Action
    def _filename(self, p):
        name = self.__class__.__name__
        uid = np.random.randint(0x7FFFFFFF)
        filename = name + '#' + time.strftime('%Y%m%d-%H%M%S')+('-%08x' % uid)

        if not os.path.exists(p.data_dir):
            os.mkdir(p.data_dir)

        return os.path.join(p.data_dir, filename)

    @_filename.params
    def _filename(self, ps):
        ps.add_default('data directory', data_dir='data')

    @Action
    def _text(self, p, _evaluate):
        text = []
        for k, v in sorted(_evaluate.items()):
            text.append('%s = %s' % (k, repr(v)))
        return text

    @Action
    def _args_text(self, p):
        args_text = []
        for k in p:
            if k not in self.hidden_params:
                args_text.append('_%s = %r' % (k, getattr(p, k)))
        return args_text

    @Action
    def gui(self, p, _model):
        import nengo_gui
        nengo_gui.GUI(
            model=_model, filename=self.__class__.__name__,
            locals=dict(model=_model), interactive=False,
            allow_file_change=False).start()

    @Action
    def _figs(self, p, _filename, _text, _args_text, _plot):
        for i in plt.get_fignums():
            fig = plt.figure(i)

            if p.overlay:
                fig.suptitle(
                    os.path.basename(_filename) +'\n' + '\n'.join(_text),
                    fontsize=8)
                fig.text(0.13, 0.12, '\n'.join(_args_text))

    @_figs.params
    def _figs(self, ps):
        ps.add_default('overlay on figures', overlay=True)

    @Action
    def save_figs(self, p, _filename, _figs):
        for i in plt.get_fignums():
            fig = plt.figure(i)
            fig.savefig(_filename + p.ext, dpi=p.dpi)

    @save_figs.params
    def save_figs(self, ps):
        ps.add_default("File extension of saved figures.", fig_ext='.png')
        ps.add_default("Resolution of saved figures.", dpi=300)

    @Action
    def show_figs(self, p, _figs):
        plt.show()

    @Action
    def save_raw(self, p, _filename, _sim, _evaluate):
        db = shelve.open(_filename + '.db')
        db['trange'] = _sim.trange()
        for k, v in self.probes:
            db[k] = _sim.data[v]
        db.close()

    @Action
    def run(self, p, _filename, _args_text, _text, _evaluate):
        text = _args_text + _text
        text = '\n'.join(text)

        with open(_filename + '.txt', 'w') as f:
            f.write(text)
        print(text)

        return _evaluate

    @run.pre
    def run(self, p, _filename):
        if p.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.ERROR)
        print('running ' + os.path.basename(_filename))
        np.random.seed(p.seed)

    @run.params
    def run(self, ps):
        ps.add_default('random number seed', seed=1)
        ps.add_default('enable debug messages', debug=False)

    def invoke(self, action, **kwargs):
        action = getattr(self, action)
        return action(action.all_params.set(**kwargs))

    def main(self):
        action, p = parse_args(self, default='run')
        return action(p)


class Benchmark(object):
    def __init__(self):
        self.parameters = ParameterSet()
        self.hidden_params = [
            'data_dir', 'show_figs', 'debug', 'save_raw', 'save_figs',
            'save_results']
        self.params()

    def model(self, p):
        raise NotImplementedError()

    def evaluate(self, p, sim, plt):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def default(self, description, **kwargs):
        self.parameters.add_default(description, **kwargs)

    def record_speed(self, t):
        now = time.time()
        self.sim_speed = t / (now - self.start_time)

    @Action
    def filename(self, p):
        name = self.__class__.__name__
        uid = np.random.randint(0x7FFFFFFF)
        filename = name + '#' + time.strftime('%Y%m%d-%H%M%S')+('-%08x' % uid)
        return filename

    @Action
    def _setup(self, p, filename):
        if p.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.ERROR)
        print('running %s' % filename)

        np.random.seed(p.seed)

    @_setup.params
    def _setup(self, ps):
        ps.add_default('random number seed', seed=1)
        ps.add_default('enable debug messages', debug=False)

    @Action
    def _model(self, p, _setup):
        return self.model(p)

    @_model.params
    def _model(self, ps):
        ps.add_parameter_set(self.parameters)

    @Action
    def gui(self, p, _model):
        import nengo_gui
        nengo_gui.GUI(
            model=_model, filename=self.__class__.__name__,
            locals=dict(model=_model), interactive=False,
            allow_file_change=False).start()

    @Action
    def run_model(self, p, _model, filename):
        module = importlib.import_module(p.backend)
        Simulator = module.Simulator

        if p.backend == 'nengo_spinnaker':
            try:
                _ = _model.config[nengo.Node].function_of_time
            except AttributeError:
                import nengo_spinnaker
                nengo_spinnaker.add_spinnaker_params(_model.config)
            for node in _model.all_nodes:
                if (node.size_in == 0 and
                    node.size_out > 0 and
                    callable(node.output)):
                        _model.config[node].function_of_time = True

        if p.save_figs or p.show_figs:
            plt = matplotlib.pyplot
            plt.figure()
        else:
            plt = None
        sim = Simulator(_model, dt=p.dt)
        self.start_time = time.time()
        self.sim_speed = None
        result = self.evaluate(p, sim, plt)

        if hasattr(sim, 'close'):
            sim.close()

        if self.sim_speed is not None and 'sim_speed' not in result:
            result['sim_speed'] = self.sim_speed

        text = []
        for k, v in sorted(result.items()):
            text.append('%s = %s' % (k, repr(v)))

        args_text = []
        for k in p:
            if k not in self.hidden_params:
                args_text.append('_%s = %r' % (k, getattr(p, k)))

        if plt is not None and not p.hide_overlay:
            plt.suptitle(filename +'\n' + '\n'.join(text),
                         fontsize=8)
            plt.figtext(0.13,0.12,'\n'.join(args_text))

        text = args_text + text
        text = '\n'.join(text)

        if not os.path.exists(p.data_dir):
            os.mkdir(p.data_dir)
        filename = os.path.join(p.data_dir, filename)
        if p.save_figs:
            plt.savefig(filename + '.png', dpi=300)

        with open(filename + '.txt', 'w') as f:
            f.write(text)
        print(text)

        if p.save_raw:
            db = shelve.open(filename + '.db')
            db['trange'] = sim.trange()
            for k, v in inspect.getmembers(self):
                if isinstance(v, nengo.Probe):
                    db[k] = sim.data[v]
            db.close()

        if p.show_figs:
            plt.show()

        return result, filename

    @run_model.params
    def run_model(self, ps):
        ps.add_default('backend to use', backend='nengo')
        ps.add_default('time step', dt=0.001)
        ps.add_default('data directory', data_dir='data')
        ps.add_default('display figures', show_figs=False)
        ps.add_default('save raw data', save_raw=False)
        ps.add_default('save figures', save_figs=False)
        ps.add_default('hide overlay on figures', hide_overlay=False)

    def process_args(self, allow_cmdline=True, **kwargs):
        if allow_cmdline and len(kwargs) == 0:
            return parse_args(self, default='run_model')
        else:
            return self._run_model.all_params.set(**kwargs)

    def run(self, **kwargs):
        action, p = self.process_args(**kwargs)
        return action(p)

    def make_model(self, **kwargs):
        return self._model(self.process_args(allow_cmdline=False, **kwargs)[1])
