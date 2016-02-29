from __future__ import absolute_import, print_function

import importlib
import inspect
import os
import os.path
import sys
import time
from weakref import WeakKeyDictionary

import matplotlib.pyplot as plt
import nengo
import numpy as np

from ctn_benchmark import parameters


class _ConnectionInfo(object):
    def __init__(self, name, post, pre=None):
        self.name = name
        self._pre = pre
        self.post = post

    def __iter__(self):
        if self.pre is None:
            raise UnconnectedError("Input '{0}' is unconnected.".format(
                self.name))
        return iter(self.pre)

    @property
    def pre(self):
        return self._pre

    @pre.setter
    def pre(self, value):
        if value is not None and self._pre is not None:
            raise ReconnectionError(
                "Input '{0}' is already connected. Use `del` first if you"
                " want to alter a connection.".format(self.name))
        elif value is None and self._pre is not None:
            self._pre.n_outgoing_connections -= 1
        elif value is not None and self._pre is None:
            value.n_outgoing_connections += 1
        self._pre = value

    def __getattr__(self, name):
        return getattr(self.pre, name)


class Connector(object):
    def __init__(self, name):
        self.name = name
        self.connectors = WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance not in self.connectors:
            self.connectors[instance] = _ConnectionInfo(
                self.name, post=instance)
        return self.connectors[instance]

    def __set__(self, instance, value):
        if instance not in self.connectors:
            self.connectors[instance] = _ConnectionInfo(
                self.name, post=instance, pre=value)
        else:
            self.connectors[instance].pre = value

    def __delete__(self, instance):
        self.connectors[instance].pre = None


class Step(object):
    def __init__(self, **kwargs):
        super(Step, self).__init__()
        self.n_outgoing_connections = 0
        self._cached = []
        self._iter = None
        self.connect(**kwargs)

    def connect(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k) or not isinstance(
                    getattr(self, k), _ConnectionInfo):
                raise InvalidConnectionError("Must connect to Connector.")
            setattr(self, k, v)

    def process(self):
        raise NotImplementedError()

    def process_all(self):
        return list(self.process())

    def __or__(self, other):
        if isinstance(other, Step):
            if len(other.connectors) <= 0:
                raise InvalidConnectionError("Target has no connectors.")
            elif len(other.connectors) == 1:
                other = other.connectors[0]
            else:
                raise InvalidConnectionError(
                    "Target has more than one connector.")

        if not isinstance(other, _ConnectionInfo):
            raise InvalidConnectionError("Must connect to Connector.")

        other.pre = self
        return other.post

    @property
    def connectors(self):
        return [getattr(self, name) for name, c in vars(self.__class__).items()
                if isinstance(c, Connector)]

    class Iter(object):
        def __init__(self, step):
            super(Step.Iter, self).__init__()
            self.step = step
            self.i = 0

        def __iter__(self):
            return self

        def next(self):
            while self.i >= len(self.step._cached):
                a = self.step._iter.next()
                self.step._cached.append(a)
            item = self.step._cached[self.i]
            self.i += 1
            return item

    def __iter__(self):
        if self._iter is None:
            self._iter = self.process()
        if self.n_outgoing_connections > 1:
            return self.Iter(self)
        else:
            return self._iter


class MappedStep(Step):
    def process(self):
        # TODO test this
        if len(self.connectors) <= 0:
            for item in self.process_item():
                yield item

        for items in zip(*self.connectors):
            yield self.process_item(
                **{k.name: v for k, v in zip(self.connectors, items)})

    def process_item(self, **kwargs):
        raise NotImplementedError()


class ParametrizedMixin(object):
    def __init__(self, **kwargs):
        super(ParametrizedMixin, self).__init__(**kwargs)
        self.p = parameters.ParameterSet()
        self.params()

    def params(self):
        pass

    @property
    def ap(self):
        return all_params(self)


# TODO lots of shared code in next two functions
def all_params(pipeline):
    ps = parameters.ParameterSet()
    steps = [pipeline]
    while len(steps) > 0:
        s = steps.pop()
        if hasattr(s, 'p'):
            ps.merge_parameter_set(s.p)
        steps.extend(c.pre for c in s.connectors)
    return ps


def set_params(pipeline, ps):
    steps = [pipeline]
    while len(steps) > 0:
        s = steps.pop()
        if hasattr(s, 'p'):
            # TODO refactor and make this a method of parameter set
            for k, v in ps.flatten().items():
                if k in s.p:
                    s.p[k] = v
        steps.extend(c.pre for c in s.connectors)
    return ps



class ConnectionError(ValueError):
    pass

class InvalidConnectionError(ConnectionError):
    pass

class UnconnectedError(ConnectionError):
    pass

class ReconnectionError(ConnectionError):
    pass


def FunctionMappedStep(fn, **kwargs):
    args = inspect.getargspec(fn).args
    d = {a: Connector(a) for a in args if a != 'self'}
    d['process_item'] = lambda self, **kwargs: fn(**kwargs)
    return type(fn.__name__, (MappedStep,), d)(**kwargs)

def ParametrizedFunctionMappedStep(fn, params_fn, **kwargs):
    args = inspect.getargspec(fn).args
    d = {a: Connector(a) for a in args if a != 'p' and a != 'self'}
    d['process_item'] = lambda self, **kwargs: fn(self.ap, **kwargs)
    d['params'] = lambda self: params_fn(self.p)
    return type(fn.__name__, (MappedStep, ParametrizedMixin), d)(**kwargs)


class GenFilename(MappedStep, ParametrizedMixin):
    items = Connector('items')

    def __init__(self, basename, **kwargs):
        self.basename = basename
        super(GenFilename, self).__init__(**kwargs)

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


class DictToText(MappedStep):
    dictionary = Connector('dictionary')

    def process_item(self, dictionary):
        text = []
        for k, v in sorted(dictionary.items()):
            text.append('%s = %s\n' % (k, repr(v)))
        return ''.join(text)


class WriteToTextFile(MappedStep):
    text = Connector('text')
    filename = Connector('filename')

    def process_item(self, filename, text, **kwargs):
        with open(filename + '.txt', 'w') as f:
            f.write(text)
        return text


class PrintText(MappedStep):
    text = Connector('text')

    def process_item(self, text, **kwargs):
        print(text)
        return text


class ShowAllFigs(Step):
    figs = Connector('figs')

    def process(self):
        list(self.figs)
        plt.show()
        yield


class SaveAllFigs(MappedStep, ParametrizedMixin):
    figs = Connector('figs')
    filename = Connector('filename')

    def process_item(self, filename, figs, **kwargs):
        for i, fig in enumerate(figs):
            fig.savefig(filename + '-' + str(i) + self.p.ext, dpi=self.p.dpi)

    def params(self):
        self.p.add_default("File extension of saved figures.", ext='.png')
        self.p.add_default("Resolution of saved figures.", dpi=300)


class Pipeline(object):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.actions = {}
        self.defaults = []

    def add_action(self, name, step):
        self.actions[name] = step

    def invoke_action(self, name, p=None):
        if p is not None:
            set_params(self.actions[name], p)
        return self.actions[name].process_all()

    def run(self, argv=None):
        actions, p = parse_args(self, default=self.defaults, argv=argv)
        return [self.invoke_action(a) for a in actions]


class FilenamePipeline(Pipeline):
    def __init__(self):
        super(FilenamePipeline, self).__init__()
        self.filename = self.create_filename_pipeline()

    def create_filename_pipeline(self):
        raise NotImplementedError()


class PlottingPipeline(FilenamePipeline):
    def plot(self, p, data, **kwargs):
        raise NotImplementedError()

    def plot_params(self, ps):
        pass

    def create_plotting_pipeline(self):
        return ParametrizedFunctionMappedStep(
            self.plot, self.plot_params, data=self.get_data())

    def get_data(self):
        raise NotImplementedError()

    def __init__(self):
        self.plotter = self.create_plotting_pipeline()
        super(PlottingPipeline, self).__init__()
        self.add_action('show_figs', ShowAllFigs(figs=self.plotter))
        self.add_action(
            'save_figs',
            SaveAllFigs(figs=self.plotter, filename=self.filename))


class EvaluationPipeline(FilenamePipeline):
    def evaluate(self, p, **kwargs):
        raise NotImplementedError()

    def evaluate_params(self, ps):
        pass

    def create_evaluation_pipeline(self):
        return ParametrizedFunctionMappedStep(
            self.evaluate, self.evaluate_params)

    def create_filename_pipeline(self):
        return GenFilename(self.__class__.__name__, items=self.evaluater)

    def __init__(self):
        self.evaluater = self.create_evaluation_pipeline()
        super(EvaluationPipeline, self).__init__()
        self.add_action('run', WriteToTextFile(
            text=PrintText(text=DictToText(dictionary=self.evaluater)),
            filename=self.filename))


class EvaluationAndPlottingPipeline(EvaluationPipeline, PlottingPipeline):
    def __init__(self):
        super(EvaluationAndPlottingPipeline, self).__init__()

    def get_data(self):
        return self.evaluater


class BuildNengoSim(MappedStep, ParametrizedMixin):
    model = Connector('model')

    def process_item(self, model):
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


class StartNengoGui(MappedStep):
    model = Connector('model')

    def process_item(self, model):
        import nengo_gui
        nengo_gui.GUI(
            model=model, filename=self.__class__.__name__,
            locals=dict(model=model), interactive=False,
            allow_file_change=False).start()


class NengoPipeline(EvaluationAndPlottingPipeline):
    def __init__(self):
        self.probes = {}
        self._model = self.create_model_pipeline()
        self.sim = BuildNengoSim(model=self._model)
        super(NengoPipeline, self).__init__()
        self.add_action('gui', StartNengoGui(model=self._model))
        self.defaults = ['run']

    def add_probes(self, **kwargs):
        self.probes.update(kwargs)

    def model(self, p):
        raise NotImplementedError()

    def models(self, p):
        yield self.model(p)

    def model_params(self, ps):
        pass

    def create_model_pipeline(self):
        return ParametrizedFunctionMappedStep(self.models, self.model_params)

    def evaluate(self, p, sim, **kwargs):
        raise NotImplementedError()

    def create_evaluation_pipeline(self):
        return ParametrizedFunctionMappedStep(
            self.evaluate, self.evaluate_params, sim=self.sim)

    def plot(self, p, data, sim, **kwargs):
        raise NotImplementedError()

    def create_plotting_pipeline(self):
        return ParametrizedFunctionMappedStep(
            self.plot, self.plot_params, data=self.get_data(), sim=self.sim)


def parse_args(pipeline, default=None, argv=None):
    if default is None:
        default = []
    if argv is None:
        argv = sys.argv[1:]

    # TODO provide useful description and helpful help to parser
    if len(argv) <= 0 or argv[0].startswith('-'):
        args_actions = default
    else:
        args_actions = argv.pop(0).split(',')

    p = parameters.ParameterSet()
    for a in args_actions:
        p.merge_parameter_set(all_params(pipeline.actions[a]))

    parser = parameters.to_argparser(p)
    args = parser.parse_args(args=argv)

    for k in p.flatten():
        p[k] = getattr(args, k)
    return args_actions, p
