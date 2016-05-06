"""Build processing pipelines."""

from __future__ import absolute_import, print_function

import logging
import sys
import time

import numpy as np

from ctn_benchmark import parameters, procstep
from ctn_benchmark.common_steps import (
    AddFigureOverlayStep, AppendTextStep, BuildNengoSimStep, GenFilenameStep,
    DictToTextStep, ParamsToDictStep, PrintTextStep, SaveAllFigsStep,
    SaveNengoRawStep, ShowAllFigsStep, StartNengoGuiStep, WriteToTextFileStep)

# TODO unit test


class Pipeline(object):
    """A pipeline allows to invoke the end steps of a pipeline as actions."""

    def __init__(self):
        super(Pipeline, self).__init__()
        self.actions = {}
        self.defaults = []

    def add_action(self, name, step):
        """Adds an action to the pipeline.

        Parameters
        ----------
        name : str
            Name of the action.
        step : :class:`ctn_benchmark.procstep.Step`
            Last processing step of the action.
        """
        self.actions[name] = step

    def setup(self, p):
        """To be called before the first action is invoked.

        Parameters
        ----------
        p : :class:`parameters.ParameterSet`
            Parameters.
        """
        yield

    def setup_params(self, p):
        """Defines setup parameters."""
        pass

    def invoke_action(self, name):
        """Invokes the necessary processing steps in a pipeline for an action.

        Parameters
        ----------
        name : str
            Action to invoke.

        Returns
        -------
        List of results.
        """
        return self.actions[name].process_all()

    def run(self, argv=None):
        """Run actions.

        Parameters
        ----------
        argv : seq, optional
            Command line arguments denoting actions to run and parameter
            values. If not given, ``sys.argv`` will be used.

        Returns
        -------
        Dictionary of the results for the actions.
        """
        actions, p = self.parse_args(argv)
        return self.run_actions(actions, p)

    def run_actions(self, actions, p):
        for a in actions:
            procstep.set_params(self.actions[a], p)
        return {a: self.invoke_action(a) for a in actions}

    def run_single_action(self, action, p):
        procstep.set_params(self.actions[action], p)
        return self.invoke_action(action)

    def parse_args(self, argv=None):
        if argv is None:
            argv = sys.argv[1:]

        # TODO provide useful description and helpful help to parser
        if len(argv) <= 0 or argv[0].startswith('-'):
            args_actions = self.defaults
        else:
            args_actions = argv.pop(0).split(',')

        p = parameters.ParameterSet()
        self.setup_params(p)
        for a in args_actions:
            p.merge_parameter_set(procstep.all_params(self.actions[a]))

        parser = parameters.to_argparser(p)
        args = parser.parse_args(args=argv)

        for k in p.flatten():
            p[k] = getattr(args, k)
        return args_actions, p


class FilenamePipeline(Pipeline):
    """Pipeline providing filenames."""
    def __init__(self):
        super(FilenamePipeline, self).__init__()
        self.filename_step = self.create_filename_step()

    def create_filename_step(self):
        raise NotImplementedError()


class PlottingPipeline(FilenamePipeline):
    """Pipeline providing plotting functionality."""

    def __init__(self):
        super(PlottingPipeline, self).__init__()
        self.plot_step = self.create_plot_step()
        self.add_action('show_figs', ShowAllFigsStep(figs=self.plot_step))
        self.add_action(
            'save_figs',
            SaveAllFigsStep(figs=self.plot_step, filename=self.filename_step))

    def plot(self, p, data, **kwargs):
        """Plot input data.

        Parameters
        ----------
        p : :class:`parameters.ParameterSet`
            Parameters (including those of steps preceding the plotting).
        data
            Data to plot.
        """
        raise NotImplementedError()

    def plot_params(self, ps):
        pass

    def create_plot_step(self):
        return procstep.ParametrizedFunctionMappedStep(
            self.plot, self.plot_params, data=self.get_data_step())

    def get_data_step(self):
        """Return processing step providing the data to plot."""
        raise NotImplementedError()


class EvaluationPipeline(FilenamePipeline):
    """Pipeline for evaluating and saving processed data."""

    def __init__(self):
        self.evaluate_step = self.create_evaluate_step()
        self.params_to_dict_step = ParamsToDictStep(['data_dir', 'debug'])
        self.text_step = AppendTextStep(
            DictToTextStep(dictionary=self.params_to_dict_step),
            DictToTextStep(dictionary=self.evaluate_step))
        super(EvaluationPipeline, self).__init__()
        self.add_action('evaluate', self.evaluate_step)
        self.add_action('run', WriteToTextFileStep(
            text=PrintTextStep(text=self.text_step),
            filename=self.filename_step))

    def setup(self, p):
        self.params_to_dict_step.params = p
        yield super(EvaluationPipeline, self).setup(p)

    def create_setup_step(self):
        return procstep.ParametrizedFunctionMappedStep(
            self.setup, self.setup_params)

    def evaluate(self, p, **kwargs):
        raise NotImplementedError()

    def evaluate_params(self, ps):
        pass

    def create_evaluate_step(self):
        return procstep.ParametrizedFunctionMappedStep(
            self.evaluate, self.evaluate_params,
            setup=self.create_setup_step())

    def create_filename_step(self):
        return GenFilenameStep(
            self.__class__.__name__, items=self.evaluate_step)


class EvaluationAndPlottingPipeline(EvaluationPipeline, PlottingPipeline):
    """Pipeline for evaluating and plotting data."""

    def __init__(self):
        super(EvaluationAndPlottingPipeline, self).__init__()

    def evaluate(self, p, **kwargs):
        raise NotImplementedError()

    def get_data_step(self):
        return self.evaluate_step


class NengoPipeline(EvaluationAndPlottingPipeline):
    """Pipeline for Nengo models."""

    def __init__(self):
        self.model_step = self.create_model_step()
        self.sim_step = BuildNengoSimStep(model=self.model_step)
        super(NengoPipeline, self).__init__()
        self.probes = {}
        self.add_action('gui', StartNengoGuiStep(model=self.model_step))
        self.save_raw_step = SaveNengoRawStep(
            sim=self.sim_step, run_result=self.evaluate_step,
            filename=self.filename_step)
        self.add_action('save_raw', self.save_raw_step)
        self.defaults = ['run']

    def add_probes(self, **kwargs):
        self.save_raw_step.add_probes(**kwargs)
        self.probes.update(**kwargs)

    def setup(self, p):
        super(NengoPipeline, self).setup(p)
        if p.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.ERROR)
        np.random.seed(p.seed)
        yield

    def setup_params(self, p):
        p.add_default("enable debug messages", debug=False)
        p.add_default("random number seed", seed=1)

    def model(self, p, setup, **kwargs):
        raise NotImplementedError()

    def model_params(self, ps):
        pass

    def create_model_step(self):
        return procstep.ParametrizedFunctionMappedStep(
            self.model, self.model_params, setup=self.create_setup_step())

    def evaluate(self, p, sim, **kwargs):
        raise NotImplementedError()

    def create_evaluate_step(self):
        return procstep.ParametrizedFunctionMappedStep(
            self.evaluate, self.evaluate_params, sim=self.sim_step)

    def plot(self, p, data, sim, **kwargs):
        raise NotImplementedError()

    def create_plot_step(self):
        return AddFigureOverlayStep(
            figs=procstep.ParametrizedFunctionMappedStep(
                self.plot, self.plot_params, data=self.get_data_step(),
                sim=self.sim_step),
            title=self.filename_step,
            text=self.text_step)


# TODO This class does not really fit into this file
class SpeedRecorder(object):
    def __init__(self, sim_time):
        self.start_time = None
        self.end_time = None
        self.sim_time = sim_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

    @property
    def duration(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def speed(self):
        return self.sim_time / self.duration
