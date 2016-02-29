"""Base classes for defining processing steps."""

from __future__ import absolute_import, print_function

import inspect
from weakref import WeakKeyDictionary


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
    """Marks an input connector in a class defining a processing step of a
    pipeline.

    Parameters
    ----------
    name : str
        Name of the connector. Should be the same as the class attribute.
    """

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
    """Processing step in a pipeline.

    To define input connectors use class attributes of the type
    :class:`Connector`.

    Instances of this class are iterable and provide the processing results
    when iterated over. Results will be cached if the step feeds into more than
    one following step.

    Parameters
    ----------
    kwargs : dict
        Preceding steps to connect as inputs to connectors.
    """

    def __init__(self, **kwargs):
        super(Step, self).__init__()
        self.n_outgoing_connections = 0
        self._cached = []
        self._iter = None
        self.connect(**kwargs)

    def connect(self, **kwargs):
        """Connects steps as inputs to connectors.

        Parameters
        ----------
        kwargs : dict
            Connections to make.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k) or not isinstance(
                    getattr(self, k), _ConnectionInfo):
                raise InvalidConnectionError("Must connect to Connector.")
            setattr(self, k, v)

    def process(self):
        """Defines the processing to do within the step.

        Returns
        -------
        generator
            Generator providing the processing results.
        """
        raise NotImplementedError()

    def process_all(self):
        """Return a list of all processing results."""
        return list(self.process())

    @property
    def connectors(self):
        """All connectors defined on the class."""
        return [getattr(self, name) for name, c in vars(self.__class__).items()
                if isinstance(c, Connector)]

    class _Iter(object):
        def __init__(self, step):
            super(Step._Iter, self).__init__()
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
            return self._Iter(self)
        else:
            return self._iter


class MappedStep(Step):
    """Processing step that maps the same function on all input items."""

    def process(self):
        if len(self.connectors) <= 0:
            for item in self.process_item():
                yield item

        for items in zip(*self.connectors):
            yield self.process_item(
                **{k.name: v for k, v in zip(self.connectors, items)})

    def process_item(self, **kwargs):
        """Called for every set of input item.

        Parameters
        ----------
        kwargs : dict
            For each input connector one item to process.
        """
        raise NotImplementedError()


class ParametrizedMixin(object):
    """Mixin class that allows to set parameters for a step.

    Attributes
    ----------
    p : :class:`parameters.ParameterSet`
        Valid parameters defined for the step.
    ap : :class:`parameters.ParameterSet`
        Valid parameters defined for the step and all preceding steps.
    """

    def __init__(self, **kwargs):
        super(ParametrizedMixin, self).__init__(**kwargs)
        self.p = parameters.ParameterSet()
        self.params()

    def params(self):
        """Defines the valid parameters for the processing step.

        Use the methods on ``self.p`` to add parameters.
        """
        pass

    @property
    def ap(self):
        return all_params(self)


def all_parameter_sets(pipeline):
    steps = [pipeline]
    while len(steps) > 0:
        s = steps.pop()
        if hasattr(s, 'p'):
            yield s.p
        steps.extend(c.pre for c in s.connectors)


def all_params(pipeline):
    ps = parameters.ParameterSet()
    for p in all_parameter_sets(pipeline):
        ps.merge_parameter_set(p)
    return ps


def set_params(pipeline, ps):
    for p in all_parameter_sets(pipeline):
        p.set_from_ps(ps)


class ConnectionError(ValueError):
    pass

class InvalidConnectionError(ConnectionError):
    pass

class UnconnectedError(ConnectionError):
    pass

class ReconnectionError(ConnectionError):
    pass


def FunctionMappedStep(fn, **kwargs):
    """Turns a function into a processing step.

    Parameters
    ----------
    fn : func
        Function that takes one item of each input and processes it.
    kwargs : dict
        Steps providing input to the function.
    """
    args = inspect.getargspec(fn).args
    d = {a: Connector(a) for a in args if a != 'self'}
    d['process_item'] = lambda self, **kwargs: fn(**kwargs)
    return type(fn.__name__, (MappedStep,), d)(**kwargs)

def ParametrizedFunctionMappedStep(fn, params_fn, **kwargs):
    """Turns a function into a processing step and allows to set parameters.

    Parameters
    ----------
    fn : func
        Function that takes one item of each input and processes it. The first
        argument is the :class:`parameters.ParameterSet` providing the
        parameters.
    params_fn : func
        Functions that gets passed a :class:`parameters.ParameterSet` as
        argument and defines the valid parameters by adding them to the
        parameter set.
    kwargs : dict
        Steps providing input to the function.
    """
    args = inspect.getargspec(fn).args
    d = {a: Connector(a) for a in args if a != 'p' and a != 'self'}
    d['process_item'] = lambda self, **kwargs: fn(self.ap, **kwargs)
    d['params'] = lambda self: params_fn(self.p)
    return type(fn.__name__, (MappedStep, ParametrizedMixin), d)(**kwargs)
