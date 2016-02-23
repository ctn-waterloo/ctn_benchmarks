import argparse
from collections import MutableMapping


class Parameter(object):
    def __init__(self, name, description, default, param_type=None):
        if param_type is None:
            param_type = type(default)

        self.name = name
        self.description = description
        self.default = default
        self.param_type = param_type
        self._value = default
        self._is_set = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._is_set = True
        self._value = value

    @property
    def is_set(self):
        return self._is_set

    def reset(self):
        self._is_set = False
        self._value = self.default


class ParameterSet(MutableMapping):
    def __init__(self):
        super(ParameterSet, self).__setattr__('params', {})

    def add_parameter(self, param):
        if param.name in self.params:
            raise ValueError(
                "Parameter {0} already exists.".format(param.name))
        self.params[param.name] = param

    def merge_parameter_set(self, parameter_set):
        for param in parameter_set.params.values():
            self.params[param.name] = param

    def add_default(self, description, param_type=None, **kwargs):
        if len(kwargs) != 1:
            raise ValueError("Must specifiy exactly one parameter.")
        k, v = kwargs.items()[0]
        self.add_parameter(Parameter(k, description, v, param_type=param_type))

    def __getitem__(self, key):
        split = key.split('.', 1)
        if key in self.params or len(split) <= 1:
            return self.params[key].value
        else:
            return self.params[split[0]].value[split[1]]

    def __setitem__(self, key, value):
        split = key.split('.', 1)
        if key in self.params or len(split) <= 1:
            self.params[key].value = value
        else:
            self.params[split[0]].value[split[1]] = value

    def __delitem__(self, key):
        split = key.split('.', 1)
        if key in self.params or len(split) <= 1:
            self.params[key].reset()
        else:
            del self.params[split[0]].value[split[1]]

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def set(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        flat = ParameterSet()
        for k, v in self.params.items():
            if v.param_type is ParameterSet:
                for p in v.value.params.values():
                    flat.params[v.name + '.' + p.name] = p
            else:
                flat.params[k] = v
        return flat


def to_argparser(parameter_set, parser=None, prefix=''):
    if parser is None:
        parser = argparse.ArgumentParser()
    for v in parameter_set.params.values():
        if v.default is True:
            parser.add_argument(
                '--no_' + prefix + v.name, action='store_false', dest=v.name,
                help=v.description)
        elif v.default is False:
            parser.add_argument(
                '--' + prefix + v.name, action='store_true',
                help=v.description)
        elif v.param_type is ParameterSet:
            to_argparser(v.default, parser=parser, prefix=v.name + '.')
        else:
            parser.add_argument(
                '--' + prefix + v.name, type=v.param_type, default=v.default,
                help=v.description)
    return parser
