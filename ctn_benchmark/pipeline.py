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
        self._pre = value


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
        for k, v in kwargs.items():
            if not hasattr(self, k) or not isinstance(
                    getattr(self, k), _ConnectionInfo):
                raise InvalidConnectionError("Must connect to Connector.")
            setattr(self, k, v)

    def process(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.process()

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
        return [getattr(self, c) for c in dir(self)
                if c != 'connectors' and isinstance(
                    getattr(self, c), _ConnectionInfo)]


class MappedStep(Step):
    def process(self):
        for items in zip(*self.connectors):
            yield self.process_item(
                **{k.name: v for k, v in zip(self.connectors, items)})

    def process_item(self, **kwargs):
        raise NotImplementedError()


class ConnectionError(ValueError):
    pass

class InvalidConnectionError(ConnectionError):
    pass

class UnconnectedError(ConnectionError):
    pass

class ReconnectionError(ConnectionError):
    pass
