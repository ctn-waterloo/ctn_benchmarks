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
            setattr(self, k, v)

    def process(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.process()

    def __or__(self, other):
        if isinstance(other, Step):
            connectors = [
                c for c in dir(other) if isinstance(
                    getattr(other, c), _ConnectionInfo)]
            if len(connectors) <= 0:
                raise InvalidConnectionError("Target has no connectors.")
            elif len(connectors) == 1:
                other = getattr(other, connectors[0])
            else:
                raise InvalidConnectionError(
                    "Target has more than one connector.")

        if not isinstance(other, _ConnectionInfo):
            raise InvalidConnectionError("Must connect to Connector.")

        other.pre = self
        return other.post


class ConnectionError(ValueError):
    pass

class InvalidConnectionError(ConnectionError):
    pass

class UnconnectedError(ConnectionError):
    pass

class ReconnectionError(ConnectionError):
    pass
