import pytest

from ctn_benchmark import parameters
from ctn_benchmark import pipeline


class Producer(pipeline.Step):
    def process(self):
        for i in range(3):
            yield i


class Square(pipeline.Step):
    inp = pipeline.Connector('inp')

    def process(self):
        return (i * i for i in self.inp)


class MappedSquare(pipeline.MappedStep):
    inp = pipeline.Connector('inp')

    def process_item(self, inp, **kwargs):
        assert len(kwargs) == 0
        return inp * inp


class Multiply(pipeline.Step):
    a = pipeline.Connector('a')
    b = pipeline.Connector('b')

    def process(self):
        return (a * b for a, b in zip(self.a, self.b))


class Parametrized(pipeline.Step, pipeline.ParametrizedMixin):
    inp = pipeline.Connector('inp')

    def process(self):
        return (i * self.p.factor for i in self.inp)

    def params(self):
        self.p.add_default("factor", factor=2)


class Increment(pipeline.Step, pipeline.ParametrizedMixin):
    inp = pipeline.Connector('inp')

    def process(self):
        return (i + self.p.increment for i in self.inp)

    def params(self):
        self.p.add_default("increment", increment=1)


class Consumer(pipeline.Step):
    inp = pipeline.Connector('inp')

    def process(self):
        yield sum(self.inp)


def test_connected_steps():
    pline = Consumer(inp=Square(inp=Producer()))
    assert list(pline) == [5]

def test_pipe_operator():
    pline = Producer() | Square().inp | Consumer().inp
    assert list(pline) == [5]

def test_pipe_operator_with_default():
    pline = Producer() | Square() | Consumer()
    assert list(pline) == [5]

def test_invalid_connections_raise_error():
    with pytest.raises(pipeline.InvalidConnectionError) as exc:
        _ = Producer() | Producer()  # no input
    assert 'no connector' in exc.value.message

    with pytest.raises(pipeline.InvalidConnectionError) as exc:
        _ = Producer() | Multiply()  # two inputs (no default)
    assert 'more than one' in exc.value.message

    with pytest.raises(pipeline.InvalidConnectionError):
        _ = Producer() | Square().process  # not a connector

def test_reading_from_unconnected_pipe_raises_error():
    with pytest.raises(pipeline.UnconnectedError) as exc:
        list(Square())
    assert "'inp'" in exc.value.message

def test_reconnection_raises_error():
    sq = Square()
    _ = Producer() | sq
    with pytest.raises(pipeline.ReconnectionError):
        _ = Producer() | sq

def test_explicit_reconnection_possible():
    sq = Square()
    _ = Producer() | sq
    del sq.inp
    _ = Producer() | sq

def test_raise_error_on_invalid_connection_in_constructor():
    with pytest.raises(pipeline.InvalidConnectionError):
        Square(nonexistent=Producer())

def test_mapped_square():
    assert list(Producer() | MappedSquare()) == [0, 1, 4]

def test_parametrized():
    assert list(Producer() | Parametrized()) == [0, 2, 4]

def test_change_parameter():
    pline = Producer() | Parametrized() | Parametrized()
    pline.p.factor = 3
    pline.inp.p.factor = 4
    assert list(pline) == [0, 12, 24]

def test_all_params():
    pline = Producer() | Increment() | Parametrized() | Parametrized()

    assert len(pipeline.all_params(pline)) == 2
    assert 'factor' in pipeline.all_params(pline)
    assert 'increment' in pipeline.all_params(pline)

def test_setting_pipeline_params():
    pline = Producer() | Increment() | Parametrized() | Parametrized()
    ps = pipeline.all_params(pline)
    ps.factor = 3
    ps.increment = 2
    pipeline.set_params(pline, ps)
    assert list(pline) == [18, 27, 36]

def test_tee():
    prod = Producer() | pipeline.Tee()
    p1 = prod | Increment()
    p2 = prod | Square()
    assert list(p1) == [1, 2, 3]
    assert list(p2) == [0, 1, 4]
