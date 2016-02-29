import pytest

from ctn_benchmark import procstep


class Producer(procstep.Step):
    def process(self):
        for i in range(3):
            yield i


class Square(procstep.Step):
    inp = procstep.Connector('inp')

    def process(self):
        return (i * i for i in self.inp)


class MappedSquare(procstep.MappedStep):
    inp = procstep.Connector('inp')

    def process_item(self, inp, **kwargs):
        assert len(kwargs) == 0
        return inp * inp


class Multiply(procstep.Step):
    a = procstep.Connector('a')
    b = procstep.Connector('b')

    def process(self):
        return (a * b for a, b in zip(self.a, self.b))


class Parametrized(procstep.Step, procstep.ParametrizedMixin):
    inp = procstep.Connector('inp')

    def process(self):
        return (i * self.p.factor for i in self.inp)

    def params(self):
        self.p.add_default("factor", factor=2)


class Increment(procstep.Step, procstep.ParametrizedMixin):
    inp = procstep.Connector('inp')

    def process(self):
        return (i + self.p.increment for i in self.inp)

    def params(self):
        self.p.add_default("increment", increment=1)


class Consumer(procstep.Step):
    inp = procstep.Connector('inp')

    def process(self):
        yield sum(self.inp)


class TestStepAPI(object):
    def test_connected_steps(self):
        pline = Consumer(inp=Square(inp=Producer()))
        assert list(pline) == [5]

    def test_invalid_connections_raise_error(self):
        with pytest.raises(procstep.InvalidConnectionError):
            Producer(prod=Producer())  # no input

        with pytest.raises(procstep.InvalidConnectionError):
            Square(process=Producer())  # not a connector

    def test_reading_from_unconnected_pipe_raises_error(self):
        with pytest.raises(procstep.UnconnectedError) as exc:
            list(Square())
        assert "'inp'" in exc.value.message

    def test_reconnection_raises_error(self):
        prod = Producer()
        sq = Square()
        sq.connect(inp=prod)
        with pytest.raises(procstep.ReconnectionError):
            sq.connect(inp=prod)

    def test_explicit_reconnection_possible(self):
        prod = Producer()
        sq = Square()
        sq.connect(inp=prod)
        del sq.inp
        sq.connect(inp=prod)

    def test_raise_error_on_invalid_connection_in_constructor(self):
        with pytest.raises(procstep.InvalidConnectionError):
            Square(nonexistent=Producer())

    def test_mapped_square(self):
        assert list(MappedSquare(inp=Producer())) == [0, 1, 4]

    def test_parametrized(self):
        assert list(Parametrized(inp=Producer())) == [0, 2, 4]

    def test_change_parameter(self):
        pline = Parametrized(inp=Parametrized(inp=Producer()))
        pline.p.factor = 3
        pline.inp.p.factor = 4
        assert list(pline) == [0, 12, 24]

    def test_all_params(self):
        pline = Parametrized(inp=Parametrized(inp=Increment(inp=Producer())))

        assert len(procstep.all_params(pline)) == 2
        assert 'factor' in procstep.all_params(pline)
        assert 'increment' in procstep.all_params(pline)

        assert len(pline.ap) == 2
        assert 'factor' in pline.ap
        assert 'increment' in pline.ap

    def test_setting_pipeline_params(self):
        pline = Parametrized(inp=Parametrized(inp=Increment(inp=Producer())))
        ps = procstep.all_params(pline)
        ps.factor = 3
        ps.increment = 2
        procstep.set_params(pline, ps)
        assert list(pline) == [18, 27, 36]

    def test_tee(self):
        prod = Producer()
        p1 = Increment(inp=prod)
        p2 = Square(inp=prod)
        assert list(p1) == [1, 2, 3]
        assert list(p2) == [0, 1, 4]


class TestFunctionMappedStep(object):
    def test_function_mapped_step(self):
        fn = lambda x: 2 * x
        pline = procstep.FunctionMappedStep(fn, x=Producer())
        assert list(pline) == [0, 2, 4]

    def test_function_mapped_step_without_inputs(self):
        fn = lambda: [23]
        pline = procstep.FunctionMappedStep(fn)
        assert list(pline) == [23]

    def test_parametrized_function_mapped_step(self):
        fn = lambda p: [p.x]
        def param_fn(p):
            p.add_default("x", x=42)

        pline = procstep.ParametrizedFunctionMappedStep(fn, param_fn)
        assert list(pline) == [42]
