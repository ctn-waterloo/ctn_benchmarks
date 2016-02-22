import pytest

from ctn_benchmark import parameters


class TestParameter(object):
    @pytest.mark.parametrize('t,v', [
        (int, 42),
        (str, 'lorem ipsum'),
        (float, 0.1),
        (parameters.ParameterSet, parameters.ParameterSet())])
    def test_type_inference(self, t, v):
        p = parameters.Parameter('p', "desc", default=v)
        assert p.param_type is t

    def test_initializes_value_with_default(self):
        p = parameters.Parameter('p', "desc", default=23)
        assert p.value == 23

    def test_knows_whether_it_was_set(self):
        p = parameters.Parameter('p', "desc", default=0)
        assert p.is_set is False
        p.value = 0
        assert p.is_set is True
        p.reset()
        assert p.is_set is False

    def test_reset_restores_default(self):
        p = parameters.Parameter('p', "desc", default=0)
        assert p.value == 0
        p.value = 1
        p.reset()
        assert p.value == 0


class TestParameterSet(object):
    @pytest.fixture()
    def ps(self):
        return parameters.ParameterSet()

    def test_can_add_parameter(self, ps):
        p = parameters.Parameter('p', "desc", default=0)
        ps.add_parameter(p)
        assert 'p' in ps.params
        assert ps.params['p'] is p

    def test_does_not_allow_duplicates(self, ps):
        with pytest.raises(ValueError):
            ps.add_parameter(parameters.Parameter('p', "desc", default=0))
            ps.add_parameter(parameters.Parameter('p', "desc", default=0))

    def test_supports_add_default_shorthand(self, ps):
        ps.add_default("desc", p=2)
        assert 'p' in ps.params
        assert ps.params['p'].default == 2

    def test_default_shorthand_does_not_allow_adding_two_params_at_same_time(
            self, ps):
        with pytest.raises(ValueError):
            ps.add_default("desc", p1=1, p2=2)

    def test_provides_dict_like_access_to_parameter_values(self, ps):
        ps.add_parameter(parameters.Parameter('p', "desc", default=0))
        assert ps['p'] == 0
        ps['p'] = 1
        assert ps['p'] == 1
        del ps['p']  # should do a reset
        assert ps['p'] == 0

        assert len(ps) == 1
        assert list(iter(ps)) == ['p']

    def test_provides_attribute_like_access_to_parameter_values(self, ps):
        ps.add_parameter(parameters.Parameter('p', "desc", default=0))
        assert ps.p == 0
        ps.p = 1
        assert ps.p == 1
        ps.params['p'].reset()
        assert ps.p == 0

    def test_provides_hierarchical_attribute_access(self, ps):
        ps.add_parameter(parameters.Parameter(
            'root', "root", default=parameters.ParameterSet()))
        ps.root.add_parameter(parameters.Parameter('leaf', "leaf", default=0))
        assert ps.root.leaf == 0
        ps.root.leaf = 1
        assert ps.root.leaf == 1

    def test_conversion_to_dict(self, ps):
        ps.add_parameter(parameters.Parameter('p', "desc", default=0))
        assert dict(ps) == {'p': 0}

    def test_setting_with_kwargs(self, ps):
        ps.add_parameter(parameters.Parameter('p1', "desc", default=0))
        ps.add_parameter(parameters.Parameter('p2', "desc", default=0))
        ps.set(p1=1, p2=2)
        assert ps.p1 == 1
        assert ps.p2 == 2

        with pytest.raises(KeyError):
            ps.set(nonexistent=3)

    def test_dict_keys_are_hierarchical(self, ps):
        ps.add_default("root", root=parameters.ParameterSet())
        ps.root.add_default("leaf", leaf=2)
        ps['root.leaf'] = 3
        assert ps['root.leaf'] == 3
        del ps['root.leaf']
        assert ps['root.leaf'] == 2

    def test_provides_flattened_version(self, ps):
        ps.add_default("foo", foo=1)
        ps.add_default("root", root=parameters.ParameterSet())
        ps.root.add_default("leaf", leaf=2)
        flat = ps.flatten()
        print flat
        assert 'foo' in flat
        assert flat['foo'] == 1
        assert 'root.leaf' in flat
        assert flat['root.leaf'] == 2


class TestArgumentParserConversion(object):
    @pytest.fixture()
    def ps(self):
        ps = parameters.ParameterSet()
        ps.add_default("intarg", intarg=0)
        ps.add_default("strarg", strarg='abc')
        return ps

    def test_sets_defaults(self, ps):
        parser = parameters.to_argparser(ps)
        args = parser.parse_args([])
        assert args.intarg == 0
        assert args.strarg == 'abc'

    def test_sets_type(self, ps):
        parser = parameters.to_argparser(ps)
        args = parser.parse_args([])
        assert isinstance(args.intarg, int)
        assert isinstance(args.strarg, str)

    def test_parses_cmd_arguments(self, ps):
        parser = parameters.to_argparser(ps)
        args = parser.parse_args(['--intarg', '42', '--strarg', 'lorem'])
        assert args.intarg == 42
        assert args.strarg == 'lorem'

    def test_handles_bool(self, ps):
        ps.add_default("true", true=True)
        ps.add_default("false", false=False)
        parser = parameters.to_argparser(ps)

        args = parser.parse_args([])
        assert args.true is True
        assert args.false is False

        args = parser.parse_args(['--no_true', '--false'])
        assert args.true is False
        assert args.false is True
