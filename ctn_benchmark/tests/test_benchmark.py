from ctn_benchmark import benchmark, parameters


class TestAction(object):
    def test_specify_and_invoke_action(self):
        invoked = False
        class ActionClass(object):
            @benchmark.Action
            def dummy_action(self, p):
                invoked = True

        inst = ActionClass()
        inst.dummy_action(parameters.ParameterSet())

    def test_handles_dependencies(self):
        class ActionClass(object):
            @benchmark.Action
            def dependency(self, p):
                return 42

            @benchmark.Action
            def dependent_action(self, p, dependency):
                assert dependency == 42

        inst = ActionClass()
        inst.dependent_action(parameters.ParameterSet())

    def test_allows_to_define_parameters(self):
        class ActionClass(object):
            @benchmark.Action
            def dummy_action(self, p):
                return p.foo

            @dummy_action.params
            def dummy_action(self, ps):
                ps.add_default("foo", foo=23)

        inst = ActionClass()
        assert inst.dummy_action.params.foo == 23
        assert inst.dummy_action.all_params.foo == 23

    def test_allows_to_retrieve_all_parameters(self):
        class ActionClass(object):
            @benchmark.Action
            def dependency(self, p):
                return p.foo

            @dependency.params
            def dependency(self, ps):
                ps.add_default("foo", foo=23)

            @benchmark.Action
            def dependent_action(self, p, dependency):
                return p.bar

            @dependent_action.params
            def dependent_action(self, ps):
                ps.add_default("bar", bar=42)

        inst = ActionClass()
        assert 'foo' not in inst.dependent_action.params
        assert inst.dependent_action.params.bar == 42
        assert inst.dependent_action.all_params.foo == 23
        assert inst.dependent_action.all_params.bar == 42

    def test_caches_results(self):
        class ActionClass(object):
            def __init__(self):
                self.n_calls = 0

            @benchmark.Action
            def dummy_action(self, p):
                self.n_calls += 1

        inst = ActionClass()
        inst.dummy_action(parameters.ParameterSet())
        inst.dummy_action(parameters.ParameterSet())
        assert inst.n_calls == 1


class TestParseArgs(object):
    class ActionClass(object):
        @benchmark.Action
        def dummy_action(self, p):
            return p.foo

        @dummy_action.params
        def dummy_action(self, ps):
            ps.add_default("foo", foo=42)

    def test_invoke_action(self):
        action, p = benchmark.parse_args(
            self.ActionClass(), argv=['dummy_action'])
        assert action.name == 'dummy_action'
        assert p.foo == 42

    def test_set_parameter(self):
        action, p = benchmark.parse_args(
            self.ActionClass(), argv=['dummy_action', '--foo', '23'])
        assert action.name == 'dummy_action'
        assert p.foo == 23

    def test_invoke_default(self):
        action, p = benchmark.parse_args(
            self.ActionClass(), default='dummy_action', argv=[])
        assert action.name == 'dummy_action'
        assert p.foo == 42

        action, p = benchmark.parse_args(
            self.ActionClass(), default='dummy_action', argv=['--foo', '23'])
        assert action.name == 'dummy_action'
        assert p.foo == 23
