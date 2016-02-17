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
            def dummy_action(self):
                ps = parameters.ParameterSet()
                ps.add_default("foo", foo=23)
                return ps

        inst = ActionClass()
        assert inst.dummy_action.params.foo == 23
