"""
Nengo SPA Benchmark Model: Semantic Memory

The model remembers and attempts to recall a sequence of bound symbols

"""

import numpy as np
import nengo
import nengo.spa as spa

import ctn_benchmark

import split

class SemanticMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('time per symbol', time_per_symbol=0.2)
        self.default('number of symbols', n_symbols=4)
        self.default('time to recall', T=1.0)
        self.default('dimensions', D=16)
        self.default('split passthrough nodes', split_max_dim=64)
        self.default('parallel filter max dimensions', pf_max_dim=16)
        self.default('parallel filter chips', pf_n_chips=1)
        self.default('parallel filter cores per chip', pf_cores=16)
        self.default('passthrough for ensembles', pass_ensembles=0)
        self.default('used a fixed seed for all ensembles', fixed_seed=False)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.word = spa.State(dimensions=p.D)
            model.marker = spa.State(dimensions=p.D)
            model.memory = spa.State(dimensions=p.D, feedback=1)

            model.cortical = spa.Cortical(spa.Actions(
                'memory = word * marker',
                ))

            def word(t):
                index = t / p.time_per_symbol
                if index < p.n_symbols:
                    return 'S%d' % index
                return '0'

            def marker(t):
                index = t / p.time_per_symbol
                if index < p.n_symbols:
                    return 'M%d' % index
                return '0'

            model.input = spa.Input(word=word, marker=marker)

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
            self.vocab = model.get_output_vocab('memory')

        split.split_input_nodes(model, max_dim=16)
        self.replaced = split.split_passthrough(model,
                                                max_dim=p.split_max_dim)
        if p.pass_ensembles > 0:
            split.pass_ensembles(model, max_dim=p.pass_ensembles)

        if p.backend == 'nengo_spinnaker':
            import nengo_spinnaker
            nengo_spinnaker.add_spinnaker_params(model.config)
            for node in model.all_nodes:
                if node.output is None:
                    if node.size_in > p.pf_max_dim:
                        print 'limiting', node
                        model.config[node].n_cores_per_chip = p.pf_cores
                        model.config[node].n_chips = p.pf_n_chips
            model.config[
                nengo_spinnaker.Simulator].placer_kwargs = dict(effort=0.1)

        split.remove_outputless_passthrough(model)

        if p.fixed_seed:
            for ens in model.all_ensembles:
                ens.seed = 1

        return model
    def evaluate(self, p, sim, plt):
        T = p.T + p.time_per_symbol * p.n_symbols
        sim.run(T)
        self.record_speed(T)

        pairs = np.zeros((p.n_symbols, p.D), dtype=float)
        for i in range(p.n_symbols):
            pairs[i] = self.vocab.parse('S%d*M%d' % (i, i)).v

        if self.p_memory not in self.replaced:
            data_memory = sim.data[self.p_memory]
        else:
            data_memory = np.hstack([sim.data[pr]
                                     for pr in self.replaced[self.p_memory]])
        memory = np.dot(pairs, data_memory.T).T

        times = sim.trange()

        mean_memory = np.mean(memory[-1])

        if plt is not None:
            plt.plot(sim.trange(), memory)


        return dict(memory = mean_memory)

if __name__ == '__main__':
    SemanticMemory().run()
