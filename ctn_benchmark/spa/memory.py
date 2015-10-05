"""
Nengo SPA Benchmark Model: Semantic Memory

The model remembers and attempts to recall a sequence of bound symbols

"""

import numpy as np
import nengo
import nengo.spa as spa

import ctn_benchmark

class SemanticMemory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('time per symbol', time_per_symbol=0.2)
        self.default('time per cue', time_per_cue=0.1)
        self.default('number of symbols', n_symbols=4)
        self.default('time to recall', T=1.0)
        self.default('dimensions', D=16)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.word = spa.State(dimensions=p.D)
            model.marker = spa.State(dimensions=p.D)
            model.memory = spa.State(dimensions=p.D, feedback=1)
            model.motor = spa.State(dimensions=p.D)
            model.cue = spa.State(dimensions=p.D)

            model.cortical = spa.Cortical(spa.Actions(
                'memory = word * marker',
                'motor = memory * ~cue',
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

            def cue(t):
                index = (t - p.time_per_symbol * p.n_symbols) / p.time_per_cue
                if index > 0:
                    index = index % (2 * p.n_symbols)
                    if index < p.n_symbols:
                        return 'S%d' % index
                    else:
                        return 'M%d' % (index - p.n_symbols)
                return '0'

            model.input = spa.Input(word=word, marker=marker, cue=cue)

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
            self.p_motor = nengo.Probe(model.motor.output, synapse=0.03)
            self.vocab = model.get_output_vocab('motor')
        return model
    def evaluate(self, p, sim, plt):
        T = p.T + p.time_per_symbol * p.n_symbols
        sim.run(T)
        self.record_speed(T)

        pairs = np.zeros((p.n_symbols, p.D), dtype=float)
        for i in range(p.n_symbols):
            pairs[i] = self.vocab.parse('S%d*M%d' % (i, i)).v
        memory = np.dot(pairs, sim.data[self.p_memory].T).T
        motor = self.vocab.dot(sim.data[self.p_motor].T).T

        times = sim.trange()

        mean_memory = np.mean(memory[-1])

        mag_correct = []
        mag_others = []
        mag_second = []
        index = 0
        for i in range(int(p.T / p.time_per_cue)):
            t = p.time_per_symbol * p.n_symbols + (i+1) * p.time_per_cue
            while index < len(times) - 1 and times[index+1] < t:
                index += 1
            correct = i % (2 * p.n_symbols)
            if correct < p.n_symbols:
                correct_index = self.vocab.keys.index('M%d' % correct)
            else:
                correct_index = self.vocab.keys.index('S%d' %
                                                      (correct - p.n_symbols))
            mag_correct.append(motor[index, correct_index])
            mag_others.append(np.mean(np.delete(motor[index], [correct_index])))
            mag_second.append(np.max(np.delete(motor[index], [correct_index])))

        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), memory)
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), motor)


        return dict(mag_correct=np.mean(mag_correct),
                    mag_others=np.mean(mag_others),
                    mag_second=np.mean(mag_second),
                    memory = mean_memory)

if __name__ == '__main__':
    SemanticMemory().run()
