"""
Nengo Benchmark Model #6: Simple Parsing

The model parses and executes simple commands sequentially presented to it

"""

import numpy as np
import nengo
import nengo.spa as spa

import ctn_benchmark

import split

class Parsing(ctn_benchmark.Benchmark):
    def params(self):
        self.default('time per word', time_per_word=0.5)
        self.default('dimensions', D=32)
        self.default('split passthrough nodes', split_max_dim=64)
        self.default('parallel filter max dimensions', pf_max_dim=16)
        self.default('parallel filter chips', pf_n_chips=1)
        self.default('parallel filter cores per chip', pf_cores=16)
        self.default('passthrough for ensembles', pass_ensembles=0)
        self.default('used a fixed seed for all ensembles', fixed_seed=False)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.vision = spa.Buffer(dimensions=p.D)
            model.phrase = spa.Buffer(dimensions=p.D)
            model.motor = spa.Buffer(dimensions=p.D)
            model.noun = spa.Memory(dimensions=p.D, synapse=0.1)
            model.verb = spa.Memory(dimensions=p.D, synapse=0.1)

            model.bg = spa.BasalGanglia(spa.Actions(
                'dot(vision, WRITE) --> verb=vision',
                'dot(vision, ONE+TWO+THREE) --> noun=vision',
                '0.5*(dot(vision, NONE-WRITE-ONE-TWO-THREE) + '
                     'dot(phrase, WRITE*VERB))'
                     '--> motor=phrase*~NOUN',
                ))
            model.thal = spa.Thalamus(model.bg)

            model.cortical = spa.Cortical(spa.Actions(
                'phrase=noun*NOUN',
                'phrase=verb*VERB',
                ))

            def vision_input(t):
                index = int(t / p.time_per_word) % 3
                return ['WRITE', 'ONE', 'NONE'][index]
            model.input = spa.Input(vision=vision_input)

            self.motor_vocab = model.get_output_vocab('motor')
            self.p_thal = nengo.Probe(model.thal.actions.output, synapse=0.03)
            self.p_motor = nengo.Probe(model.motor.state.output, synapse=0.03)

        split.split_input_nodes(model, max_dim=64)
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

        if p.fixed_seed:
            for ens in model.all_ensembles:
                ens.seed = 1

        return model
    def evaluate(self, p, sim, plt):
        T = p.time_per_word * 3
        sim.run(T)
        self.record_speed(T)

        if self.p_motor not in self.replaced:
            data_motor = sim.data[self.p_motor]
        else:
            data_motor = np.hstack([sim.data[pr]
                                     for pr in self.replaced[self.p_motor]])

        data = self.motor_vocab.dot(data_motor.T).T
        mean = np.mean(data[int(p.time_per_word*2.5/p.dt):], axis=0)
        correct_index = self.motor_vocab.keys.index('ONE')
        mag_correct = mean[correct_index]
        mag_others = np.mean(np.delete(mean, [correct_index]))
        mag_second = np.max(np.delete(mean, [correct_index]))


        if plt is not None:
            plt.subplot(2, 1, 1)
            plt.plot(sim.trange(), sim.data[self.p_thal])
            plt.subplot(2, 1, 2)
            plt.plot(sim.trange(), data)
            plt.plot(sim.trange(), data[:,correct_index], lw=2)

        return dict(mag_correct=mag_correct, mag_others=mag_others,
                    mag_second=mag_second)

if __name__ == '__main__':
    Parsing().run()
