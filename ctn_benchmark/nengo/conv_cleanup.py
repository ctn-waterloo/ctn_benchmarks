import ctn_benchmark
import nengo
import nengo.spa as spa
import numpy as np

class ConvolutionCleanup(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensionality', D=16)
        self.default('memory time constant', mem_tau=0.1)
        self.default('input scaling on memory', mem_input_scale=0.5)
        self.default('amount of time to test memory for', test_time=10.0)
        self.default('amount of time per test', test_present_time=0.1)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.shape = spa.Buffer(p.D)
            model.color = spa.Buffer(p.D)
            model.bound = spa.Buffer(p.D)

            cconv = nengo.networks.CircularConvolution(n_neurons=200,
                                        dimensions=p.D)

            nengo.Connection(model.shape.state.output, cconv.A)
            nengo.Connection(model.color.state.output, cconv.B)
            nengo.Connection(cconv.output, model.bound.state.input,
                             transform=p.mem_input_scale, synapse=p.mem_tau)

            deconv = nengo.networks.CircularConvolution(n_neurons=200,
                                        dimensions=p.D, invert_b=True)
            deconv.label = 'deconv'

            model.query = spa.Buffer(p.D)
            model.result = spa.Buffer(p.D)

            nengo.Connection(model.bound.state.output, deconv.A)
            nengo.Connection(model.query.state.output, deconv.B)

            nengo.Connection(deconv.output, model.result.state.input,
                            transform=2)

            nengo.Connection(model.bound.state.output, model.bound.state.input,
                                synapse=p.mem_tau)

            vocab = model.get_output_vocab('result')
            model.cleanup = spa.AssociativeMemory([
                vocab.parse('RED').v,
                vocab.parse('BLUE').v,
                vocab.parse('CIRCLE').v,
                vocab.parse('SQUARE').v])

            model.clean_result = spa.Buffer(p.D)

            nengo.Connection(model.result.state.output, model.cleanup.input)
            nengo.Connection(model.cleanup.output,
                             model.clean_result.state.input)

            stim_time = p.mem_tau / p.mem_input_scale
            self.stim_time = stim_time
            def stim_color(t):
                if 0 < t < stim_time:
                    return 'BLUE'
                elif stim_time < t < stim_time*2:
                    return 'RED'
                else:
                    return '0'

            def stim_shape(t):
                if 0 < t < stim_time:
                    return 'CIRCLE'
                elif stim_time < t < stim_time*2:
                    return 'SQUARE'
                else:
                    return '0'

            def stim_query(t):
                if t < stim_time*2:
                    return '0'
                else:
                    index = int((t - stim_time) / p.test_present_time)
                    return ['BLUE', 'RED', 'CIRCLE', 'SQUARE'][index % 4]

            model.input = spa.Input(
                shape = stim_shape,
                color = stim_color,
                query = stim_query,
                )

            self.probe = nengo.Probe(model.clean_result.state.output,
                                     synapse=0.02)
            self.probe_wm = nengo.Probe(model.bound.state.output, synapse=0.02)

        self.vocab = model.get_output_vocab('clean_result')
        self.vocab_wm = model.get_output_vocab('bound')
        return model

    def evaluate(self, p, sim, plt):
        stim_time = self.stim_time
        T = stim_time * 2 + p.test_time
        sim.run(T)
        self.record_speed(T)

        answer_offset = 0.025
        if p.backend == 'nengo_spinnaker':
            answer_offset += 0.010  # spinnaker has delays due to passthroughs

        vocab = self.vocab
        vals = [None] * 4
        vals[0] = np.dot(sim.data[self.probe], vocab.parse('CIRCLE').v)
        vals[1] = np.dot(sim.data[self.probe], vocab.parse('SQUARE').v)
        vals[2] = np.dot(sim.data[self.probe], vocab.parse('BLUE').v)
        vals[3] = np.dot(sim.data[self.probe], vocab.parse('RED').v)
        vals = np.array(vals)

        vocab_wm = self.vocab_wm
        vals_wm = [None] * 2
        vals_wm[0] = np.dot(sim.data[self.probe_wm],
                            vocab_wm.parse('BLUE*CIRCLE').v)
        vals_wm[1] = np.dot(sim.data[self.probe_wm],
                            vocab_wm.parse('RED*SQUARE').v)
        vals_wm = np.array(vals_wm)

        correct = np.zeros_like(vals)
        for i, t in enumerate(sim.trange()):
            if t > stim_time * 2 + answer_offset:
                index = int((t - stim_time * 2 - answer_offset) /
                            p.test_present_time)
                correct[index % 4, i] = 1.0

        rmse = np.sqrt(np.mean((vals - correct).flatten()**2))

        if plt is not None:
            plt.subplot(2,1,1)
            plt.plot(sim.trange(), vals.T)
            plt.plot(sim.trange(), correct.T)
            plt.subplot(2,1,2)
            plt.plot(sim.trange(), vals_wm.T)

        return dict(rmse=rmse)

if __name__ == '__main__':
    ConvolutionCleanup().run()
