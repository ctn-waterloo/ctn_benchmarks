"""
Nengo Benchmark Model: Circular Convolution

Input: two random D-dimensional vectors
Output: the circular convolution of the inputs

"""

import ctn_benchmark
import nengo
import nengo.spa as spa
import numpy as np

class CircularConvolution(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensionality', D=8)
        self.default('time to run simulation', T=0.5)
        self.default('subdimensions', SD=8)
        self.default('post-synaptic time constant', pstc=0.01)
        self.default('neurons per dimension I/O', n_per_d=50)
        self.default('neurons per cconv', n_cconv=200)
    def model(self, p):
        model = spa.SPA()
        with model:
            model.inA = spa.Buffer(p.D, subdimensions=p.SD, neurons_per_dimension=p.n_per_d)
            model.inB = spa.Buffer(p.D, subdimensions=p.SD, neurons_per_dimension=p.n_per_d)

            model.result = spa.Buffer(p.D, subdimensions=p.SD, neurons_per_dimension=p.n_per_d)

            model.cortical = spa.Cortical(spa.Actions('result = inA * inB'),
                                          synapse=p.pstc,
                                          neurons_cconv=p.n_cconv)

            model.input = spa.Input(inA='A', inB='B')

            self.probe = nengo.Probe(model.result.state.output, synapse=p.pstc)

            ideal = nengo.Node(model.get_output_vocab('inA').parse('A*B').v)
            self.probe_ideal = nengo.Probe(ideal, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        self.record_speed(p.T)

        ideal = np.array(sim.data[self.probe_ideal])
        for i in range(3):
            ideal = nengo.Lowpass(p.pstc).filt(ideal, dt=p.dt, y0=0)


        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.plot(sim.trange(), ideal)


        rmse = np.sqrt(np.mean((sim.data[self.probe] - ideal)**2))
        return dict(rmse=rmse)

if __name__ == '__main__':
    CircularConvolution().run()
