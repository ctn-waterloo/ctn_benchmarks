"""
Nengo Benchmark Model: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

import nengo
import numpy as np

import ctn_benchmark

class CommunicationChannel(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of dimensions', D=2)
        self.default('number of layers', L=2)
        self.default('number of neurons per layer', N=100)
        self.default('synaptic time constant', pstc=0.01)
        self.default('simulation time', T=1.0)

    def model(self, p):
        model = nengo.Network()
        with model:
            value = np.random.randn(p.D)
            value /= np.linalg.norm(value)

            input = nengo.Node(value)

            layers = [nengo.Ensemble(p.N, p.D) for i in range(p.L)]

            nengo.Connection(input, layers[0])
            for i in range(p.L-1):
                nengo.Connection(layers[i], layers[i+1], synapse=p.pstc)

            self.pInput = nengo.Probe(input)
            self.pOutput = nengo.Probe(layers[-1], synapse=p.pstc)
        return model


    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        self.record_speed(p.T)

        ideal = sim.data[self.pInput]
        for i in range(p.L):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.pstc), p.dt)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.pOutput])
            plt.plot(sim.trange(), ideal)
            plt.ylim(-1,1)

        rmse = np.sqrt(np.mean(sim.data[self.pOutput] - ideal)**2)
        return dict(rmse=rmse)

if __name__ == '__main__':
    CommunicationChannel().run()
