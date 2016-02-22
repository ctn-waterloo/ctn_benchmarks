"""
Nengo Benchmark Model: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

import matplotlib.pyplot as plt
import nengo
import numpy as np

import ctn_benchmark

class CommunicationChannel(ctn_benchmark.benchmark.NengoBenchmark):
    def model_params(self, ps):
        ps.add_default('number of dimensions', D=2)
        ps.add_default('number of layers', L=2)
        ps.add_default('number of neurons per layer', N=100)
        ps.add_default('synaptic time constant', pstc=0.01)

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

            self.add_probes(
                pInput=nengo.Probe(input),
                pOutput=nengo.Probe(layers[-1], synapse=p.pstc))
        return model


    def evaluate_params(self, ps):
        ps.add_default('simulation time', T=1.0)

    def evaluate(self, p, sim):
        sim.run(p.T)
        self.record_speed(p.T)

        ideal = sim.data[self.probes['pInput']]
        for i in range(p.L):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.pstc), p.dt)

        rmse = np.sqrt(np.mean(sim.data[self.probes['pOutput']] - ideal)**2)
        return dict(rmse=rmse, ideal=ideal)

    def plot(self, p, results):
        plt.plot(self.sim.trange(), self.sim.data[self.probes['pOutput']])
        plt.plot(self.sim.trange(), results['ideal'])
        plt.ylim(-1, 1)


if __name__ == '__main__':
    CommunicationChannel().main()
