"""
Nengo Benchmark Model: Communication Channel

Input: Randomly chosen D-dimensional value
Ouput: the same value as the input
"""

import matplotlib.pyplot as plt
import nengo
import numpy as np

from ctn_benchmark.pipeline import NengoPipeline, SpeedRecorder

class CommunicationChannel(NengoPipeline):
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

            inp = nengo.Node(value)

            layers = [nengo.Ensemble(p.N, p.D) for i in range(p.L)]

            nengo.Connection(inp, layers[0])
            for i in range(p.L-1):
                nengo.Connection(layers[i], layers[i+1], synapse=p.pstc)

            self.add_probes(
                pInput=nengo.Probe(inp),
                pOutput=nengo.Probe(layers[-1], synapse=p.pstc))
        return model


    def evaluate_params(self, ps):
        ps.add_default('simulation time', T=1.0)

    def evaluate(self, p, sim):
        with SpeedRecorder(p.T) as speed_recorder:
            sim.run(p.T)

        ideal = sim.data[self.probes['pInput']]
        for _ in range(p.L):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.pstc), p.dt)

        rmse = np.sqrt(np.mean(sim.data[self.probes['pOutput']] - ideal)**2)
        return dict(rmse=rmse, ideal=ideal, speed=speed_recorder.speed)

    def plot(self, p, data, sim):
        plt.plot(sim.trange(), sim.data[self.probes['pOutput']])
        plt.plot(sim.trange(), data['ideal'])
        plt.ylim(-1, 1)


if __name__ == '__main__':
    CommunicationChannel().run()
