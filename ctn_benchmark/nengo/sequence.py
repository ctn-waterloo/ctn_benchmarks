"""
Nengo Benchmark Model: SPA Sequence

Given no input, the model will cycle between cortical states using a
basal ganglia and thalamus.
"""


import ctn_benchmark
import numpy as np
import nengo
import nengo.spa as spa

class SPASequence(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensionality', D=32)
        self.default('number of actions', n_actions=5)
        self.default('time to simulator', T=1.0)

    def model(self, p):
        model = spa.SPA()
        with model:
            model.state = spa.Memory(dimensions=p.D)
            actions = ['dot(state, S%d) --> state=S%d' % (i,(i+1)%p.n_actions)
                       for i in range(p.n_actions)]
            model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
            model.thal = spa.Thalamus(model.bg)

            def state_input(t):
                if t < 0.1:
                    return 'S0'
                else:
                    return '0'
            model.input = spa.Input(state=state_input)

            self.probe = nengo.Probe(model.thal.actions.output, synapse=0.03)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        self.record_speed(p.T)

        index = int(0.05 / p.dt)
        best = np.argmax(sim.data[self.probe][index:], axis=1)
        change = np.diff(best)
        change_points = np.where(change != 0)[0]
        intervals = np.diff(change_points * p.dt)

        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.probe])
            plt.plot(sim.trange()[index + 1:], np.where(change!=0,1,0))



        return dict(period=np.mean(intervals), period_sd=np.std(intervals))

if __name__ == '__main__':
    SPASequence().run()
