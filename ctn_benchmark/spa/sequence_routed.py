"""
Nengo Benchmark Model: SPA Sequence

Given no input, the model will cycle between cortical states using a
basal ganglia and thalamus.
"""

import split


import ctn_benchmark
import numpy as np
import nengo
import nengo.spa as spa

class SPASequenceRouted(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensionality', D=32)
        self.default('number of actions', n_actions=5)
        self.default('time to simulate', T=1.0)
        self.default('starting action', start=0)
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
            model.state = spa.Memory(dimensions=p.D)
            actions = ['dot(state, S%d) --> state=S%d' % (i,(i+1))
                       for i in range(p.n_actions - 1)]
            actions.append('dot(state, S%d) --> state=vision' %
                           (p.n_actions - 1))
            model.bg = spa.BasalGanglia(actions=spa.Actions(*actions))
            model.thal = spa.Thalamus(model.bg)

            model.input = spa.Input(vision='S%d' % p.start,
                                    state=lambda t: 'S%d' % p.start if
                                              t < 0.1 else '0')

            self.probe = nengo.Probe(model.thal.actions.output, synapse=0.03)

        split.split_passthrough(model, max_dim=p.split_max_dim)
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

            model.config[nengo_spinnaker.Simulator].placer_kwargs = dict(effort=0.1)

        if p.fixed_seed:
            for ens in model.all_ensembles:
                ens.seed = 1


        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        self.record_speed(p.T)

        index = int(0.05 / p.dt)  # ignore the first 50ms
        best = np.argmax(sim.data[self.probe][index:], axis=1)
        times = sim.trange()
        change = np.diff(best)
        change_points = np.where(change != 0)[0]
        intervals = np.diff(change_points * p.dt)

        best_index = best[change_points][1:]
        route_intervals = intervals[np.where(best_index == p.n_actions-1)[0]]
        seq_intervals = intervals[np.where(best_index != p.n_actions-1)[0]]

        data = sim.data[self.probe][index:]
        peaks = [np.max(data[change_points[i]:change_points[i+1]])
                 for i in range(len(change_points)-1)]

        if plt is not None:
            plt.plot(times, sim.data[self.probe]
            plt.plot(times[index + 1:], np.where(change!=0,1,0))

            for i, peak in enumerate(peaks):
                plt.hlines(peak, times[change_points[i] + index],
                                  times[change_points[i+1] + index])



        return dict(period=np.mean(seq_intervals),
                    period_sd=np.std(seq_intervals),
                    route_period=np.mean(route_intervals),
                    route_period_sp=np.std(route_intervals),
                    peak=np.mean(peaks), peak_sd=np.std(peaks))

if __name__ == '__main__':
    SPASequenceRouted().run()
