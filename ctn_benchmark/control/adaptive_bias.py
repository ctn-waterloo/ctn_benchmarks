import time

import nengo
import numpy as np

import ctn_benchmark
import ctn_benchmark.control as ctrl


class ZeroDecoder(nengo.solvers.Solver):
    weights = False

    def __call__(self, A, Y, rng=None, E=None):
        return np.zeros((A.shape[1], Y.shape[1]), dtype=float), []


class AdaptiveBias(ctn_benchmark.Benchmark):

    def params(self):
        self.default('Kp', Kp=2.0)
        self.default('Kd', Kd=1.0)
        self.default('Ki', Ki=0.0)
        self.default('tau_d', tau_d=0.001)
        self.default('T', T=10.0)
        self.default('period', period=4.0)
        self.default('use adaptation', adapt=False)
        self.default('n_neurons', n_neurons=500)
        self.default('learning rate', learning_rate=1.0)
        self.default('max_freq', max_freq=1.0)
        self.default('synapse', synapse=0.01)
        self.default('radius', radius=1.0)
        self.default('number of dimensions', D=1)
        self.default('scale_add', scale_add=1)
        self.default('noise', noise=0.1)
        self.default('filter', filter=0.01)
        self.default('delay', delay=0.01)

    def model(self, p):

        model = nengo.Network()
        with model:

            system = ctrl.System(p.D, p.D, dt=p.dt, seed=p.seed,
                                 motor_noise=p.noise, sense_noise=p.noise,
                                 scale_add=p.scale_add,
                                 motor_scale=10,
                                 motor_delay=p.delay, sensor_delay=p.delay,
                                 motor_filter=p.filter, sensor_filter=p.filter)
            self.system = system

            self.system_state = []
            self.system_desired = []
            self.system_t = []
            self.system_control = []

            def minsim_system(t, x):
                self.system_control.append(x)
                self.system_desired.append(signal.value(t))
                self.system_t.append(t)
                self.system_state.append(system.state)
                return system.step(x)

            minsim = nengo.Node(minsim_system, size_in=p.D, size_out=p.D,
                                label='minsim')

            state_node = nengo.Node(lambda t: system.state, label='state')

            pid = ctrl.PID(p.Kp, p.Kd, p.Ki, tau_d=p.tau_d)
            control = nengo.Node(lambda t, x: pid.step(x[:p.D], x[p.D:]),
                                 size_in=p.D*2, label='control')
            nengo.Connection(minsim, control[:p.D], synapse=0)
            nengo.Connection(control, minsim, synapse=None)

            if p.adapt:

                adapt = nengo.Ensemble(p.n_neurons, dimensions=p.D,
                                       radius=p.radius, label='adapt')
                nengo.Connection(minsim, adapt, synapse=None)
                # adapt_signal = nengo.Node(None, size_in=p.D,
                #                           label='adapt_signal')
                # nengo.Connection(adapt_signal, minsim, synapse=None)
                conn = nengo.Connection(adapt, minsim, synapse=p.synapse,
                                        function=lambda x: [0]*p.D,
                                        solver=ZeroDecoder(),
                                        learning_rule_type=nengo.PES(
                                           learning_rate=1e-4*p.learning_rate))
                # conn.learning_rule_type.learning_rate *= p.learning_rate
                nengo.Connection(control, conn.learning_rule, synapse=None,
                                 transform=-1)

            signal = ctrl.Signal(
                p.D, p.period, dt=p.dt, max_freq=p.max_freq, seed=p.seed)
            desired = nengo.Node(signal.value, label='desired')
            nengo.Connection(desired, control[p.D:], synapse=None)

            self.p_desired = nengo.Probe(desired, synapse=None)
            # TODO: why doesn't this probe work on nengo_spinnaker?
            # self.p_q = nengo.Probe(state_node, synapse=None)
            self.p_u = nengo.Probe(control, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        # start_time = time.time()
        # while time.time() - start_time < p.T:
        #    sim.run(p.dt, progress_bar=False)
        sim.run(p.T)

        data_p_q = np.array(self.system_state)
        data_p_desired = np.array(self.system_desired)
        t = np.array(self.system_t)

        q = data_p_q[:, 0]
        d = data_p_desired[:, 0]

        N = len(q) / 2

        # find an offset that lines up the data best (this is the delay)
        offsets = []
        for i in range(p.D):
            q = data_p_q[:, i]
            d = data_p_desired[:, i]
            offset = ctn_benchmark.stats.find_offset(q[N:], d[N:])
            if offset == 0:
                offset = 1
            offsets.append(offset)
        offset = int(np.mean(offsets))
        delay = np.mean(t[1:] - t[:-1]) * offset

        if plt is not None:
            plt.plot(t[offset:], d[:-offset], label='$q_d$')
            #plt.plot(t[offset:], d[offset:])
            plt.plot(t[offset:], q[offset:], label='$q$')
            plt.legend(loc='upper left')

            #plt.plot(np.correlate(d, q, 'full')[len(q):])

        diff = data_p_desired[:-offset] - data_p_q[offset:]
        diff = diff[N:]
        rmse = np.sqrt(np.mean(diff.flatten()**2))

        return dict(delay=delay, rmse=rmse)

if __name__ == '__main__':
    AdaptiveBias().run()
