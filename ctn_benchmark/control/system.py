import numpy as np

class System(object):
    def __init__(self, d_state, d_motor, dt=0.001, seed=None,
            scale_mult=1, scale_add=1, diagonal=True,
            sense_noise=0.1, motor_noise=0.1,
            motor_delay=0, motor_filter=None,
            scale_inertia=0, motor_scale=1.0,
            sensor_delay=0, sensor_filter=None,
            nonlinear=True):

        self.rng = np.random.RandomState(seed=seed)

        self.d_motor = d_motor
        self.d_state = d_state
        self.dt = dt
        self.motor_scale = motor_scale

        sensor_steps = int(sensor_delay / dt) + 1
        self.sensor_delay = np.zeros((sensor_steps, d_state), dtype=float)
        motor_steps = int(motor_delay / dt) + 1
        self.motor_delay = np.zeros((motor_steps, d_motor), dtype=float)
        self.sensor_index = 0
        self.motor_index = 0
        self.scale_inertia = scale_inertia

        self.sensor = np.zeros(d_state, dtype=float)
        self.motor = np.zeros(d_motor, dtype=float)
        if sensor_filter is None or sensor_filter < dt:
            self.sensor_filter_scale = 0.0
        else:
            self.sensor_filter_scale = np.exp(-dt / sensor_filter)
        if motor_filter is None or motor_filter < dt:
            self.motor_filter_scale = 0.0
        else:
            self.motor_filter_scale = np.exp(-dt / motor_filter)

        if diagonal:
            assert d_state == d_motor
            self.J = np.eye(d_motor) * scale_mult
            #self.J = np.abs(np.diag(self.rng.randn(d_motor))) * scale_mult
        else:
            self.J = self.rng.randn(d_motor, d_state) * scale_mult
        self.sense_noise = sense_noise
        self.motor_noise = motor_noise

        self.nonlinear = nonlinear
        self.additive_a = self.rng.randn(d_state) 
        if nonlinear:
            D = len(self.nonlinearity(self.sensor))
            self.additive_b = self.rng.randn(d_state, D)
            self.additive_c = self.rng.randn(d_state) 
            self.additive_d = self.rng.randn(d_state)
        self.scale_add = scale_add

        self.reset()
    def reset(self):
        self.state = self.rng.randn(self.d_state)
        self.dstate = np.zeros_like(self.state)
        self.sensor_delay *= 0
        self.motor_delay *= 0

    def nonlinearity(self, q):
        return np.hstack([q, np.sin(q)])#, q**2, q**3])


    def step(self, motor):
        motor = np.tanh(motor) * self.motor_scale
        self.motor_delay[self.motor_index] = motor
        self.motor_index = (self.motor_index + 1) % len(self.motor_delay)
        motor = self.motor_delay[self.motor_index]

        motor = motor + self.rng.randn(self.d_motor) * self.motor_noise
        self.motor = (self.motor * self.motor_filter_scale +
                      motor * (1.0 - self.motor_filter_scale))
        self.dstate *= self.scale_inertia

        if self.nonlinear:
            q2 = self.nonlinearity(self.additive_d * self.state + 
                            self.additive_c)
            additive = np.dot(self.additive_b, q2) + self.additive_a
            additive *= self.scale_add
        else:
            additive = self.additive_a * self.scale_add
        self.additive = additive
        self.dstate += (np.dot(self.motor, self.J) + additive)
        self.state = self.state + self.dstate * self.dt

        sensor = self.state + self.rng.randn(self.d_state) * self.sense_noise
        self.sensor = (self.sensor * self.sensor_filter_scale +
                       sensor * (1.0 - self.sensor_filter_scale))

        self.sensor_delay[self.sensor_index] = self.sensor
        self.sensor_index = (self.sensor_index + 1) % len(self.sensor_delay)
        return self.sensor_delay[self.sensor_index]


if __name__ == '__main__':
    D = 1

    data = []
    for i in range(10000):
        system = System(D, D)
        system.step(np.zeros(D))
        data.append(system.additive)

    data = np.array(data).flatten()
    data = np.sort(data)
    index = int(len(data)*0.025)
    print('95%% range: %f to %f' % (data[index], data[-index]))
    
