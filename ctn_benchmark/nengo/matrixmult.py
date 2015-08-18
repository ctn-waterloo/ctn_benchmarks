"""
Nengo Benchmark Model: Matrix Multiplication

Input: two random matrices of size D1xD2 and D2xD3
Output: a D1xD3 matrix that is the product of the two inputs

"""

import ctn_benchmark
import nengo
import numpy as np

class MatrixMultiply(ctn_benchmark.Benchmark):
    def params(self):
        self.default('size of matrices', D1=1)
        self.default('size of matrices', D2=2)
        self.default('size of matrices', D3=2)
        self.default('range of values', radius=1)
        self.default('number of neurons for input&output', N=50)
        self.default('number of neurons for pairwise multiply', N_mult=200)
        self.default('post-synaptic time constant', pstc=0.01)
        self.default('time to run simulation', T=0.5)

    def model(self, p):
        model = nengo.Network()
        inputA = np.random.uniform(-p.radius, p.radius, p.D1*p.D2)
        inputB = np.random.uniform(-p.radius, p.radius, p.D2*p.D3)
        answer = np.dot(inputA.reshape(p.D1, p.D2),
                        inputB.reshape(p.D2, p.D3)).flatten()

        with model:
            inA = nengo.Node(inputA, label='inA')
            inB = nengo.Node(inputB, label='inB')
            ideal = nengo.Node(answer, label='ideal')

            A = nengo.networks.EnsembleArray(p.N, n_ensembles=p.D1*p.D2,
                                             radius=p.radius, label='A')
            B = nengo.networks.EnsembleArray(p.N, n_ensembles=p.D2*p.D3,
                                             radius=p.radius, label='B')
            D = nengo.networks.EnsembleArray(p.N, n_ensembles=p.D1*p.D3,
                                             radius=p.radius, label='D')

            encoders = nengo.dists.Choice([[1,1],[1,-1],[-1,1],[-1,-1]])

            # the C matrix holds the intermediate product calculations
            #  need to compute D1*D2*D3 products to multiply 2 matrices together
            C = nengo.networks.EnsembleArray(p.N_mult,
                    n_ensembles=p.D1*p.D2*p.D3,
                    label='C',
                    radius=1.5*p.radius, ens_dimensions=2, encoders=encoders)

            nengo.Connection(inA, A.input, synapse=p.pstc)
            nengo.Connection(inB, B.input, synapse=p.pstc)

            # determine the transformation matrices to get the correct pairwise
            # products computed.  This looks a bit like black magic but if
            # you manually try multiplying two matrices together, you can see
            # the underlying pattern.  Basically, we need to build up D1*D2*D3
            # pairs of numbers in C to compute the product of.  If i,j,k are the
            # indexes into the D1*D2*D3 products, we want to compute the product
            # of element (i,j) in A with the element (j,k) in B.  The index in
            # A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
            # The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
            # two values per ensemble.  We add 1 to the B index so it goes into
            # the second value in the ensemble.
            transformA = [[0]*(p.D1*p.D2) for i in range(p.D1*p.D2*p.D3*2)]
            transformB = [[0]*(p.D2*p.D3) for i in range(p.D1*p.D2*p.D3*2)]
            for i in range(p.D1):
                for j in range(p.D2):
                    for k in range(p.D3):
                        transformA[(j + k*p.D2 + i*p.D2*p.D3)*2][j + i*p.D2] = 1
                        transformB[(j + k*p.D2 + i*p.D2*p.D3)*2 + 1][k + j*p.D3] = 1

            nengo.Connection(A.output, C.input, transform=transformA,
                        synapse=p.pstc)
            nengo.Connection(B.output, C.input, transform=transformB,
                        synapse=p.pstc)


            # now compute the products and do the appropriate summing
            def product(x):
                return x[0]*x[1]

            C.add_output('product', product)

            # the mapping for this transformation is much easier,
            # since we want to
            # combine D2 pairs of elements (we sum D2 products together)
            nengo.Connection(C.product,
                             D.input[[i/p.D2 for i in range(p.D1*p.D2*p.D3)]],
                             synapse=p.pstc)

            self.pA = nengo.Probe(A.output, synapse=p.pstc)
            self.pB = nengo.Probe(B.output, synapse=p.pstc)
            self.pD = nengo.Probe(D.output, synapse=p.pstc)
            self.pIdeal = nengo.Probe(ideal, synapse=None)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        self.record_speed(p.T)

        ideal = sim.data[self.pIdeal]
        for i in range(4):
            ideal = nengo.synapses.filt(ideal, nengo.Lowpass(p.pstc), p.dt)

        if plt is not None:
            plt.subplot(1,3,1)
            plt.plot(sim.trange(), sim.data[self.pA])
            plt.ylim(-p.radius, p.radius)
            plt.subplot(1,3,2)
            plt.plot(sim.trange(), sim.data[self.pB])
            plt.ylim(-p.radius, p.radius)
            plt.subplot(1,3,3)
            plt.plot(sim.trange(), sim.data[self.pD])
            plt.plot(sim.trange(), ideal)
            plt.ylim(-p.radius, p.radius)

        rmse = np.sqrt(np.mean(sim.data[self.pD] - ideal)**2)
        return dict(rmse=rmse)


if __name__ == '__main__':
    MatrixMultiply().run()
