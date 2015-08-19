
import ctn_benchmark

def run_sims(backend, data_dir):
    for i in range(30):
        ctn_benchmark.nengo.SPASequence().run(seed=i, data_dir=data_dir,
                                              backend=backend, debug=True)

def task_nengo():
    return dict(actions=[(run_sims, ['nengo', 'nengo'])])
def task_ocl():
    return dict(actions=[(run_sims, ['nengo_ocl', 'ocl'])])
def task_spinn():
    return dict(actions=[(run_sims, ['nengo_spinnaker', 'spinn'])])
def task_sp_rmv():
    return dict(actions=[(run_sims, ['nengo_spinnaker', 'sp_rmv'])])

def task_plot():
    """plot results"""

    def plot():

        plot = ctn_benchmark.Plot([ctn_benchmark.Data('nengo'),
                     ctn_benchmark.Data('ocl'),
                     ctn_benchmark.Data('spinn'),
                     ctn_benchmark.Data('sp_rmv')])

        plot.measures(['period', 'period_sd', 'peak', 'peak_sd'],
                show_outliers=False)

        import pylab
        pylab.show()

    return dict(actions=[plot])

