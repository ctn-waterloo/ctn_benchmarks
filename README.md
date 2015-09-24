# ctn_benchmarks
Benchmarks and benchmarking software for Nengo

This repository collects together benchmarks and tools for creating benchmarks,
all of which are meant for evaluating [Nengo](http://nengo.ca) models.

### Code organization

 * `ctn_benchmark/*`: basic classes
 * `ctn_benchmark/nengo`: a set of test models meant to cover a variety of neural models
 * `ctn_benchmark/control`: benchmarks for adaptive control of embodied systems

### Basic usage

All benchmarks are meant to be usable in three different ways:

 * From the command line
 * From within nengo_gui
 * From a separate script

For example, for the Lorenz attractor model, you can run it from the command
line as `python ctn_benchmark/nengo/lorenz.py`.  When run in this way,
you can use command line arguments to adjust the behaviour of the benchmark.
Some of these parameters adjust features of how the model is run, such
as whether to display a figure plotting model data after the run (`--show_figs`)
or what nengo backend to use (`--backend nengo_ocl` or
`--backend nengo_spinnaker`, for example).  These parameters are available
for all benchmarks using this system.  Other parameters are specific to the
individual model being run.  For example, in the Lorenz model you can set
any of the Lorenz variables (`sigma`, `beta`, and `rho`), or the number of
neurons (`N`).  For a full list of all parameters, do `--help`.

To use a benchmark with nengo_gui, create an instance of the class and call
`make_model()` on it.  For the Lorenz example, this would be
`ctn_benchmark.nengo.Lorenz().make_model()`.  To specify parameters, pass
them into the `make_model()` call.  For example,
`ctn_benchmark.nengo.Lorenz().make_model(N=1000)`.

Finally, you can run the benchmark using other scripts.  To do this, call
`ctn_benchmark.nengo.Lorenz().run()`.  This will construct the model, run
it, and perform any data analysis defined in the benchmark.  You can also
pass parameters in to the `run()` function.

### Data analysis

Each time the benchmark is run, data will be stored in the data directory
(default: `data`).  Each run is stored in a separate file, with a file name
based on the time it was run.  These data files store the processed results
in text format.

The `Data` class can be used to read values from these data files.  For example,
`ctn_benchmark.Data(path='data').get('rmse')` will return the RMSE value for
all the benchmark runs in the `data` directory.




