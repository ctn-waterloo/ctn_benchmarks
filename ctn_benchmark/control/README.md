# ctn_benchmarks: control

This directory contains code for benchmarks involving adaptive control.

To run the benchmark, run `python adaptive_bias.py`

Command-line arguments can be used to adjust the benchmark.  Here are some
basic examples:

 * Show a plot of control behaviour: `python adaptive_bias.py --show_figs`
 * Turn on adaptation: `python adaptive_bias.py --show_figs --adapt`
 * Run simulation for longer: `python adaptive_bias.py --show_figs --adapt --T 30`

For a full list of command line arguments, do `python adaptive_bias.py --help`
