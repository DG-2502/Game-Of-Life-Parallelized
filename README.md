# How to build

* Build the project using `cmake`, or manually.
```cmd
gcc gol_serial.c -o gol_serial -lraylib -lm
gcc gol_parallel.c -o gol_parallel -lraylib -lm -fopenmp
```

# Usage:

```cmd
./gol_serial [--bench] [cols rows gens] [--repeat N]
./gol_parallel [--bench] [cols rows [gens]] [--threads N] [--repeat N]
```

* `--bench` will only run the simulation, without visual mode. `cols`, `rows` are the dimensions if the grid, `gens` is the amount of steps of the simulation
* `--repeat` is only used with `--bench`, it runs the benchmarks for the specified amount `N`, and calculates average gen/sec, min and max values, as well as the standard deviation
* `--threads` is used to run the parallel program with `N` threads

**Note that env variables in `config.bat` are overwritten by the command line arguments if specified**