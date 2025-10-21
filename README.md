# Damnati
<img width="256" alt="Warm Flame Against Red Backdrop" src="https://github.com/user-attachments/assets/a9f9aa3a-90f0-431e-8727-53fb3ab3277f" />

Damnati is a CUDA-accelerated tournament simulator for the Iterated Prisoner's Dilemma (IPD). It pits a population of strategies against each other, including classical IPD strategies and optional N-gram learning agents.

## Features

- Runs all pairwise matches between agents on the GPU.
- Supports classic strategies such as **Tit for Tat**, **Grim**, **Pavlov**, and others.
- Optional N-gram learners with adjustable depth, exploration rate, and forgiveness parameters.
- Reports aggregate statistics and per-strategy performance after the tournament.

## Building

The program requires the NVIDIA CUDA toolkit. Build the executable with `nvcc`:

```bash
nvcc -O3 -arch=sm_86 damnati.cu -o damnati
```

## Usage

Run the simulator and adjust parameters using command line flags:

```bash
./damnati --agents 512 --rounds 200 --seed 42 --p-ngram 0.6 --depth 3 --epsilon 0.1 --gtft 0.1
```

### Options

- `--agents N`   – number of agents in the population (default: 256)
- `--rounds R`   – number of rounds per match (default: 200)
- `--seed S`     – RNG seed (default: 1234567)
- `--p-ngram F`  – fraction of agents using the N-gram strategy (default: 0.5)
- `--depth D`    – N-gram history depth (default: 3, maximum: 15)
- `--epsilon E`  – exploration rate for N-gram learners (default: 0.1)
- `--gtft P`     – forgiveness probability for Generous Tit for Tat (default: 0.1)

Depths above 15 are rejected to keep GPU memory allocations within practical
limits.

Passing an unknown option will print the usage information and exit with a
non-zero status, so double-check flag names.


## Output

The simulator prints summary statistics in JSON format, including average score, minimum and maximum scores, standard deviation, and per-strategy results. The top few agents and their scores are also listed.

## Testing

Unit tests use the [Catch2](https://github.com/catchorg/Catch2) framework and can be
compiled with `nvcc`. From the repository root run:

```bash
nvcc -std=c++17 tests/test_core.cu -o tests/test_core
nvcc -std=c++17 tests/test_cli.cu -o tests/test_cli
```

Execute each binary to run the associated suite:

```bash
./tests/test_core
./tests/test_cli
```

`test_core` exercises the strategy decision logic in `choose_action`, the payoff
accumulation performed by `ngram_update`, and the 64-bit integer square root helper.
`test_cli` verifies that command-line validation reports errors for malformed depth
values and unknown options.
