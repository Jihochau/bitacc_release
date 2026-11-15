# BitAcc: Cycle-Level Simulator and Evaluation Scripts

Generate, simulate, and verify the command trace of one BitAcc decoding iteration, and reproduce the evaluation of the paper. The default example is
**BitNet-125M**, batch size 1, one decode token, and **visible KV history = 2**.

## Requirements

- Python 3.10 or newer.
- CMake 3.14 or newer, Git, and a C++20 compiler.
- Internet access during the first CMake configure. CMake fetches pinned
  `yaml-cpp 0.7.0`, `spdlog 1.11.0`, and `argparse 2.9` dependencies.

## Build

From this directory:

```bash
cmake -S ramulator2 -B ramulator2/build -DCMAKE_BUILD_TYPE=Release
cmake --build ramulator2/build -j
```

On Windows with MinGW, select the generator explicitly if needed:

```powershell
cmake -S ramulator2 -B ramulator2/build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build ramulator2/build -j
```

## Reproduce the default result

```bash
python generate_trace.py
```

The command generates the trace and verifies it against the bundled reference:

```text
result/bitnet_125m_kv2/trace_out.ch0
```

Use `--keep-intermediates` to retain the Ramulator input, YAML configuration,
and generation summary. If the executable is outside the default build path:

```bash
python generate_trace.py --ramulator-bin /path/to/ramulator2
```

The trace can also be checked separately:

```bash
python verify_trace.py result/bitnet_125m_kv2/trace_out.ch0
```

## Run a different model

First list the model names defined in `config/models.json`:

```bash
python generate_trace.py --list-models
```

Choose a model with `--model` and set the number of cached tokens visible to
the new decode token with `--kv-history`:

```bash
python generate_trace.py --model BITNET-350M --kv-history 128
python generate_trace.py --model BITNET-70B --kv-history 32
```

The model name is case-insensitive, and the `BITNET-` prefix may be omitted.
For example, `--model 70b` also selects `BITNET-70B`.

The commonly used options are:

| Option | Meaning | Default |
| --- | --- | --- |
| `--model NAME` | Model from the selected model registry | `BITNET-125M` |
| `--kv-history N` | Positive number of cached KV tokens | `2` |
| `--output-dir PATH` | Result directory | `result/<model>_kv<N>` |
| `--ramulator-bin PATH` | Ramulator2 executable | Auto-detected from `ramulator2/build` |
| `--timeout SECONDS` | Ramulator2 timeout | `3600` |
| `--keep-intermediates` | Keep generated trace input, YAML, and summary | Disabled |
| `--expected FILE` | Golden manifest used for exact verification | Auto-detected when available |
| `--models-file FILE` | Replace the built-in model registry | `config/models.json` |
| `--config FILE` | Merge runtime/hardware settings over `config/default.json` | None |

For example, select an explicit output directory and allow a longer simulation:

```bash
python generate_trace.py \
  --model BITNET-70B \
  --kv-history 32 \
  --output-dir result/70b_kv32 \
  --timeout 14400
```

Without `--output-dir`, this workload is written to:

```text
result/bitnet_70b_kv32/trace_out.ch0
```

### Add a model configuration

To use a model that is not in `config/models.json`, create a JSON model
registry. `--models-file` replaces the built-in registry, so the file must
contain every model needed by that run:

```json
{
  "BITNET-CUSTOM": {
    "parameter_count": 500000000,
    "n_layers": 24,
    "hidden_size": 1280,
    "num_heads": 20,
    "head_dim": 64,
    "ffn_size": 4096,
    "vocab_size": 32000
  }
}
```

Run it with:

```bash
python generate_trace.py \
  --models-file custom_models.json \
  --model BITNET-CUSTOM \
  --kv-history 128
```

The generator derives the layer addresses, weight/KV row requirements, command
counts, and a DDR5 organization with sufficient capacity from these values.

### Override runtime or hardware configuration

`--config` accepts a JSON object that is merged over `config/default.json`.
For example:

```json
{
  "ramulator": {
    "org": "auto",
    "timing": "DDR5_3200AN"
  }
}
```

```bash
python generate_trace.py \
  --model BITNET-7B \
  --kv-history 256 \
  --config my_config.json
```

Model selection and KV history come from the command-line options. The simulated
design requires batch size 1, one decoded token, one channel/rank, eight x8
chips, 32 banks per chip, a 128-bit Bank PE, and 32 LUT pairs.

### Verify another workload exactly

The bundled golden manifest covers only 125M/KV=2. Other workloads still run
the internal command-count and trace-structure checks. To require an exact
trace match, create a corresponding expected JSON file and pass it explicitly:

```bash
python generate_trace.py \
  --model BITNET-350M \
  --kv-history 128 \
  --expected expected/bitnet_350m_kv128.json
```

## Expected default result

- 10,661 output lines: 9,576 semantic commands and 1,085 DRAM protocol commands.
- Final cycle: 598,463.
- 924 `ALL_BANK_LUT_LOAD` commands.
- 8,316 `ALL_BANK_BITLINEAR` commands.
- 44 `ALL_BANK_RD_K_V_CACHE` commands.
- Normalized SHA-256:
  `cf9834c9800b55dc4f3a1919d6b009b5593fd14a58fe6ddd36dd0096a6d19b1e`.

The verifier normalizes line endings before comparing the required hash, so
Linux and Windows builds are checked against the same trace content.

## Reproduce the evaluation of the paper

Install the Python dependencies of the experiment scripts:

```bash
python -m pip install -r requirements.txt
```

Run all experiments. The launcher runs `--jobs` simulations at a time and skips
the cases that are already complete, so it can be interrupted and restarted:

```bash
python scripts/run_sweep.py all --jobs 8
python scripts/run_sweep.py all --status
```

Parse the results, compare them with the values reported in the paper
(`expected/paper/`), and draw the figures:

```bash
python scripts/parse_results.py --check
python scripts/plot_all_figures.py
```

`--check` requires cycle counts to be equal and derived values to agree to a
relative 1e-9. It reports cases that have not been run separately and exits
with a non-zero status on any mismatch. CSV files are written to
`results/csv/` and figures to `figures/`.

| Experiment | Cases | Output in `results/csv/` | Paper |
| --- | --- | --- | --- |
| `context` | 12 models x 7 KV histories | `context_sensitivity.csv`, `end_to_end_throughput.csv` | Decoding throughput, context length |
| `locality` | base cases at KV 128, 512, 1024 | `locality_breakdown.csv`, `time_breakdown.csv` | Operand locality, decoding-time breakdown |
| `dimm` | 4 variants x 12 models x KV 128, 1024 | `dimm_sensitivity.csv` | Sensitivity to the DIMM-level PE |
| `scaling` | 12 models x 4 chip counts x 4 PE counts | `scaling.csv` | Scaling with chips and bank-level PEs |
| `energy` | no additional simulation | `energy_per_token.csv`, `energy_sensitivity.csv` | Energy per token |

A subset is selected by experiment, model, and KV history, for example
`python scripts/run_sweep.py context --models 7B --kv 128`.

Notes on the methodology:

- The decoding latency of a case is its final cycle multiplied by the 0.625-ns
  period of the memory controller clock.
- `scaling` evaluates the command plan from which the trace is generated,
  without the DRAM command timing that the simulator adds, because the
  simulated all-bank commands always address every bank of every chip.
- `energy` is an analytical model with assumed host powers
  (`config/energy.json`), not a measurement.
- The CPU baselines are published numbers; `baselines/README.md` gives the
  table and row of every value.

## Main files

- `scripts/`: experiment launcher, result parser, and plotting script.
- `baselines/`, `expected/paper/`: published baselines and reference results.
- `generate_trace.py`: trace-generation entry point.
- `verify_trace.py` and `expected/`: deterministic reference checks.
- `src/`: workload configuration, command planning, and timing model.
- `config/`: model and hardware configuration.
- `ramulator2/`: BitAcc-enabled Ramulator2 implementation.

## License and attribution

BitAcc is released under the MIT license (`LICENSE`).

`ramulator2/` is a modified copy of Ramulator 2.0 (Luo et al., IEEE CAL 2023,
github.com/CMU-SAFARI/ramulator2), which retains its upstream MIT license in
`ramulator2/LICENSE`. Its processing-in-memory extensions (the PIM memory
system, controller, scheduler, and trace frontend) are adapted from the AttAcc
simulator (Park et al., ASPLOS 2024, github.com/scale-snu/attacc_simulator),
whose MIT license is retained in `ramulator2/LICENSE.AttAcc`.
