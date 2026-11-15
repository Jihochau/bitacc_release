# CPU baselines

The CPU numbers BitAcc is compared against are measured throughputs that were published by
other people. We did not measure or change them. Each row carries its origin in the `source`
column, and `scripts/experiments/context.py` reads these files directly.

## `bitnet_cpp_m2_ultra.csv`

BitNet.cpp token-generation throughput (tokens/s) on an Apple M2 Ultra (Mac Studio, 64 GB) for
the twelve BitNet b1.58 model configurations, 125M to 100B. Source:

> J. Wang et al., "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs,"
> arXiv:2410.16144, 2024.

| `threads`   | Table   | Row in the table            | Setting in the paper           |
|-------------|---------|-----------------------------|--------------------------------|
| `unlimited` | Table 4 | APPLE M2, kernel bitnet.cpp | unlimited number of threads    |
| `2`         | Table 5 | APPLE M2, kernel bitnet.cpp | inference limited to 2 threads |

The tables label the platform "APPLE M2". Section 3.1 of the paper says it is a Mac Studio with
an Apple M2 Ultra processor. The `unlimited` rows are the baseline of the context-length
sensitivity (`results/csv/context_sensitivity.csv`). The `2` rows are not used by any
experiment and are included only for completeness.

## `kernels_m2_ultra.csv`

Token-generation throughput (tokens/s) of the inference kernels on the Apple M2 Ultra, in the
unlimited-thread setting. It covers the eight configurations from 700M to 100B. Source:

> J. Wang et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs," ACL 2025,
> doi:10.18653/v1/2025.acl-long.457, Table 7 (row group "APPLE M2").

| `kernel`  | Column in Table 7            |
|-----------|------------------------------|
| `float16` | llama.cpp Float16, b(16)     |
| `q4_0`    | llama.cpp Q4_0, b(4.5)       |
| `t_mac`   | T-MAC, b(2)                  |
| `tq1_0`   | llama.cpp TQ1_0, b(1.69)     |
| `tq2_0`   | llama.cpp TQ2_0, b(2.06)     |
| `tl1_0`   | Bitnet.cpp TL1_0, b(2)       |
| `tl2_0`   | Bitnet.cpp TL2_0, b(1.67)    |
| `i2_s`    | Bitnet.cpp I2_S, b(2)        |

Table 7 reports Float16 for 100B as "N/A" (the CPU cannot host the model with that kernel),
so the file has no row for it. The table states that each number is the average of 10 tests.
The end-to-end comparison (`results/csv/end_to_end_throughput.csv`) uses `float16`, `t_mac`,
and `i2_s`, and it normalizes to `i2_s`. The other kernels are included only for completeness.

## `bitnet_cpp_energy_m2_ultra.csv`

BitNet.cpp energy per generated token (J/token) on the Apple M2 Ultra. The source reports it
only for the 700M, 7B, and 70B configurations:

> J. Wang et al., arXiv:2410.16144 (as above), Table 6, "Comparison of Energy Costs Across CPUs
> (Unit: J/Token)", row group "APPLE M2", kernel bitnet.cpp.

Section 3.1 of the paper says the energy is that of the run "with the best inference speed in
the unlimited thread setting"; it does not say how the energy was measured or what it includes.
`scripts/experiments/energy.py` reads this file and compares it with the modeled energy per
token of the BitAcc system (`results/csv/energy_per_token.csv`, `energy_sensitivity.csv`).

## Comparison point

The tables do not give the benchmark command. The public BitNet.cpp benchmark script
(`utils/e2e_benchmark.py -m MODEL -p 512 -n 128 -t THREADS`) reports `tg128`. Our reading of
the `llama-bench` version that the script pins is that `tg128` times 128 sequential
single-token decodes that start from an empty KV cache. BitAcc is compared at one fixed visible KV history against the same published number:
128 tokens for the end-to-end comparison, and 32 to 2048 tokens for the context-length
sensitivity.
