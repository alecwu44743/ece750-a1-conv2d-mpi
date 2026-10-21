# ECE 750 T39 – A1: Distributed 2D Convolution (C++ / Open MPI)

A parallel/distributed implementation of the 2D convolution layer of a CNN
(`torch.nn.Conv2d` semantics, i.e. cross-correlation), written in C++17 with
MPI. It supports, beyond the required single-channel / padding 0 / stride 1 case:
**padding, stride > 1, bias, multiple input and output channels**, and two
local algorithms: **direct sliding window** and **im2col + hand-written GEMM**
(no BLAS / oneMKL, as required).

## Layout

```
conv2d_mpi/
├── include/
│   ├── conv2d.hpp          # API: ConvParams, serial + MPI convolution, filters
│   ├── io.hpp              # PGM image and .npy array I/O
│   └── cli.hpp             # tiny argument parser
├── src/
│   ├── conv2d_serial.cpp   # CNN core: direct conv, im2col, blocked GEMM, padding, filters
│   ├── conv2d_mpi.cpp      # distributed conv: spatial (row-block) and batch decompositions
│   ├── conv2d_dist.cpp     # distributed-input mode: pre-distributed slabs / MPI_Sendrecv halo exchange
│   ├── io.cpp
│   ├── main.cpp            # conv2d_run   – run / visualise / verify one convolution
│   └── benchmark.cpp       # conv2d_bench – measurement: latency, scaling, throughput -> CSV
├── tests/test_conv2d.cpp   # conv2d_test  – unit tests (run under 1..N processes)
├── scripts/
│   ├── run_experiments.sh  # whole evaluation (correctness + 4 performance experiments)
│   ├── plot_results.py     # CSV -> plots + summary.md (speedup, efficiency, Karp-Flatt)
│   ├── verify_reference.py # compare against PyTorch conv2d and SciPy correlate2d
│   └── prepare_image.py    # JPG/PNG -> grayscale PGM (optional resize)
└── Makefile
```

## Build

Requirements: a C++17 compiler and an MPI implementation (tested with Open MPI 5.0.7,
Apple clang 21). Python 3 with `numpy scipy matplotlib` (+ optionally `torch`, `pillow`)
for the verification/plot scripts.

```bash
make            # builds bin/conv2d_run, bin/conv2d_bench, bin/conv2d_test
make test       # unit tests with 1, 2, 3, 4 and 8 processes
```

On the ECE servers (`eceubuntu*`, `ecetesla*`, `ecehadoop`), load/locate Open MPI
(`which mpicxx`) and run `make`; set `MPICXX=/path/to/mpicxx` if needed.

## Running

```bash
# Edge detection (slide 3) on a real image, 4 processes, verified against the serial code
python3 scripts/prepare_image.py photo.jpg data/photo.pgm --resize 4096x4096
mpirun -np 4 ./bin/conv2d_run --image data/photo.pgm --filter edge --out edge.pgm --verify

# Synthetic 8192x8192 image, 7x7 Gaussian, "same" padding
mpirun -np 8 ./bin/conv2d_run --size 8192 --kernel-size 7 --filter gaussian --padding 3 --repeat 5

# CNN layer: 3 -> 16 channels, 5x5, stride 2, padding 2, bias, im2col+GEMM
mpirun -np 4 ./bin/conv2d_run --size 1024 --cin 3 --cout 16 --kernel-size 5 \
       --stride 2 --padding 2 --bias --filter random --algo im2col --verify
```

`conv2d_run --help` / `conv2d_bench --help` list all options.

### Multi-node (ecehadoop: 16 nodes × 4 cores)

```bash
for i in $(seq 0 15); do echo "ecehadoop$i.private.uwaterloo.ca slots=4"; done > hosts.txt
mpirun --hostfile hosts.txt --map-by core --bind-to core \
       --mca plm_rsh_args "-o StrictHostKeyChecking=no" -np 64 ./bin/conv2d_bench --sizes 16384 --kernels 3,7
NP_LIST="1 2 4 8 16 32 64" \
MPIRUN_ARGS="--hostfile hosts.txt --map-by core --bind-to core --mca plm_rsh_args -oStrictHostKeyChecking=no" \
  ./scripts/run_experiments.sh
```

The binaries and data path must be visible on every node (shared home directory).

## Design

### Local kernels (`src/conv2d_serial.cpp`)

* **Direct**: loop order `(co, ci, oy, ky, kx, ox)` keeps one output row in L1 and makes
  the innermost loop a unit-stride AXPY (`out[ox] += w * in[ox]`) that the compiler
  vectorises (NEON / AVX). Complexity `O(Cout·Cin·f²·Ho·Wo)`.
* **im2col + GEMM** (slides 13–14): builds the `(Cin·f²) × (Ho·Wo)` column matrix in
  row chunks (bounded ~16 MB), then `Out[Cout × N] = W[Cout × Cin·f²] · Col`. The GEMM is
  our own cache-blocked kernel (1024-column × 128-depth blocks, 4-way k-unrolling).
  im2col pays off when `Cout` is large (many filters reuse each column); for a single
  filter it is just extra memory traffic, so `direct` is faster there.

### Distribution (`src/conv2d_mpi.cpp`, `src/conv2d_dist.cpp`)

1. **Spatial / row-block decomposition** (one large image — latency & strong scaling).
   Output rows are split into `P` contiguous blocks (`block_range`, sizes differ by ≤ 1).
   Rank `r` needs input rows `[start·s, (start+cnt−1)·s + f)`, i.e. its block plus a halo
   of `f − s` rows. Root pads the image, packs each rank's slab (halo duplicated, so ranks
   never communicate with each other), and `MPI_Scatterv`s it; weights/bias go by
   `MPI_Bcast`; results return by `MPI_Gatherv` (directly into the output when `Cout = 1`).
   Ranks with no rows (P > H_out) are handled.
2. **Batch / data-parallel decomposition** (many images — throughput): whole images are
   scattered with `MPI_Scatterv`, each rank convolves its share, `MPI_Gatherv` collects.

Communication volume per image: `≈ Cin·H·W·(1 + P(f−s)/H)` floats in and `Cout·Ho·Wo`
out, versus `2·Cout·Cin·f²·Ho·Wo` FLOPs — so the computation-to-communication ratio grows
with `f²` and the number of channels. That is why small kernels on one image are
communication-bound, and why the benchmark reports each phase separately.

3. **Distributed-input mode** (`conv2d_bench --input-dist local|halo`, `src/conv2d_dist.cpp`).
   Modes 1 and 2 measure a *single, isolated* convolution whose input starts and ends on
   rank 0, so every call pays `O(H·W)` scatter + gather through one process. That is not
   how a CNN runs: in a network of many layers the activations **stay distributed between
   layers** — the output block a rank produces for layer *l* is exactly the input block it
   owns for layer *l + 1* — and data enters/leaves only once at the start and end of the
   network (and, in practice, is read in parallel from storage). The only communication a
   convolution layer then needs is its halo. This mode models that:
   * `row_layout()` gives every rank a non-overlapping block of padded input rows
     (`own`) and the rows it needs (`need` = own block + `f − s` halo rows below it).
     Each rank generates its own rows (`make_padded_rows`, a deterministic hash, so any
     rank can produce any row without holding the image); weights are resident on all ranks.
   * `--input-dist local` (`--distributed-input`): each rank already holds `need`
     (halo included) — only local compute is timed. This is the upper bound on scalability.
   * `--input-dist halo` (`--halo-exchange`): each rank holds only `own` and fetches
     its halo from the next rank with `MPI_Sendrecv` (a strided `MPI_Type_vector`
     covering all channels). The halo is `Cin·(f−s)·Wp` floats per rank — independent of
     `H` and `P` — so it is the realistic per-layer communication cost. If blocks become
     thinner than the halo, extra rounds reach rank `r + 2, r + 3, …` automatically.
     Its `distribute` time also includes copying the owned rows into the compute slab
     (≈ 0.35 ms for 2048² on the laptop, visible already at P = 1).
   * Outputs stay on their ranks (`collect` = 0); `--verify` gathers them to rank 0
     outside the timed region and compares with the serial result.

## Correctness (20 %)

* `bin/conv2d_test` (165 checks, run with P = 1, 2, 3, 4, 8):
  hand-computed examples from the slides (α on slide 4, padded α on slide 6, stride-2 β on
  slide 10, the 4×4 / 2×2 im2col example on slides 13–14); the output-size formula
  `k = (d − f + 2p)/s + 1`; identity / box / edge kernels with known outputs; blocked GEMM
  vs triple loop; direct and im2col vs an independent 7-loop oracle over 12 random
  configurations (odd sizes, `f` = 1…11, stride 1–3, padding, bias, up to 4→8 channels);
  MPI spatial and batch results vs serial, including `P > H_out`; the distributed-input
  modes (pre-distributed and halo exchange) vs serial for all of the above plus a 6-row halo
  spanning several thin blocks and stride > kernel (no halo); `row_layout` tiles all rows.
* `scripts/verify_reference.py --np 4`: dumps tensors as `.npy` and compares against
  **PyTorch `torch.nn.functional.conv2d`** and **SciPy `signal.correlate2d`**
  (max abs error ≈ 1e-6, float32).
* `conv2d_run --verify` / `conv2d_bench --verify` compare every distributed result with
  the serial reference.

## Performance & scalability evaluation (40 % + 40 %)

`./scripts/run_experiments.sh` then `python3 scripts/plot_results.py results` runs:

| # | Experiment | Configuration (full / `QUICK=1`) | Metric |
|---|---|---|---|
| 1 | Latency vs size | images 512²…8192² (512²…2048²), kernels 3,5,7,9,11 (3,5,7), P = 1 and P_max | median latency, GFLOP/s |
| 2 | Strong scaling | 8192² (2048²) single channel, k = 3, 7; CNN layer 1024² (256²) 16→32 ch, 3×3, p = 1, direct vs im2col; P ∈ `NP_LIST` | speedup, efficiency (total and compute-only), Karp–Flatt serial fraction, load imbalance |
| 2b | Strong scaling: root vs distributed input | same single-channel image, k = 3, 7, with `--input-dist root`, `local`, `halo` (`results/strong_dist.csv`, `strong_dist_scaling.png`) | speedup, efficiency, halo-exchange time vs scatter/gather time |
| 3 | Weak scaling | 1024 (256) rows × 8192 (2048) cols **per process**, k = 3, 7 | efficiency T(1)/T(P) |
| 4 | Batch throughput | CIFAR-10-sized 32×32×3 → 16 ch, 3×3, batch 4096 (512); ImageNet-sized 224×224×3 → 16 ch, 7×7, s = 2, p = 3, batch 256 (32) | images/s, speedup |

Methodology:
* `--warmup 2` untimed iterations, then `--repeat 10` timed; we report the **median**
  (plus min, mean, std-dev in the CSV).
* Every iteration starts after `MPI_Barrier`; each phase is timed with `MPI_Wtime` and the
  **maximum over ranks** is taken (the slowest rank determines latency).
* Phases: `distribute` (bcast + pad/pack + scatter), `compute`, `collect` (gather).
  Compute-only speedup isolates the algorithm's scalability from the root-centred I/O.
* Load imbalance = max/mean of per-rank compute time (1.0 = perfect).
* Inputs are synthetic uniform random tensors (allowed by the assignment) — timing does not
  depend on the values; real images can be used with `conv2d_run --image`.
* Knobs: `NP_LIST`, `MPIRUN_ARGS` (hostfile, binding), `QUICK=1`, `RESULTS=dir`.

### Example results (MacBook, Apple M-series: 5 performance + 6 efficiency cores, `QUICK=1`)

See `results_quick/summary.md` and the plots there. Highlights:

* 2048², 7×7, single channel: compute speedup 3.7× on 4 processes (efficiency 0.94),
  end-to-end 3.1× (efficiency 0.78) — the remainder is scatter/gather on the root.
* Batch, ImageNet-sized 7×7: 270 → 900 images/s from 1 to 4 processes (3.3×).
* Beyond 4 processes this laptop falls onto its slower efficiency cores (load imbalance
  up to 2.3) and shared memory bandwidth, so P = 8 does not help there. Run on the
  homogeneous ECE servers / ecehadoop cluster with `--bind-to core` for the final numbers.

## Datasets

Synthetic data is the default. For real images:
CIFAR-10/100 <https://www.cs.toronto.edu/~kriz/cifar.html>,
ImageNet-mini <https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000>,
high-resolution images (Kaggle / Hugging Face links in the assignment). Convert with
`scripts/prepare_image.py`. Do not submit datasets or `results*/` to LEARN.
