#!/usr/bin/env bash
# Runs the full evaluation: correctness + latency + strong/weak scaling + batch throughput.
# Every experiment launches a separate mpirun per process count P and appends a
# CSV row per configuration to results/*.csv; plot with scripts/plot_results.py.
#
# Environment knobs:
#   NP_LIST="1 2 4 8"        process counts to sweep (default: 1 2 4 8)
#   MPIRUN=mpirun            launcher
#   MPIRUN_ARGS="..."        extra launcher args, e.g. on the ecehadoop cluster:
#                            MPIRUN_ARGS="--hostfile hosts.txt --map-by node"
#   QUICK=1                  smaller problem sizes (fast smoke run, ~1 min)
#   RESULTS=results          output directory
#
#   ./scripts/run_experiments.sh
#   NP_LIST="1 2 4 8 16 32 64" MPIRUN_ARGS="--hostfile hosts.txt" ./scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NP_LIST=${NP_LIST:-"1 2 4 8"}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_ARGS=${MPIRUN_ARGS:-"--oversubscribe"}
RESULTS=${RESULTS:-results}
QUICK=${QUICK:-0}
BENCH=./bin/conv2d_bench
MAX_NP=$(echo "$NP_LIST" | tr ' ' '\n' | sort -n | tail -1)

if [[ "$QUICK" == 1 ]]; then
  LAT_SIZES=512,1024,2048;      LAT_KERNELS=3,5,7
  STRONG_SIZE=2048;             WEAK_ROWS=256; WEAK_W=2048
  CNN_SIZE=256;                 BATCH_CIFAR=512; BATCH_IMNET=32; REP=5
else
  LAT_SIZES=512,1024,2048,4096,8192; LAT_KERNELS=3,5,7,9,11
  STRONG_SIZE=8192;             WEAK_ROWS=1024; WEAK_W=8192
  CNN_SIZE=1024;                BATCH_CIFAR=4096; BATCH_IMNET=256; REP=10
fi

make -s all
mkdir -p "$RESULTS"
rm -f "$RESULTS"/{latency,strong,strong_dist,weak,batch}.csv

run() {  # run <np> <args...>
  local np=$1; shift
  echo "+ $MPIRUN $MPIRUN_ARGS -np $np $BENCH $*"
  $MPIRUN $MPIRUN_ARGS -np "$np" $BENCH "$@"
}

echo "=== [0] correctness: unit tests + reference comparison ==="
for np in 1 3 $MAX_NP; do $MPIRUN $MPIRUN_ARGS -np "$np" ./bin/conv2d_test; done
if python3 -c "import numpy, scipy" 2>/dev/null; then
  python3 scripts/verify_reference.py --np "$MAX_NP" --mpirun "$MPIRUN" --mpirun-args="$MPIRUN_ARGS"
fi

echo "=== [1] latency vs image size and kernel size (P=$MAX_NP and P=1) ==="
for np in 1 "$MAX_NP"; do
  run "$np" --mode spatial --sizes "$LAT_SIZES" --kernels "$LAT_KERNELS" --algos direct \
      --repeat "$REP" --csv "$RESULTS/latency.csv" --label latency
done

echo "=== [2] strong scaling (fixed problem, vary P) ==="
for np in $NP_LIST; do
  # single-channel image (the assignment's minimum case)
  run "$np" --mode spatial --sizes "$STRONG_SIZE" --kernels 3,7 --algos direct \
      --repeat "$REP" --csv "$RESULTS/strong.csv" --label strong-1ch --verify
  # CNN-like layer: 16 -> 32 channels, 3x3, same padding (higher arithmetic intensity)
  run "$np" --mode spatial --sizes "$CNN_SIZE" --kernels 3 --cin 16 --cout 32 --padding 1 \
      --algos direct,im2col --repeat "$REP" --csv "$RESULTS/strong.csv" --label strong-cnn
done

echo "=== [2b] strong scaling: root scatter/gather vs distributed input (CNN layer-to-layer) ==="
for np in $NP_LIST; do
  for dist in root local halo; do
    run "$np" --mode spatial --input-dist "$dist" --sizes "$STRONG_SIZE" --kernels 3,7 --algos direct \
        --repeat "$REP" --csv "$RESULTS/strong_dist.csv" --label "strong-dist" --verify
  done
done

echo "=== [3] weak scaling (work per process fixed: ${WEAK_ROWS} rows x ${WEAK_W} per rank) ==="
for np in $NP_LIST; do
  run "$np" --mode spatial --sizes "$((WEAK_ROWS * np))x${WEAK_W}" --kernels 3,7 --algos direct \
      --repeat "$REP" --csv "$RESULTS/weak.csv" --label weak
done

echo "=== [4] batch throughput (data-parallel over images) ==="
for np in $NP_LIST; do
  # CIFAR-10 sized images: 32x32x3 -> 16 channels, 3x3, same padding
  run "$np" --mode batch --batch "$BATCH_CIFAR" --sizes 32 --kernels 3 --cin 3 --cout 16 --padding 1 \
      --algos im2col --repeat "$REP" --csv "$RESULTS/batch.csv" --label batch-cifar
  # ImageNet sized images: 224x224x3 -> 16 channels, 7x7, stride 2, padding 3 (ResNet stem-like)
  run "$np" --mode batch --batch "$BATCH_IMNET" --sizes 224 --kernels 7 --cin 3 --cout 16 --stride 2 \
      --padding 3 --algos im2col --repeat "$REP" --csv "$RESULTS/batch.csv" --label batch-imagenet
done

echo "Done. Results in $RESULTS/. Plot with: python3 scripts/plot_results.py $RESULTS"
