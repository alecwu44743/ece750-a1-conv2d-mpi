// benchmark.cpp - conv2d_bench: performance / scalability measurement.
//
// For every (algorithm, image size, kernel size) configuration it runs
// `--warmup` untimed + `--repeat` timed iterations of the distributed
// convolution and records, per iteration, the MAX over ranks of each phase
// (the slowest rank determines the latency).  Statistics are appended as one
// CSV row per configuration so that scripts/run_experiments.sh can sweep the
// number of processes P and scripts/plot_results.py can derive:
//   * latency vs image / kernel size
//   * throughput (images/s) for a batch of images
//   * strong scaling: speedup S(P) = T(1)/T(P), efficiency E(P) = S(P)/P
//   * weak scaling:   efficiency E(P) = T(1)/T(P) with work per rank fixed
//
// --input-dist selects where the input lives in spatial mode:
//   root   (default) root scatters slabs and gathers the output (end-to-end)
//   local  (--distributed-input) every rank already holds its slab incl. halo;
//          only compute is timed - the layer-to-layer situation inside a CNN
//   halo   (--halo-exchange) every rank holds only its own rows and fetches
//          the (f - s) halo rows from its neighbour with MPI_Sendrecv
//
//   mpirun -np 4 ./bin/conv2d_bench --mode spatial --sizes 1024,2048,4096 --kernels 3,5,7 \
//          --csv results/latency.csv --label latency
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "cli.hpp"
#include "conv2d.hpp"

namespace {

struct Stats {
    double median = 0, min = 0, mean = 0, stddev = 0;
};

Stats stats_of(std::vector<double> v) {
    Stats s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.min = v.front();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double acc = 0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stddev = n > 1 ? std::sqrt(acc / (n - 1)) : 0.0;
    return s;
}

double median_of(std::vector<double> v) { return stats_of(std::move(v)).median; }

const char* kCsvHeader =
    "label,mode,algo,P,H,W,cin,cout,kernel,stride,padding,batch,reps,"
    "t_total_med,t_total_min,t_total_mean,t_total_std,"
    "t_distribute_med,t_compute_med,t_collect_med,load_imbalance,gflops,images_per_s,verified\n";

void usage() {
    std::printf(
        "Usage: mpirun -np P conv2d_bench [options]\n"
        "  --mode spatial|batch     row-block split of one image / split a batch of images\n"
        "  --input-dist root|local|halo  spatial mode: root scatter/gather (default),\n"
        "                           pre-distributed slabs, or neighbour halo exchange\n"
        "  --distributed-input      same as --input-dist local\n"
        "  --halo-exchange          same as --input-dist halo\n"
        "  --sizes LIST             image sizes, e.g. 512,1024,2048 or 4096x2048 (default 2048)\n"
        "  --kernels LIST           kernel sizes, e.g. 3,5,7                    (default 3)\n"
        "  --algos LIST             direct,im2col                               (default direct)\n"
        "  --cin N --cout N         channels                                    (default 1/1)\n"
        "  --stride s --padding p   (default 1/0)\n"
        "  --batch N                images per batch in batch mode              (default 64)\n"
        "  --warmup N --repeat N    iterations                                  (default 2/10)\n"
        "  --verify                 check each config against the serial reference\n"
        "  --csv FILE               append results to FILE (header written if new)\n"
        "  --label TEXT             tag stored in the CSV (e.g. strong, weak, latency)\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    try {
        cli::Args args(argc, argv);
        if (args.has("help")) {
            if (rank == 0) usage();
            MPI_Finalize();
            return 0;
        }
        const std::string mode = args.str("mode", "spatial");
        if (mode != "spatial" && mode != "batch") throw std::invalid_argument("--mode must be spatial|batch");
        const bool batch_mode = mode == "batch";
        std::string input_dist = args.str("input-dist", "root");
        if (args.has("distributed-input")) input_dist = "local";
        if (args.has("halo-exchange")) input_dist = "halo";
        if (input_dist != "root" && input_dist != "local" && input_dist != "halo")
            throw std::invalid_argument("--input-dist must be root|local|halo");
        if (batch_mode && input_dist != "root")
            throw std::invalid_argument("--input-dist local/halo only applies to --mode spatial");
        const bool dist_input = input_dist != "root";
        const std::string mode_tag = dist_input ? mode + "-" + input_dist : mode;
        const int batch = batch_mode ? std::max(1, args.num("batch", 64)) : 1;
        const int warmup = std::max(0, args.num("warmup", 2));
        const int repeat = std::max(1, args.num("repeat", 10));
        const bool verify = args.has("verify");
        const std::string csv = args.str("csv", "");
        const std::string label = args.str("label", "run");

        std::vector<conv::Algo> algos;
        for (const auto& a : cli::split(args.str("algos", "direct"))) algos.push_back(conv::parse_algo(a));
        const std::vector<std::string> sizes = cli::split(args.str("sizes", "2048"));
        const std::vector<int> kernels = cli::split_ints(args.str("kernels", "3"));

        FILE* out = nullptr;
        if (rank == 0) {
            std::printf("conv2d_bench: mode=%s P=%d warmup=%d repeat=%d%s\n", mode_tag.c_str(), P, warmup,
                        repeat, batch_mode ? (" batch=" + std::to_string(batch)).c_str() : "");
            std::printf("%-7s %11s %3s %10s %10s %10s %10s %7s %9s %10s %s\n", "algo", "size", "k",
                        "total(ms)", "dist(ms)", "comp(ms)", "coll(ms)", "imbal", "GFLOP/s", "img/s",
                        "check");
            if (!csv.empty()) {
                const bool fresh = !std::ifstream(csv).good();
                out = std::fopen(csv.c_str(), "a");
                if (!out) throw std::runtime_error("cannot open " + csv);
                if (fresh) std::fputs(kCsvHeader, out);
            }
        }

        for (conv::Algo algo : algos)
            for (const std::string& size : sizes)
                for (int k : kernels) {
                    conv::ConvParams p;
                    cli::parse_size(size, &p.height, &p.width);
                    p.kernel = k;
                    p.in_channels = args.num("cin", 1);
                    p.out_channels = args.num("cout", 1);
                    p.stride = args.num("stride", 1);
                    p.padding = args.num("padding", 0);
                    std::string why;
                    if (!p.valid(&why)) {
                        if (rank == 0) std::printf("skip %s k=%d: %s\n", size.c_str(), k, why.c_str());
                        continue;
                    }

                    // Synthetic data (allowed by the assignment); values do not affect timing.
                    std::vector<float> input, weight, output, local_in, local_out;
                    const conv::RowLayout L = conv::row_layout(p, P, rank);
                    if (dist_input) {
                        // Every rank generates its own rows; weights are resident on all ranks.
                        weight.resize(p.weight_size());
                        conv::fill_random(weight, 99);
                        const int rows = input_dist == "halo" ? L.own_rows() : L.need_rows();
                        local_in.resize(static_cast<size_t>(p.in_channels) * rows * p.padded_w());
                        conv::make_padded_rows(p, 1234, input_dist == "halo" ? L.own_start : L.need_start,
                                               rows, local_in.data());
                        local_out.resize(static_cast<size_t>(p.out_channels) * L.out_cnt * p.out_w());
                    } else if (rank == 0) {
                        input.resize(p.input_size() * batch);
                        weight.resize(p.weight_size());
                        output.resize(p.output_size() * batch);
                        conv::fill_random(input, 1234);
                        conv::fill_random(weight, 99);
                    }

                    auto run_once = [&]() {
                        if (input_dist == "local")
                            return conv::conv2d_mpi_local(p, local_in.data(), weight.data(), nullptr,
                                                          local_out.data(), algo, MPI_COMM_WORLD);
                        if (input_dist == "halo")
                            return conv::conv2d_mpi_halo(p, local_in.data(), weight.data(), nullptr,
                                                         local_out.data(), algo, MPI_COMM_WORLD);
                        return batch_mode
                                   ? conv::conv2d_mpi_batch(p, batch, input.data(), weight.data(), nullptr,
                                                            output.data(), algo, MPI_COMM_WORLD)
                                   : conv::conv2d_mpi(p, input.data(), weight.data(), nullptr,
                                                      output.data(), algo, MPI_COMM_WORLD);
                    };

                    for (int i = 0; i < warmup; ++i) run_once();

                    std::vector<double> tot, dist, comp, coll, imb;
                    for (int i = 0; i < repeat; ++i) {
                        const conv::Timing t = run_once();
                        double local[4] = {t.distribute, t.compute, t.collect, t.total}, mx[4];
                        double comp_sum = 0;
                        MPI_Reduce(local, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                        MPI_Reduce(&t.compute, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                        if (rank == 0) {
                            dist.push_back(mx[0]);
                            comp.push_back(mx[1]);
                            coll.push_back(mx[2]);
                            tot.push_back(mx[3]);
                            // max / mean of per-rank compute time (1.0 = perfectly balanced)
                            imb.push_back(comp_sum > 0 ? mx[1] / (comp_sum / P) : 1.0);
                        }
                    }

                    if (dist_input && verify) {  // collect outside the timed region
                        if (rank == 0) {
                            input = conv::make_synthetic_input(p, 1234);
                            output.resize(p.output_size());
                        }
                        conv::gather_output_rows(p, local_out.data(), output.data(), MPI_COMM_WORLD);
                    }
                    if (rank != 0) continue;

                    const char* check = "-";
                    if (verify) {
                        std::vector<float> ref(p.output_size());
                        double max_abs = 0, max_ref = 0;
                        for (int b = 0; b < batch; ++b) {
                            conv::conv2d_serial(p, input.data() + b * p.input_size(), weight.data(),
                                                nullptr, ref.data(), conv::Algo::Direct);
                            const float* o = output.data() + b * p.output_size();
                            for (size_t i = 0; i < ref.size(); ++i) {
                                max_abs = std::max(max_abs, static_cast<double>(std::fabs(ref[i] - o[i])));
                                max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref[i])));
                            }
                        }
                        check = max_abs <= 1e-5 * std::max(max_ref, 1.0) ? "PASS" : "FAIL";
                    }

                    const Stats st = stats_of(tot);
                    const double gflops = p.flops() * batch / st.median * 1e-9;
                    const double ips = batch / st.median;
                    const double imbalance = median_of(imb);
                    char sz[32];
                    std::snprintf(sz, sizeof sz, "%dx%d", p.height, p.width);
                    std::printf("%-7s %11s %3d %10.3f %10.3f %10.3f %10.3f %7.3f %9.2f %10.1f %s\n",
                                conv::algo_name(algo), sz, k, st.median * 1e3, median_of(dist) * 1e3,
                                median_of(comp) * 1e3, median_of(coll) * 1e3, imbalance, gflops, ips,
                                check);
                    if (out) {
                        std::fprintf(out,
                                     "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
                                     "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.4f,%.4f,%.4f,%s\n",
                                     label.c_str(), mode_tag.c_str(), conv::algo_name(algo), P, p.height,
                                     p.width, p.in_channels, p.out_channels, k, p.stride, p.padding,
                                     batch, repeat, st.median, st.min, st.mean, st.stddev, median_of(dist),
                                     median_of(comp), median_of(coll), imbalance, gflops, ips, check);
                        std::fflush(out);
                    }
                }
        if (out) std::fclose(out);
    } catch (const std::exception& e) {
        if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}
