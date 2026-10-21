// main.cpp - conv2d_run: run one distributed convolution on a real or synthetic
// image, print the timing breakdown, optionally verify against the serial
// reference and dump tensors for checking with PyTorch / SciPy.
//
//   mpirun -np 4 ./bin/conv2d_run --image data/lena.pgm --filter edge --out edge.pgm --verify
//   mpirun -np 4 ./bin/conv2d_run --size 4096 --kernel-size 5 --filter gaussian --repeat 5
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "conv2d.hpp"
#include "io.hpp"

namespace {

void usage() {
    std::printf(
        "Usage: mpirun -np P conv2d_run [options]\n"
        "  --image FILE.pgm        grayscale input image (otherwise synthetic)\n"
        "  --size HxW|N            synthetic image size            (default 1024)\n"
        "  --cin N --cout N        input / output channels         (default 1 / 1)\n"
        "  --kernel-size f         kernel size f x f               (default 3)\n"
        "  --filter NAME           edge|sobel_x|sobel_y|sharpen|box|gaussian|identity|random (default edge)\n"
        "  --stride s --padding p  (default 1 / 0)\n"
        "  --bias                  add a random bias per output channel\n"
        "  --algo direct|im2col    local kernel                    (default direct)\n"
        "  --repeat N              timed repetitions               (default 3)\n"
        "  --verify                compare with the serial reference on rank 0\n"
        "  --out FILE.pgm          write output channel 0 as an image\n"
        "  --dump DIR              write input/weight/bias/output .npy for verify_reference.py\n"
        "  --seed N                RNG seed                        (default 42)\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int rc = 0;
    try {
        cli::Args args(argc, argv);
        if (args.has("help")) {
            if (rank == 0) usage();
            MPI_Finalize();
            return 0;
        }

        conv::ConvParams p;
        p.in_channels = args.num("cin", 1);
        p.out_channels = args.num("cout", 1);
        p.kernel = args.num("kernel-size", 3);
        p.stride = args.num("stride", 1);
        p.padding = args.num("padding", 0);
        p.use_bias = args.has("bias");
        const conv::Algo algo = conv::parse_algo(args.str("algo", "direct"));
        const int repeat = std::max(1, args.num("repeat", 3));
        const unsigned seed = static_cast<unsigned>(args.num("seed", 42));
        const std::string filter = args.str("filter", "edge");

        // ---- input (rank 0 owns the data; only the shape is broadcast) ----
        std::vector<float> input;
        int dims[2] = {0, 0};
        if (rank == 0) {
            if (args.has("image")) {
                std::vector<float> gray;
                if (!io::read_pgm(args.str("image", ""), &gray, &dims[0], &dims[1]))
                    throw std::runtime_error("cannot read PGM image " + args.str("image", ""));
                // Replicate the grayscale plane if more than one input channel is requested.
                input.resize(gray.size() * p.in_channels);
                for (int c = 0; c < p.in_channels; ++c)
                    std::copy(gray.begin(), gray.end(), input.begin() + c * gray.size());
            } else {
                cli::parse_size(args.str("size", "1024"), &dims[0], &dims[1]);
            }
        }
        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
        p.height = dims[0];
        p.width = dims[1];
        std::string why;
        if (!p.valid(&why)) throw std::invalid_argument("invalid parameters: " + why);
        if (rank == 0 && input.empty()) {
            input.resize(p.input_size());
            conv::fill_random(input, seed);
        }

        // ---- weights ----
        std::vector<float> weight, bias;
        if (rank == 0) {
            weight.resize(p.weight_size());
            if (filter == "random") {
                conv::fill_random(weight, seed + 1);
            } else {
                std::vector<float> k;
                if (!conv::make_named_kernel(filter, p.kernel, &k))
                    throw std::invalid_argument("filter '" + filter + "' not available for f=" +
                                                std::to_string(p.kernel));
                for (long long i = 0; i < p.out_channels * p.in_channels; ++i)
                    std::copy(k.begin(), k.end(), weight.begin() + i * k.size());
            }
            if (p.use_bias) {
                bias.resize(p.out_channels);
                conv::fill_random(bias, seed + 2);
            }
            std::printf("conv2d_run: P=%d algo=%s filter=%s\n  %s\n", P, conv::algo_name(algo),
                        filter.c_str(), p.to_string().c_str());
        }

        // ---- run ----
        std::vector<float> output(rank == 0 ? p.output_size() : 0);
        for (int it = 0; it < repeat; ++it) {
            conv::Timing t = conv::conv2d_mpi(p, input.data(), weight.data(), bias.data(),
                                              output.data(), algo, MPI_COMM_WORLD);
            double local[4] = {t.distribute, t.compute, t.collect, t.total}, mx[4];
            MPI_Reduce(local, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                std::printf("  run %d: total %.3f ms (distribute %.3f, compute %.3f, collect %.3f) "
                            "%.2f GFLOP/s\n",
                            it, mx[3] * 1e3, mx[0] * 1e3, mx[1] * 1e3, mx[2] * 1e3,
                            p.flops() / mx[3] * 1e-9);
        }

        if (rank == 0) {
            if (args.has("verify")) {
                std::vector<float> ref(p.output_size());
                conv::conv2d_serial(p, input.data(), weight.data(), bias.data(), ref.data(),
                                    conv::Algo::Direct);
                double max_abs = 0, max_ref = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    max_abs = std::max(max_abs, static_cast<double>(std::fabs(ref[i] - output[i])));
                    max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref[i])));
                }
                const double rel = max_abs / std::max(max_ref, 1e-30);
                const bool ok = rel < 1e-5;
                std::printf("  verify vs serial: max|err|=%.3e rel=%.3e -> %s\n", max_abs, rel,
                            ok ? "PASS" : "FAIL");
                if (!ok) rc = 1;
            }
            if (args.has("out")) {
                const std::string out = args.str("out", "out.pgm");
                const bool abs_val = filter == "edge" || filter.rfind("sobel", 0) == 0;
                if (!io::write_pgm(out, output.data(), p.out_h(), p.out_w(), abs_val))
                    throw std::runtime_error("cannot write " + out);
                std::printf("  wrote %s (%dx%d)\n", out.c_str(), p.out_h(), p.out_w());
            }
            if (args.has("dump")) {
                const std::string d = args.str("dump", ".");
                const long long f = p.kernel;
                bool ok = io::write_npy(d + "/input.npy", input.data(), {p.in_channels, p.height, p.width}) &&
                          io::write_npy(d + "/weight.npy", weight.data(), {p.out_channels, p.in_channels, f, f}) &&
                          io::write_npy(d + "/output.npy", output.data(), {p.out_channels, p.out_h(), p.out_w()});
                if (p.use_bias) ok = ok && io::write_npy(d + "/bias.npy", bias.data(), {p.out_channels});
                FILE* js = std::fopen((d + "/params.json").c_str(), "w");
                if (js) {
                    std::fprintf(js, "{\"stride\": %d, \"padding\": %d, \"bias\": %s}\n", p.stride,
                                 p.padding, p.use_bias ? "true" : "false");
                    std::fclose(js);
                }
                if (!ok || !js) throw std::runtime_error("cannot write dump files to " + d);
                std::printf("  dumped tensors to %s/\n", d.c_str());
            }
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
