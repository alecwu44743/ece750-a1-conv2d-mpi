// test_conv2d.cpp - unit tests (correctness = 20% of the grade).
//
// Run serially and with several process counts, e.g.
//   mpirun -np 1 ./bin/conv2d_test && mpirun -np 3 ./bin/conv2d_test && mpirun -np 8 ./bin/conv2d_test
//
// Tests:
//   1. Hand-computed examples from the course slides (slides 4, 6, 10, 13-14).
//   2. Output-size formula k = (d - f + 2p)/s + 1.
//   3. Identity / box kernels with known results.
//   4. Direct vs im2col+GEMM vs naive 7-loop reference on random tensors
//      (many combinations of C_in, C_out, f, stride, padding, bias, odd sizes).
//   5. MPI spatial and batch decompositions vs serial, including P > H_out
//      (ranks with no rows) and non-divisible row counts.
//   6. Distributed-input modes (pre-distributed slab and MPI_Sendrecv halo
//      exchange) vs the serial result, incl. halos spanning several ranks.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "conv2d.hpp"

namespace {

int g_rank = 0, g_size = 1, g_failed = 0, g_passed = 0;

void check(bool cond, const std::string& name) {
    if (cond) {
        ++g_passed;
    } else {
        ++g_failed;
        if (g_rank == 0) std::printf("  FAIL: %s\n", name.c_str());
    }
}

bool close(const std::vector<float>& a, const std::vector<float>& b, double tol = 1e-4) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol * std::max(1.0, static_cast<double>(std::fabs(b[i])))) return false;
    return true;
}

// Textbook definition - deliberately the simplest possible code (7 nested loops,
// bounds-checked padding) so it can serve as an independent oracle.
std::vector<float> naive_conv(const conv::ConvParams& p, const std::vector<float>& in,
                              const std::vector<float>& w, const std::vector<float>& b) {
    const int Ho = p.out_h(), Wo = p.out_w(), f = p.kernel;
    std::vector<float> out(p.output_size());
    for (int co = 0; co < p.out_channels; ++co)
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                double acc = p.use_bias ? b[co] : 0.0;
                for (int ci = 0; ci < p.in_channels; ++ci)
                    for (int ky = 0; ky < f; ++ky)
                        for (int kx = 0; kx < f; ++kx) {
                            const int y = oy * p.stride + ky - p.padding;
                            const int x = ox * p.stride + kx - p.padding;
                            if (y < 0 || y >= p.height || x < 0 || x >= p.width) continue;
                            acc += static_cast<double>(in[(ci * p.height + y) * p.width + x]) *
                                   w[((co * p.in_channels + ci) * f + ky) * f + kx];
                        }
                out[(co * Ho + oy) * Wo + ox] = static_cast<float>(acc);
            }
    return out;
}

conv::ConvParams make(int H, int W, int f, int s = 1, int pad = 0, int cin = 1, int cout = 1,
                      bool bias = false) {
    conv::ConvParams p;
    p.height = H;
    p.width = W;
    p.kernel = f;
    p.stride = s;
    p.padding = pad;
    p.in_channels = cin;
    p.out_channels = cout;
    p.use_bias = bias;
    return p;
}

std::vector<float> serial(const conv::ConvParams& p, const std::vector<float>& in,
                          const std::vector<float>& w, const std::vector<float>& b, conv::Algo a) {
    std::vector<float> out(p.output_size());
    conv::conv2d_serial(p, in.data(), w.data(), b.empty() ? nullptr : b.data(), out.data(), a);
    return out;
}

std::vector<float> distributed(const conv::ConvParams& p, const std::vector<float>& in,
                               const std::vector<float>& w, const std::vector<float>& b, conv::Algo a) {
    std::vector<float> out(g_rank == 0 ? p.output_size() : 0);
    conv::conv2d_mpi(p, in.data(), w.data(), b.empty() ? nullptr : b.data(), out.data(), a,
                     MPI_COMM_WORLD);
    return out;
}

// ---------------------------------------------------------------------------

void test_slide_examples() {
    // Slide 4: 5x5 image (a..z -> 1..25), 3x3 kernel (A..J -> 1..9), no padding.
    std::vector<float> img(25), ker(9);
    for (int i = 0; i < 25; ++i) img[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 9; ++i) ker[i] = static_cast<float>(i + 1);
    auto p = make(5, 5, 3);
    check(p.out_h() == 3 && p.out_w() == 3, "slide 4: 5x5 * 3x3 -> 3x3");
    // alpha = (a*A + b*B + c*C) + (f*D + g*E + h*F) + (k*G + l*H + m*J)
    const float alpha = 1 * 1 + 2 * 2 + 3 * 3 + 6 * 4 + 7 * 5 + 8 * 6 + 11 * 7 + 12 * 8 + 13 * 9;
    for (conv::Algo a : {conv::Algo::Direct, conv::Algo::Im2colGemm}) {
        auto out = serial(p, img, ker, {}, a);
        check(out[0] == alpha, std::string("slide 4 alpha (") + conv::algo_name(a) + ")");
        check(close(out, naive_conv(p, img, ker, {})), std::string("slide 4 full map (") + conv::algo_name(a) + ")");
    }

    // Slide 6: padding = 1 keeps the 5x5 size; alpha = a*E + b*F + f*H + g*J.
    auto pp = make(5, 5, 3, 1, 1);
    check(pp.out_h() == 5 && pp.out_w() == 5, "slide 6: padding 1 -> 5x5");
    auto outp = serial(pp, img, ker, {}, conv::Algo::Direct);
    check(outp[0] == 1 * 5 + 2 * 6 + 6 * 8 + 7 * 9, "slide 6 padded alpha");

    // Slide 10: stride 2 -> 2x2; beta uses window starting at column 2 (c..o).
    auto ps = make(5, 5, 3, 2);
    check(ps.out_h() == 2 && ps.out_w() == 2, "slide 10: stride 2 -> 2x2");
    auto outs = serial(ps, img, ker, {}, conv::Algo::Im2colGemm);
    const float beta = 3 * 1 + 4 * 2 + 5 * 3 + 8 * 4 + 9 * 5 + 10 * 6 + 13 * 7 + 14 * 8 + 15 * 9;
    check(outs[1] == beta, "slide 10 stride-2 beta");

    // Slides 13-14: 4x4 image, 2x2 kernel -> 3x3 (9 im2col columns).
    std::vector<float> img4(16), ker2 = {1, 2, 3, 4};
    for (int i = 0; i < 16; ++i) img4[i] = static_cast<float>(i + 1);
    auto p4 = make(4, 4, 2);
    auto out4 = serial(p4, img4, ker2, {}, conv::Algo::Im2colGemm);
    check(out4.size() == 9, "slide 13: 9 sliding windows");
    check(out4[0] == 1 * 1 + 2 * 2 + 5 * 3 + 6 * 4, "slide 14: alpha = aA+bB+eC+fD");
    check(out4[8] == 11 * 1 + 12 * 2 + 15 * 3 + 16 * 4, "slide 14: last window");
}

void test_output_size() {
    const int cases[][5] = {  // d, f, p, s, expected
        {5, 3, 0, 1, 3}, {5, 3, 1, 1, 5}, {5, 3, 0, 2, 2}, {7, 3, 0, 2, 3},
        {224, 7, 3, 2, 112}, {32, 5, 2, 1, 32}, {4, 2, 0, 1, 3}, {10, 4, 1, 3, 3}};
    for (const auto& c : cases) {
        auto p = make(c[0], c[0], c[1], c[3], c[2]);
        check(p.out_h() == c[4] && p.out_w() == c[4],
              "output size d=" + std::to_string(c[0]) + " f=" + std::to_string(c[1]) +
                  " p=" + std::to_string(c[2]) + " s=" + std::to_string(c[3]));
    }
}

void test_known_kernels() {
    std::vector<float> img(13 * 17);
    conv::fill_random(img, 7);
    // Identity kernel with "same" padding reproduces the input exactly.
    for (int f : {1, 3, 5}) {
        std::vector<float> k;
        conv::make_named_kernel("identity", f, &k);
        auto p = make(13, 17, f, 1, f / 2);
        for (conv::Algo a : {conv::Algo::Direct, conv::Algo::Im2colGemm})
            check(close(serial(p, img, k, {}, a), img, 1e-6), "identity kernel f=" + std::to_string(f));
    }
    // Box filter on a constant image gives the constant; Gaussian kernel sums to 1.
    std::vector<float> ones(20 * 20, 2.5f), box, gauss;
    conv::make_named_kernel("box", 5, &box);
    conv::make_named_kernel("gaussian", 7, &gauss);
    auto out = serial(make(20, 20, 5), ones, box, {}, conv::Algo::Direct);
    check(close(out, std::vector<float>(out.size(), 2.5f), 1e-5), "box filter on constant image");
    double s = 0;
    for (float g : gauss) s += g;
    check(std::fabs(s - 1.0) < 1e-5, "gaussian kernel normalised");
    // Edge kernel (slide 3) on a constant image is zero everywhere.
    std::vector<float> edge;
    conv::make_named_kernel("edge", 3, &edge);
    auto e = serial(make(20, 20, 3), ones, edge, {}, conv::Algo::Im2colGemm);
    check(close(e, std::vector<float>(e.size(), 0.0f), 1e-6), "edge kernel on constant image");
}

struct Case {
    int H, W, f, s, pad, cin, cout;
    bool bias;
};

const Case kRandomCases[] = {
    {1, 1, 1, 1, 0, 1, 1, false},   {5, 5, 3, 1, 0, 1, 1, false},  {31, 17, 3, 1, 1, 1, 1, true},
    {64, 64, 5, 1, 2, 3, 4, true},  {37, 53, 7, 2, 3, 2, 3, false}, {40, 40, 4, 3, 1, 1, 2, true},
    {9, 100, 3, 1, 0, 1, 1, false}, {100, 9, 9, 1, 0, 1, 1, false}, {50, 50, 1, 2, 0, 3, 5, true},
    {23, 29, 11, 1, 5, 2, 2, true}, {128, 96, 3, 2, 1, 4, 8, true}, {3, 3, 3, 1, 0, 1, 1, true},
};

void test_serial_algorithms() {
    unsigned seed = 1;
    for (const Case& c : kRandomCases) {
        auto p = make(c.H, c.W, c.f, c.s, c.pad, c.cin, c.cout, c.bias);
        std::vector<float> in(p.input_size()), w(p.weight_size()), b(c.bias ? c.cout : 0);
        conv::fill_random(in, seed++);
        conv::fill_random(w, seed++);
        conv::fill_random(b, seed++);
        auto ref = naive_conv(p, in, w, b);
        check(close(serial(p, in, w, b, conv::Algo::Direct), ref), "direct == naive: " + p.to_string());
        check(close(serial(p, in, w, b, conv::Algo::Im2colGemm), ref), "im2col == naive: " + p.to_string());
    }
}

void test_gemm() {
    const int M = 7, N = 1500, K = 133;  // crosses the NB/KB block boundaries
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
    conv::fill_random(A, 11);
    conv::fill_random(B, 12);
    conv::gemm(M, N, K, A.data(), K, B.data(), N, C.data(), N);
    std::vector<float> R(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += static_cast<double>(A[i * K + k]) * B[k * N + j];
            R[i * N + j] = static_cast<float>(acc);
        }
    check(close(C, R, 1e-4), "blocked GEMM == triple loop");
}

void test_mpi_spatial() {
    unsigned seed = 100;
    for (const Case& c : kRandomCases)
        for (conv::Algo a : {conv::Algo::Direct, conv::Algo::Im2colGemm}) {
            auto p = make(c.H, c.W, c.f, c.s, c.pad, c.cin, c.cout, c.bias);
            std::vector<float> in(p.input_size()), w(p.weight_size()), b(c.bias ? c.cout : 0);
            conv::fill_random(in, seed++);
            conv::fill_random(w, seed++);
            conv::fill_random(b, seed++);
            auto out = distributed(p, in, w, b, a);
            if (g_rank == 0)
                check(close(out, naive_conv(p, in, w, b)),
                      std::string("mpi spatial (") + conv::algo_name(a) + ", P=" +
                          std::to_string(g_size) + "): " + p.to_string());
        }
}

void test_mpi_batch() {
    for (int batch : {1, 5, 2 * g_size + 1}) {
        auto p = make(19, 23, 3, 1, 1, 2, 3, true);
        std::vector<float> in(p.input_size() * batch), w(p.weight_size()), b(p.out_channels);
        conv::fill_random(in, 500 + batch);
        conv::fill_random(w, 600);
        conv::fill_random(b, 700);
        std::vector<float> out(g_rank == 0 ? p.output_size() * batch : 0);
        conv::conv2d_mpi_batch(p, batch, in.data(), w.data(), b.data(), out.data(),
                               conv::Algo::Im2colGemm, MPI_COMM_WORLD);
        if (g_rank != 0) continue;
        bool ok = true;
        for (int i = 0; i < batch; ++i) {
            std::vector<float> img(in.begin() + i * p.input_size(), in.begin() + (i + 1) * p.input_size());
            std::vector<float> got(out.begin() + i * p.output_size(), out.begin() + (i + 1) * p.output_size());
            ok = ok && close(got, naive_conv(p, img, w, b));
        }
        check(ok, "mpi batch == serial, batch=" + std::to_string(batch));
    }
}

void test_block_range() {
    for (int n : {0, 1, 7, 100})
        for (int P : {1, 3, 8, 16}) {
            int next = 0;
            bool ok = true;
            for (int r = 0; r < P; ++r) {
                int s, c;
                conv::block_range(n, P, r, &s, &c);
                ok = ok && s == next && c >= n / P && c <= n / P + 1;
                next = s + c;
            }
            check(ok && next == n, "block_range n=" + std::to_string(n) + " P=" + std::to_string(P));
        }
}


void test_row_layout() {
    for (const Case& c : kRandomCases) {
        auto p = make(c.H, c.W, c.f, c.s, c.pad, c.cin, c.cout, c.bias);
        bool ok = true;
        int next_own = 0, next_out = 0;
        for (int r = 0; r < g_size; ++r) {
            const conv::RowLayout L = conv::row_layout(p, g_size, r);
            ok = ok && L.own_start == next_own && L.own_end >= L.own_start && L.out_start == next_out;
            ok = ok && (L.out_cnt == 0 || (L.need_start == L.own_start && L.need_end <= p.padded_h()));
            next_own = L.own_end;
            next_out = L.out_start + L.out_cnt;
        }
        check(ok && next_own == p.padded_h() && next_out == p.out_h(),
              "row_layout tiles rows, P=" + std::to_string(g_size) + ": " + p.to_string());
    }
}

void test_mpi_distributed_input() {
    std::vector<Case> cases(std::begin(kRandomCases), std::end(kRandomCases));
    cases.push_back({16, 20, 7, 1, 0, 2, 3, true});  // 6-row halo: spans several thin blocks
    cases.push_back({30, 30, 3, 4, 1, 1, 1, false}); // stride > kernel: no halo at all
    unsigned seed = 900;
    for (const Case& c : cases)
        for (conv::Algo a : {conv::Algo::Direct, conv::Algo::Im2colGemm})
            for (bool halo : {false, true}) {
                auto p = make(c.H, c.W, c.f, c.s, c.pad, c.cin, c.cout, c.bias);
                const unsigned s_in = seed++;
                std::vector<float> w(p.weight_size()), b(c.bias ? c.cout : 0);
                conv::fill_random(w, seed++);  // same on every rank (resident model)
                conv::fill_random(b, seed++);
                const conv::RowLayout L = conv::row_layout(p, g_size, g_rank);
                const int rows = halo ? L.own_rows() : L.need_rows();
                std::vector<float> local_in(static_cast<size_t>(p.in_channels) * rows * p.padded_w());
                conv::make_padded_rows(p, s_in, halo ? L.own_start : L.need_start, rows, local_in.data());
                std::vector<float> local_out(static_cast<size_t>(p.out_channels) * L.out_cnt * p.out_w());
                const float* bp = b.empty() ? nullptr : b.data();
                if (halo)
                    conv::conv2d_mpi_halo(p, local_in.data(), w.data(), bp, local_out.data(), a, MPI_COMM_WORLD);
                else
                    conv::conv2d_mpi_local(p, local_in.data(), w.data(), bp, local_out.data(), a, MPI_COMM_WORLD);
                std::vector<float> out(g_rank == 0 ? p.output_size() : 0);
                conv::gather_output_rows(p, local_out.data(), out.data(), MPI_COMM_WORLD);
                if (g_rank == 0) {
                    auto in = conv::make_synthetic_input(p, s_in);
                    check(close(out, serial(p, in, w, b, conv::Algo::Direct)),
                          std::string("mpi ") + (halo ? "halo-exchange" : "pre-distributed") + " (" +
                              conv::algo_name(a) + ", P=" + std::to_string(g_size) + "): " + p.to_string());
                }
            }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    if (g_rank == 0) std::printf("conv2d_test on %d process(es)\n", g_size);
    if (g_rank == 0) {  // serial tests only on rank 0
        test_slide_examples();
        test_output_size();
        test_known_kernels();
        test_gemm();
        test_serial_algorithms();
        test_block_range();
    }
    test_mpi_spatial();
    test_mpi_batch();
    test_mpi_distributed_input();
    if (g_rank == 0) test_row_layout();

    if (g_rank == 0) std::printf("%d passed, %d failed\n", g_passed, g_failed);
    MPI_Bcast(&g_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return g_failed == 0 ? 0 : 1;
}
