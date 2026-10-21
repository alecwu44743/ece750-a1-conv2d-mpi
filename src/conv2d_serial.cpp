// conv2d_serial.cpp - single-process convolution kernels (direct + im2col/GEMM).
#include "conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

namespace conv {

Algo parse_algo(const std::string& name) {
    if (name == "direct") return Algo::Direct;
    if (name == "im2col" || name == "gemm") return Algo::Im2colGemm;
    throw std::invalid_argument("unknown algorithm '" + name + "' (use direct|im2col)");
}

const char* algo_name(Algo a) { return a == Algo::Direct ? "direct" : "im2col"; }

bool ConvParams::valid(std::string* why) const {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (in_channels < 1 || out_channels < 1) return fail("channels must be >= 1");
    if (height < 1 || width < 1) return fail("image must be at least 1x1");
    if (kernel < 1) return fail("kernel size must be >= 1");
    if (stride < 1) return fail("stride must be >= 1");
    if (padding < 0) return fail("padding must be >= 0");
    if (kernel > padded_h() || kernel > padded_w()) return fail("kernel larger than (padded) image");
    return true;
}

std::string ConvParams::to_string() const {
    std::ostringstream os;
    os << "C_in=" << in_channels << " C_out=" << out_channels << " in=" << height << "x" << width
       << " kernel=" << kernel << "x" << kernel << " stride=" << stride << " padding=" << padding
       << " bias=" << (use_bias ? 1 : 0) << " -> out=" << out_h() << "x" << out_w();
    return os.str();
}

std::vector<float> pad_input(const float* in, int C, int H, int W, int p) {
    const int Hp = H + 2 * p, Wp = W + 2 * p;
    std::vector<float> out(static_cast<size_t>(C) * Hp * Wp, 0.0f);
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            std::memcpy(&out[(static_cast<size_t>(c) * Hp + y + p) * Wp + p],
                        &in[(static_cast<size_t>(c) * H + y) * W], sizeof(float) * W);
    return out;
}

// ---------------------------------------------------------------------------
// GEMM: C += A * B.  No BLAS (not allowed by the assignment) - blocked so the
// active row of C stays in L1 and the B panel stays in L2; the inner j-loop is
// unit-stride and is auto-vectorised by the compiler (-O3 -march=native).
// ---------------------------------------------------------------------------
void gemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
          float* C, int ldc) {
    constexpr int NB = 1024;  // columns of C/B per block
    constexpr int KB = 128;   // depth per block
    for (int j0 = 0; j0 < N; j0 += NB) {
        const int nb = std::min(NB, N - j0);
        for (int k0 = 0; k0 < K; k0 += KB) {
            const int kend = std::min(K, k0 + KB);
            for (int i = 0; i < M; ++i) {
                float* __restrict c = C + static_cast<size_t>(i) * ldc + j0;
                const float* a = A + static_cast<size_t>(i) * lda;
                int k = k0;
                for (; k + 3 < kend; k += 4) {  // 4-way unroll: fewer C loads/stores
                    const float a0 = a[k], a1 = a[k + 1], a2 = a[k + 2], a3 = a[k + 3];
                    const float* __restrict b0 = B + static_cast<size_t>(k) * ldb + j0;
                    const float* __restrict b1 = b0 + ldb;
                    const float* __restrict b2 = b1 + ldb;
                    const float* __restrict b3 = b2 + ldb;
                    for (int j = 0; j < nb; ++j)
                        c[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                }
                for (; k < kend; ++k) {
                    const float a0 = a[k];
                    const float* __restrict b0 = B + static_cast<size_t>(k) * ldb + j0;
                    for (int j = 0; j < nb; ++j) c[j] += a0 * b0[j];
                }
            }
        }
    }
}

namespace {

void init_output(const ConvParams& p, int out_rows, const float* bias, float* out) {
    const size_t plane = static_cast<size_t>(out_rows) * p.out_w();
    for (int co = 0; co < p.out_channels; ++co)
        std::fill(out + co * plane, out + (co + 1) * plane,
                  (p.use_bias && bias) ? bias[co] : 0.0f);
}

// Sliding-window convolution.  Loop order (co, ci, oy, ky, kx, ox) keeps one
// output row hot in L1 and makes the innermost loop a unit-stride AXPY when
// stride == 1.
void slab_direct(const ConvParams& p, const float* slab, int slab_rows, int out_rows, const float* weight,
                 const float* bias, float* out) {
    const int f = p.kernel, s = p.stride, Wp = p.padded_w(), Wo = p.out_w();
    init_output(p, out_rows, bias, out);
    for (int co = 0; co < p.out_channels; ++co) {
        float* o = out + static_cast<size_t>(co) * out_rows * Wo;
        for (int ci = 0; ci < p.in_channels; ++ci) {
            const float* in_c = slab + static_cast<size_t>(ci) * slab_rows * Wp;
            const float* w = weight + (static_cast<size_t>(co) * p.in_channels + ci) * f * f;
            for (int oy = 0; oy < out_rows; ++oy) {
                float* __restrict orow = o + static_cast<size_t>(oy) * Wo;
                for (int ky = 0; ky < f; ++ky) {
                    const float* irow = in_c + static_cast<size_t>(oy * s + ky) * Wp;
                    for (int kx = 0; kx < f; ++kx) {
                        const float wv = w[ky * f + kx];
                        const float* __restrict ip = irow + kx;
                        if (s == 1) {
                            for (int ox = 0; ox < Wo; ++ox) orow[ox] += wv * ip[ox];
                        } else {
                            for (int ox = 0; ox < Wo; ++ox) orow[ox] += wv * ip[ox * s];
                        }
                    }
                }
            }
        }
    }
}

// im2col + GEMM (slides 13-14).  The column matrix is built for a chunk of
// output rows at a time so its size stays bounded (~16 MB) for huge images.
void slab_im2col(const ConvParams& p, const float* slab, int slab_rows, int out_rows, const float* weight,
                 const float* bias, float* out) {
    const int f = p.kernel, s = p.stride, Wp = p.padded_w(), Wo = p.out_w();
    const int K = p.in_channels * f * f;
    constexpr long long kColBudget = 4LL << 20;  // floats
    const int chunk = static_cast<int>(
        std::max(1LL, std::min<long long>(out_rows, kColBudget / (1LL * K * Wo))));
    std::vector<float> col(static_cast<size_t>(K) * chunk * Wo);

    init_output(p, out_rows, bias, out);
    for (int r0 = 0; r0 < out_rows; r0 += chunk) {
        const int cr = std::min(chunk, out_rows - r0);
        const int N = cr * Wo;
        // im2col: row k = (ci, ky, kx), column n = (oy, ox) sliding-window position.
        for (int ci = 0; ci < p.in_channels; ++ci) {
            const float* in_c = slab + static_cast<size_t>(ci) * slab_rows * Wp;
            for (int ky = 0; ky < f; ++ky)
                for (int kx = 0; kx < f; ++kx) {
                    float* dst = col.data() + static_cast<size_t>((ci * f + ky) * f + kx) * N;
                    for (int oy = 0; oy < cr; ++oy) {
                        const float* irow = in_c + static_cast<size_t>((r0 + oy) * s + ky) * Wp + kx;
                        float* d = dst + static_cast<size_t>(oy) * Wo;
                        if (s == 1) {
                            std::memcpy(d, irow, sizeof(float) * Wo);
                        } else {
                            for (int ox = 0; ox < Wo; ++ox) d[ox] = irow[ox * s];
                        }
                    }
                }
        }
        // out[:, r0:r0+cr, :] (C_out x N, row pitch out_rows*Wo) += W (C_out x K) * col (K x N)
        gemm(p.out_channels, N, K, weight, K, col.data(), N,
             out + static_cast<size_t>(r0) * Wo, out_rows * Wo);
    }
}

}  // namespace

void conv2d_slab(const ConvParams& p, const float* slab, int slab_rows, int out_rows,
                 const float* weight, const float* bias, float* out, Algo algo) {
    if (out_rows <= 0) return;
    if (algo == Algo::Direct)
        slab_direct(p, slab, slab_rows, out_rows, weight, bias, out);
    else
        slab_im2col(p, slab, slab_rows, out_rows, weight, bias, out);
}

void conv2d_serial(const ConvParams& p, const float* input, const float* weight,
                   const float* bias, float* output, Algo algo) {
    if (p.padding == 0) {
        conv2d_slab(p, input, p.height, p.out_h(), weight, bias, output, algo);
    } else {
        std::vector<float> padded = pad_input(input, p.in_channels, p.height, p.width, p.padding);
        conv2d_slab(p, padded.data(), p.padded_h(), p.out_h(), weight, bias, output, algo);
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------
bool make_named_kernel(const std::string& name, int f, std::vector<float>* k) {
    k->assign(static_cast<size_t>(f) * f, 0.0f);
    auto set3 = [&](std::initializer_list<float> v) {
        if (f != 3) return false;
        std::copy(v.begin(), v.end(), k->begin());
        return true;
    };
    if (name == "edge") return set3({-1, -1, -1, -1, 8, -1, -1, -1, -1});
    if (name == "sobel_x") return set3({-1, 0, 1, -2, 0, 2, -1, 0, 1});
    if (name == "sobel_y") return set3({-1, -2, -1, 0, 0, 0, 1, 2, 1});
    if (name == "sharpen") return set3({0, -1, 0, -1, 5, -1, 0, -1, 0});
    if (name == "identity") {
        if (f % 2 == 0) return false;
        (*k)[(f / 2) * f + f / 2] = 1.0f;
        return true;
    }
    if (name == "box") {
        std::fill(k->begin(), k->end(), 1.0f / (f * f));
        return true;
    }
    if (name == "gaussian") {
        // Same sigma rule as cv::getGaussianKernel(ksize, sigma <= 0).
        const double sigma = 0.3 * ((f - 1) * 0.5 - 1) + 0.8;
        std::vector<double> g(f);
        double sum = 0;
        for (int i = 0; i < f; ++i) {
            const double x = i - (f - 1) * 0.5;
            g[i] = std::exp(-x * x / (2 * sigma * sigma));
            sum += g[i];
        }
        for (int y = 0; y < f; ++y)
            for (int x = 0; x < f; ++x) (*k)[y * f + x] = static_cast<float>(g[y] * g[x] / (sum * sum));
        return true;
    }
    return false;
}

void fill_random(std::vector<float>& w, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : w) x = dist(rng);
}

}  // namespace conv
