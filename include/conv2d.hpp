// conv2d.hpp - 2D convolution layer (the core of a CNN) with serial and
// MPI-distributed implementations.
//
// Conventions (same as PyTorch nn.Conv2d / the course slides):
//   * The operation is cross-correlation (the kernel is NOT flipped).
//   * All tensors are dense, row-major float32 arrays:
//       input  [C_in ][H    ][W    ]
//       weight [C_out][C_in ][f][f]
//       bias   [C_out]
//       output [C_out][H_out][W_out]
//   * H_out = (H - f + 2p) / s + 1, W_out likewise (slide 11).
#pragma once

#include <mpi.h>

#include <string>
#include <vector>

namespace conv {

enum class Algo {
    Direct,      // sliding window, vectorised over output columns
    Im2colGemm,  // im2col + hand-written blocked GEMM (no BLAS)
};

Algo parse_algo(const std::string& name);
const char* algo_name(Algo a);

struct ConvParams {
    int in_channels = 1;
    int out_channels = 1;
    int height = 0;  // input H
    int width = 0;   // input W
    int kernel = 3;  // f (square kernel)
    int stride = 1;
    int padding = 0;
    bool use_bias = false;

    int padded_h() const { return height + 2 * padding; }
    int padded_w() const { return width + 2 * padding; }
    int out_h() const { return (padded_h() - kernel) / stride + 1; }
    int out_w() const { return (padded_w() - kernel) / stride + 1; }

    long long input_size() const { return 1LL * in_channels * height * width; }
    long long weight_size() const { return 1LL * out_channels * in_channels * kernel * kernel; }
    long long output_size() const { return 1LL * out_channels * out_h() * out_w(); }

    // 1 multiply + 1 add per kernel tap per output element.
    double flops() const { return 2.0 * weight_size() * out_h() * out_w(); }

    bool valid(std::string* why = nullptr) const;
    std::string to_string() const;
};

// ---------------------------------------------------------------------------
// Serial building blocks
// ---------------------------------------------------------------------------

// Zero-pads every channel of `in` ([C][H][W]) by p on each side.
std::vector<float> pad_input(const float* in, int C, int H, int W, int p);

// C[M x N] += A[M x K] * B[K x N]; cache-blocked, auto-vectorised inner loop.
void gemm(int M, int N, int K,
          const float* A, int lda,
          const float* B, int ldb,
          float* C, int ldc);

// Convolves a "slab" of an already padded input: `slab` is [C_in][slab_rows][Wp]
// where Wp = p.padded_w() and slab_rows >= (out_rows - 1) * stride + kernel is
// the per-channel row pitch of the slab.
// Writes `out` = [C_out][out_rows][W_out].  This is the unit of work that each
// MPI rank executes on its block of output rows.
void conv2d_slab(const ConvParams& p, const float* slab, int slab_rows, int out_rows,
                 const float* weight, const float* bias, float* out, Algo algo);

// Whole-image serial convolution (reference / single-process baseline).
void conv2d_serial(const ConvParams& p, const float* input, const float* weight,
                   const float* bias, float* output, Algo algo);

// ---------------------------------------------------------------------------
// MPI-distributed convolution
// ---------------------------------------------------------------------------

struct Timing {
    double distribute = 0;  // bcast weights + pad/pack + scatter input slabs
    double compute = 0;     // local convolution
    double collect = 0;     // gather output blocks + unpack
    double total = 0;
};

// Spatial (row-block) decomposition of ONE image/feature map.
// Output rows are split into P contiguous blocks; each rank receives exactly the
// input rows (including the (f - s) halo rows) needed for its block.
// `input`, `weight`, `bias`, `output` only need to be valid on `root`.
// `p` must be identical on all ranks.  Returns this rank's phase timings.
Timing conv2d_mpi(const ConvParams& p, const float* input, const float* weight,
                  const float* bias, float* output, Algo algo,
                  MPI_Comm comm, int root = 0);

// Batch (data-parallel) decomposition: `batch` independent images are split
// across ranks; every rank convolves its images with the full kernel.
// input [batch][C_in][H][W], output [batch][C_out][H_out][W_out] at root.
Timing conv2d_mpi_batch(const ConvParams& p, int batch, const float* input,
                        const float* weight, const float* bias, float* output,
                        Algo algo, MPI_Comm comm, int root = 0);

// Block partition helper: first index and count of block `r` of `n` items over `P`.
inline void block_range(int n, int P, int r, int* start, int* count) {
    int base = n / P, rem = n % P;
    *count = base + (r < rem ? 1 : 0);
    *start = r * base + (r < rem ? r : rem);
}

// ---------------------------------------------------------------------------
// Distributed-input mode (data already lives on the ranks, as it does between
// the layers of a CNN): no root, no scatter/gather.
// ---------------------------------------------------------------------------

// Row ownership for the spatial decomposition, in PADDED input coordinates.
// Rank r computes output rows [out_start, out_start + out_cnt) and therefore
// needs padded input rows [need_start, need_end) = its own block plus the
// (f - s) halo rows below it.  Ownership [own_start, own_end) tiles 0..Hp
// without overlap (the layout a previous layer would leave behind).
struct RowLayout {
    int out_start = 0, out_cnt = 0;
    int own_start = 0, own_end = 0;
    int need_start = 0, need_end = 0;
    int own_rows() const { return own_end - own_start; }
    int need_rows() const { return need_end - need_start; }
};
RowLayout row_layout(const ConvParams& p, int P, int r);

// (a) Pre-distributed slab: `slab` already holds this rank's needed rows
//     [C_in][need_rows][Wp] including the halo.  Only local compute is timed.
// (b) Halo exchange: `owned` holds only this rank's owned rows
//     [C_in][own_rows][Wp]; the halo rows are fetched from the following
//     rank(s) with MPI_Sendrecv (normally exactly one neighbour) and timed as
//     `distribute`.
// In both cases `weight`/`bias` must be valid on every rank (model parameters
// are resident) and `local_out` receives [C_out][out_cnt][W_out]; nothing is
// gathered (`collect` = 0).
Timing conv2d_mpi_local(const ConvParams& p, const float* slab, const float* weight,
                        const float* bias, float* local_out, Algo algo, MPI_Comm comm);
Timing conv2d_mpi_halo(const ConvParams& p, const float* owned, const float* weight,
                       const float* bias, float* local_out, Algo algo, MPI_Comm comm);

// Gathers every rank's [C_out][out_cnt][W_out] block into the full output at root
// (used for verification of the distributed-input modes, outside the timed region).
void gather_output_rows(const ConvParams& p, const float* local_out, float* output,
                        MPI_Comm comm, int root = 0);

// Deterministic synthetic input that any rank can generate for any row without
// holding the whole image: fills [C_in][rows][Wp] with padded rows
// [row0, row0 + rows) of the zero-padded image whose unpadded pixel (c, y, x)
// is synth_value(seed, (c*H + y)*W + x) in [-1, 1).
float synth_value(unsigned seed, long long index);
void make_padded_rows(const ConvParams& p, unsigned seed, int row0, int rows, float* dst);
// The whole unpadded image [C_in][H][W] with the same values.
std::vector<float> make_synthetic_input(const ConvParams& p, unsigned seed);

// ---------------------------------------------------------------------------
// Kernels (filters) used in experiments
// ---------------------------------------------------------------------------

// Named single-channel f x f filters: "edge" (Laplacian-like, slide 3), "sobel_x",
// "sobel_y", "box", "gaussian", "sharpen", "identity".  Fixed-size ones (edge,
// sobel, sharpen) require f == 3.  Returns false if the name/size is unsupported.
bool make_named_kernel(const std::string& name, int f, std::vector<float>* k);

// Fills `w` with uniform random numbers in [-1, 1) (deterministic for a seed).
void fill_random(std::vector<float>& w, unsigned seed);

}  // namespace conv
