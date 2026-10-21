// conv2d_dist.cpp - distributed-input mode: the feature map already lives on
// the ranks (as it does between the layers of a CNN), so there is no root and
// no scatter/gather in the timed region.
//
//   padded input rows                 rank r owns   rank r needs
//   0  +----------------------+
//      | owned by rank 0      |  <- own_0         need_0 = own_0 + halo
//      +----------------------+                    |
//      | owned by rank 1      |  <- own_1 ... (f - s) halo rows of rank 0
//      |                      |                    come from the top of own_1
//      +----------------------+
//
// Variant (a) conv2d_mpi_local: each rank already holds need_r (halo included).
// Variant (b) conv2d_mpi_halo : each rank holds own_r and receives its halo from
//             the next rank(s) with MPI_Sendrecv (one neighbour unless blocks are
//             thinner than the halo).
#include "conv2d.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace conv {

RowLayout row_layout(const ConvParams& p, int P, int r) {
    const int Hp = p.padded_h(), s = p.stride;
    auto own_start = [&](int q) {
        int st, cnt;
        block_range(p.out_h(), P, q, &st, &cnt);
        return std::min(st * s, Hp);
    };
    RowLayout L;
    block_range(p.out_h(), P, r, &L.out_start, &L.out_cnt);
    L.own_start = own_start(r);
    L.own_end = r == P - 1 ? Hp : own_start(r + 1);
    if (L.out_cnt > 0) {
        L.need_start = L.out_start * s;
        L.need_end = (L.out_start + L.out_cnt - 1) * s + p.kernel;
    } else {
        L.need_start = L.need_end = L.own_start;
    }
    return L;
}

Timing conv2d_mpi_local(const ConvParams& p, const float* slab, const float* weight,
                        const float* bias, float* local_out, Algo algo, MPI_Comm comm) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    const RowLayout L = row_layout(p, P, rank);

    Timing t;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    conv2d_slab(p, slab, L.need_rows(), L.out_cnt, weight, bias, local_out, algo);
    const double t1 = MPI_Wtime();
    t.compute = t1 - t0;
    t.total = t.compute;
    return t;
}

namespace {

// [Cin] blocks of `rows` rows each, `pitch_rows` rows apart (one strided channel stack).
MPI_Datatype rows_type(int Cin, int rows, int pitch_rows, int Wp) {
    MPI_Datatype t;
    MPI_Type_vector(Cin, rows * Wp, pitch_rows * Wp, MPI_FLOAT, &t);
    MPI_Type_commit(&t);
    return t;
}

}  // namespace

Timing conv2d_mpi_halo(const ConvParams& p, const float* owned, const float* weight,
                       const float* bias, float* local_out, Algo algo, MPI_Comm comm) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    const int Cin = p.in_channels, Wp = p.padded_w();

    std::vector<RowLayout> lay(P);
    for (int r = 0; r < P; ++r) lay[r] = row_layout(p, P, r);
    const RowLayout& L = lay[rank];

    // Number of exchange rounds: how many ranks further down a halo can reach
    // (1 in the normal case; more only when blocks are thinner than f - s).
    int rounds = 0;
    for (int r = 0; r < P; ++r)
        for (int q = r + 1; q < P && lay[q].own_start < lay[r].need_end; ++q)
            if (lay[q].own_end > lay[q].own_start) rounds = std::max(rounds, q - r);

    std::vector<float> slab(static_cast<size_t>(Cin) * L.need_rows() * Wp);

    Timing t;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    // Own part of the needed rows (need_start == own_start for active ranks).
    const int local_rows = std::max(0, std::min(L.need_end, L.own_end) - L.need_start);
    for (int ci = 0; ci < Cin; ++ci)
        std::memcpy(slab.data() + static_cast<size_t>(ci) * L.need_rows() * Wp,
                    owned + (static_cast<size_t>(ci) * L.own_rows() + (L.need_start - L.own_start)) * Wp,
                    sizeof(float) * local_rows * Wp);

    // Round d: send my top rows to rank - d, receive rows below my block from rank + d.
    for (int d = 1; d <= rounds; ++d) {
        const int dst = rank - d >= 0 ? rank - d : MPI_PROC_NULL;
        const int src = rank + d < P ? rank + d : MPI_PROC_NULL;

        int s0 = 0, s1 = 0;  // rows of mine that dst needs
        if (dst != MPI_PROC_NULL) {
            s0 = std::max(L.own_start, lay[dst].need_start);
            s1 = std::min(L.own_end, lay[dst].need_end);
        }
        int r0 = 0, r1 = 0;  // rows of src that I need
        if (src != MPI_PROC_NULL) {
            r0 = std::max(lay[src].own_start, L.need_start);
            r1 = std::min(lay[src].own_end, L.need_end);
        }
        const int ns = std::max(0, s1 - s0), nr = std::max(0, r1 - r0);
        MPI_Datatype st = rows_type(Cin, std::max(ns, 1), std::max(L.own_rows(), 1), Wp);
        MPI_Datatype rt = rows_type(Cin, std::max(nr, 1), std::max(L.need_rows(), 1), Wp);
        const float* sbuf = ns ? owned + static_cast<size_t>(s0 - L.own_start) * Wp : owned;
        float* rbuf = nr ? slab.data() + static_cast<size_t>(r0 - L.need_start) * Wp : slab.data();
        MPI_Sendrecv(sbuf, ns ? 1 : 0, st, dst, d, rbuf, nr ? 1 : 0, rt, src, d, comm,
                     MPI_STATUS_IGNORE);
        MPI_Type_free(&st);
        MPI_Type_free(&rt);
    }
    const double t1 = MPI_Wtime();

    conv2d_slab(p, slab.data(), L.need_rows(), L.out_cnt, weight, bias, local_out, algo);
    const double t2 = MPI_Wtime();

    t.distribute = t1 - t0;
    t.compute = t2 - t1;
    t.total = t2 - t0;
    return t;
}

void gather_output_rows(const ConvParams& p, const float* local_out, float* output,
                        MPI_Comm comm, int root) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    const int Cout = p.out_channels, Ho = p.out_h(), Wo = p.out_w();
    std::vector<int> row_start(P), row_cnt(P), cnt(P), disp(P);
    long long total = 0;
    for (int r = 0; r < P; ++r) {
        block_range(Ho, P, r, &row_start[r], &row_cnt[r]);
        cnt[r] = Cout * row_cnt[r] * Wo;
        disp[r] = static_cast<int>(total);
        total += cnt[r];
    }
    if (Cout == 1) {  // packed layout == final layout
        MPI_Gatherv(local_out, cnt[rank], MPI_FLOAT, output, cnt.data(), disp.data(), MPI_FLOAT,
                    root, comm);
        return;
    }
    std::vector<float> recvbuf(rank == root ? total : 0);
    MPI_Gatherv(local_out, cnt[rank], MPI_FLOAT, recvbuf.data(), cnt.data(), disp.data(), MPI_FLOAT,
                root, comm);
    if (rank != root) return;
    for (int r = 0; r < P; ++r)
        for (int co = 0; co < Cout; ++co)
            std::memcpy(output + (static_cast<size_t>(co) * Ho + row_start[r]) * Wo,
                        recvbuf.data() + disp[r] + static_cast<size_t>(co) * row_cnt[r] * Wo,
                        sizeof(float) * row_cnt[r] * Wo);
}

float synth_value(unsigned seed, long long index) {
    // splitmix64 hash -> uniform float in [-1, 1)
    uint64_t z = (static_cast<uint64_t>(seed) << 40) ^ static_cast<uint64_t>(index);
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

void make_padded_rows(const ConvParams& p, unsigned seed, int row0, int rows, float* dst) {
    const int H = p.height, W = p.width, pad = p.padding, Wp = p.padded_w();
    for (int c = 0; c < p.in_channels; ++c)
        for (int i = 0; i < rows; ++i) {
            float* d = dst + (static_cast<size_t>(c) * rows + i) * Wp;
            const int y = row0 + i - pad;
            if (y < 0 || y >= H) {
                std::fill(d, d + Wp, 0.0f);
                continue;
            }
            std::fill(d, d + pad, 0.0f);
            std::fill(d + pad + W, d + Wp, 0.0f);
            const long long base = (static_cast<long long>(c) * H + y) * W;
            for (int x = 0; x < W; ++x) d[pad + x] = synth_value(seed, base + x);
        }
}

std::vector<float> make_synthetic_input(const ConvParams& p, unsigned seed) {
    std::vector<float> v(p.input_size());
    for (long long i = 0; i < p.input_size(); ++i) v[i] = synth_value(seed, i);
    return v;
}

}  // namespace conv
