// conv2d_mpi.cpp - distributed-memory convolution with MPI.
//
// Spatial decomposition (one large image):
//
//     padded input (root)                 output
//   +--------------------+            +-----------+
//   | rows for rank 0    |  -- r0 --> | block 0   |
//   |   ... halo (f-s) ..|            +-----------+
//   | rows for rank 1    |  -- r1 --> | block 1   |
//   |        ...         |            |    ...    |
//   +--------------------+            +-----------+
//
// 1. MPI_Bcast the kernel weights/bias (small).
// 2. Root pads the image and packs, for every rank, the contiguous input rows
//    [start*s, (start+cnt-1)*s + f) of every channel (the halo rows are
//    duplicated so that ranks never need to talk to each other).
// 3. MPI_Scatterv the slabs; every rank runs conv2d_slab() independently.
// 4. MPI_Gatherv the output blocks back to root.
//
// Batch decomposition (many images): whole images are scattered block-wise;
// no halo is needed so communication is just input + output volume.
#include "conv2d.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace conv {

namespace {

void check_int_range(long long v, const char* what) {
    if (v > std::numeric_limits<int>::max())
        throw std::overflow_error(std::string(what) + " exceeds MPI int count limit");
}

void bcast_weights(const ConvParams& p, const float* weight, const float* bias,
                   std::vector<float>& w, std::vector<float>& b, MPI_Comm comm, int root) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    w.resize(p.weight_size());
    if (rank == root) std::memcpy(w.data(), weight, sizeof(float) * w.size());
    MPI_Bcast(w.data(), static_cast<int>(w.size()), MPI_FLOAT, root, comm);
    if (p.use_bias) {
        b.resize(p.out_channels);
        if (rank == root) std::memcpy(b.data(), bias, sizeof(float) * b.size());
        MPI_Bcast(b.data(), p.out_channels, MPI_FLOAT, root, comm);
    }
}

}  // namespace

Timing conv2d_mpi(const ConvParams& p, const float* input, const float* weight,
                  const float* bias, float* output, Algo algo, MPI_Comm comm, int root) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    const int Cin = p.in_channels, Cout = p.out_channels;
    const int f = p.kernel, s = p.stride;
    const int Hp = p.padded_h(), Wp = p.padded_w(), Ho = p.out_h(), Wo = p.out_w();

    // Partition of output rows and the matching input slabs (identical on all ranks).
    std::vector<int> row_start(P), row_cnt(P), send_cnt(P), send_disp(P), recv_cnt(P), recv_disp(P);
    long long send_total = 0, recv_total = 0;
    for (int r = 0; r < P; ++r) {
        block_range(Ho, P, r, &row_start[r], &row_cnt[r]);
        const long long slab_rows = row_cnt[r] > 0 ? 1LL * (row_cnt[r] - 1) * s + f : 0;
        const long long sc = 1LL * Cin * slab_rows * Wp;
        const long long rc = 1LL * Cout * row_cnt[r] * Wo;
        check_int_range(send_total + sc, "input slabs");
        check_int_range(recv_total + rc, "output blocks");
        send_cnt[r] = static_cast<int>(sc);
        send_disp[r] = static_cast<int>(send_total);
        recv_cnt[r] = static_cast<int>(rc);
        recv_disp[r] = static_cast<int>(recv_total);
        send_total += sc;
        recv_total += rc;
    }

    Timing t;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    // ---- 1+2+3: distribute -------------------------------------------------
    std::vector<float> w, b;
    bcast_weights(p, weight, bias, w, b, comm, root);

    std::vector<float> sendbuf;
    if (rank == root) {
        std::vector<float> padded;
        const float* src = input;
        if (p.padding > 0) {
            padded = pad_input(input, Cin, p.height, p.width, p.padding);
            src = padded.data();
        }
        sendbuf.resize(send_total);
        for (int r = 0; r < P; ++r) {
            if (row_cnt[r] == 0) continue;
            const int in_start = row_start[r] * s;
            const int slab_rows = (row_cnt[r] - 1) * s + f;
            float* dst = sendbuf.data() + send_disp[r];
            for (int ci = 0; ci < Cin; ++ci)
                std::memcpy(dst + static_cast<size_t>(ci) * slab_rows * Wp,
                            src + (static_cast<size_t>(ci) * Hp + in_start) * Wp,
                            sizeof(float) * slab_rows * Wp);
        }
    }
    std::vector<float> slab(send_cnt[rank]);
    MPI_Scatterv(sendbuf.data(), send_cnt.data(), send_disp.data(), MPI_FLOAT,
                 slab.data(), send_cnt[rank], MPI_FLOAT, root, comm);
    const double t1 = MPI_Wtime();

    // ---- compute -----------------------------------------------------------
    std::vector<float> local_out(recv_cnt[rank]);
    const int my_slab_rows = row_cnt[rank] > 0 ? (row_cnt[rank] - 1) * s + f : 0;
    conv2d_slab(p, slab.data(), my_slab_rows, row_cnt[rank], w.data(), p.use_bias ? b.data() : nullptr,
                local_out.data(), algo);
    const double t2 = MPI_Wtime();

    // ---- 4: collect --------------------------------------------------------
    gather_output_rows(p, local_out.data(), output, comm, root);
    const double t3 = MPI_Wtime();

    t.distribute = t1 - t0;
    t.compute = t2 - t1;
    t.collect = t3 - t2;
    t.total = t3 - t0;
    return t;
}

Timing conv2d_mpi_batch(const ConvParams& p, int batch, const float* input, const float* weight,
                        const float* bias, float* output, Algo algo, MPI_Comm comm, int root) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    const long long in_sz = p.input_size(), out_sz = p.output_size();
    std::vector<int> img_start(P), img_cnt(P), send_cnt(P), send_disp(P), recv_cnt(P), recv_disp(P);
    for (int r = 0; r < P; ++r) {
        block_range(batch, P, r, &img_start[r], &img_cnt[r]);
        check_int_range(in_sz * (img_start[r] + img_cnt[r]), "batch input");
        check_int_range(out_sz * (img_start[r] + img_cnt[r]), "batch output");
        send_cnt[r] = static_cast<int>(in_sz * img_cnt[r]);
        send_disp[r] = static_cast<int>(in_sz * img_start[r]);
        recv_cnt[r] = static_cast<int>(out_sz * img_cnt[r]);
        recv_disp[r] = static_cast<int>(out_sz * img_start[r]);
    }

    Timing t;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    std::vector<float> w, b;
    bcast_weights(p, weight, bias, w, b, comm, root);
    std::vector<float> local_in(send_cnt[rank]);
    MPI_Scatterv(input, send_cnt.data(), send_disp.data(), MPI_FLOAT, local_in.data(),
                 send_cnt[rank], MPI_FLOAT, root, comm);
    const double t1 = MPI_Wtime();

    std::vector<float> local_out(recv_cnt[rank]);
    for (int i = 0; i < img_cnt[rank]; ++i)
        conv2d_serial(p, local_in.data() + i * in_sz, w.data(), p.use_bias ? b.data() : nullptr,
                      local_out.data() + i * out_sz, algo);
    const double t2 = MPI_Wtime();

    MPI_Gatherv(local_out.data(), recv_cnt[rank], MPI_FLOAT, output, recv_cnt.data(),
                recv_disp.data(), MPI_FLOAT, root, comm);
    const double t3 = MPI_Wtime();

    t.distribute = t1 - t0;
    t.compute = t2 - t1;
    t.collect = t3 - t2;
    t.total = t3 - t0;
    return t;
}

}  // namespace conv
