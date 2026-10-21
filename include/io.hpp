// io.hpp - minimal image / array I/O (no external dependencies).
#pragma once

#include <string>
#include <vector>

namespace io {

// Reads a binary (P5) or ASCII (P2) grayscale PGM. Pixels are scaled to [0, 1].
bool read_pgm(const std::string& path, std::vector<float>* pixels, int* height, int* width);

// Writes a P5 PGM. If `abs_value` the magnitude is used (edge maps), then values
// are min-max normalised to 0..255.
bool write_pgm(const std::string& path, const float* pixels, int height, int width, bool abs_value);

// Writes a float32 C-order NumPy .npy file (for verification against PyTorch/SciPy).
bool write_npy(const std::string& path, const float* data, const std::vector<long long>& shape);

// Reads a float32 C-order .npy file written by write_npy() or numpy.save().
bool read_npy(const std::string& path, std::vector<float>* data, std::vector<long long>* shape);

}  // namespace io
