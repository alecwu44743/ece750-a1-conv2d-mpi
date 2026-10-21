// io.cpp - PGM and .npy readers/writers.
#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace io {

namespace {

// Reads the next whitespace-separated header token, skipping '#' comments.
bool next_token(std::istream& in, std::string* tok) {
    tok->clear();
    int c;
    while ((c = in.get()) != EOF) {
        if (c == '#') {
            while ((c = in.get()) != EOF && c != '\n') {}
        } else if (!std::isspace(c)) {
            tok->push_back(static_cast<char>(c));
            break;
        }
    }
    while ((c = in.peek()) != EOF && !std::isspace(c) && c != '#') tok->push_back(static_cast<char>(in.get()));
    return !tok->empty();
}

}  // namespace

bool read_pgm(const std::string& path, std::vector<float>* pixels, int* height, int* width) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string magic, ws, hs, ms;
    if (!next_token(in, &magic) || (magic != "P5" && magic != "P2")) return false;
    if (!next_token(in, &ws) || !next_token(in, &hs) || !next_token(in, &ms)) return false;
    const int W = std::stoi(ws), H = std::stoi(hs), maxval = std::stoi(ms);
    if (W <= 0 || H <= 0 || maxval <= 0 || maxval > 65535) return false;
    pixels->resize(static_cast<size_t>(W) * H);
    if (magic == "P5") {
        in.get();  // single whitespace after maxval
        const int bpp = maxval < 256 ? 1 : 2;
        std::vector<unsigned char> raw(static_cast<size_t>(W) * H * bpp);
        if (!in.read(reinterpret_cast<char*>(raw.data()), raw.size())) return false;
        for (size_t i = 0; i < pixels->size(); ++i) {
            const int v = bpp == 1 ? raw[i] : (raw[2 * i] << 8 | raw[2 * i + 1]);
            (*pixels)[i] = static_cast<float>(v) / maxval;
        }
    } else {
        for (size_t i = 0; i < pixels->size(); ++i) {
            int v;
            if (!(in >> v)) return false;
            (*pixels)[i] = static_cast<float>(v) / maxval;
        }
    }
    *height = H;
    *width = W;
    return true;
}

bool write_pgm(const std::string& path, const float* pixels, int height, int width, bool abs_value) {
    const size_t n = static_cast<size_t>(height) * width;
    auto val = [&](size_t i) { return abs_value ? std::fabs(pixels[i]) : pixels[i]; };
    float lo = val(0), hi = val(0);
    for (size_t i = 1; i < n; ++i) { lo = std::min(lo, val(i)); hi = std::max(hi, val(i)); }
    const float scale = hi > lo ? 255.0f / (hi - lo) : 0.0f;
    std::vector<unsigned char> raw(n);
    for (size_t i = 0; i < n; ++i)
        raw[i] = static_cast<unsigned char>(std::lround((val(i) - lo) * scale));
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(raw.data()), raw.size());
    return static_cast<bool>(out);
}

bool write_npy(const std::string& path, const float* data, const std::vector<long long>& shape) {
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    long long n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        hdr << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
        n *= shape[i];
    }
    hdr << "), }";
    std::string h = hdr.str();
    // magic(6) + version(2) + len(2) + header + '\n' must be a multiple of 64.
    const size_t total = 10 + h.size() + 1;
    h.append((64 - total % 64) % 64, ' ');
    h.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint16_t len = static_cast<uint16_t>(h.size());
    out.write("\x93NUMPY\x01\x00", 8);
    out.put(static_cast<char>(len & 0xff));
    out.put(static_cast<char>(len >> 8));
    out.write(h.data(), h.size());
    out.write(reinterpret_cast<const char*>(data), sizeof(float) * n);
    return static_cast<bool>(out);
}

bool read_npy(const std::string& path, std::vector<float>* data, std::vector<long long>* shape) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint32_t len = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        len = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        len = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    std::string h(len, '\0');
    if (!in.read(&h[0], len)) return false;
    if (h.find("'<f4'") == std::string::npos || h.find("'fortran_order': False") == std::string::npos)
        return false;
    const size_t a = h.find('(', h.find("'shape'")), b = h.find(')', a);
    shape->clear();
    std::stringstream ss(h.substr(a + 1, b - a - 1));
    std::string item;
    long long n = 1;
    while (std::getline(ss, item, ','))
        if (item.find_first_not_of(" ") != std::string::npos) {
            shape->push_back(std::stoll(item));
            n *= shape->back();
        }
    data->resize(n);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data->data()), sizeof(float) * n));
}

}  // namespace io
