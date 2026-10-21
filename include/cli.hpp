// cli.hpp - tiny "--key value" / "--flag" command-line parser and helpers.
#pragma once

#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cli {

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + a);
            a = a.substr(2);
            const size_t eq = a.find('=');
            if (eq != std::string::npos) {
                kv_[a.substr(0, eq)] = a.substr(eq + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                kv_[a] = argv[++i];
            } else {
                kv_[a] = "";
            }
        }
    }
    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    int num(const std::string& k, int def) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : std::stoi(it->second);
    }

private:
    std::map<std::string, std::string> kv_;
};

inline std::vector<std::string> split(const std::string& s, char sep = ',') {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

inline std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    for (const auto& x : split(s)) out.push_back(std::stoi(x));
    return out;
}

// Parses "HxW" or "N" (square).
inline void parse_size(const std::string& s, int* h, int* w) {
    const size_t x = s.find('x');
    if (x == std::string::npos) {
        *h = *w = std::stoi(s);
    } else {
        *h = std::stoi(s.substr(0, x));
        *w = std::stoi(s.substr(x + 1));
    }
}

}  // namespace cli
