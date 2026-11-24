// Fast C++17 streaming parser for CattleGTEx {tissue}_splicing_count.gct.gz
// Reads uncompressed GCT from stdin (`tar -xzOf ... | gzip -dc | ./parse_cattle_gct <out.tsv>`).
// Computes exact Half-A (even sample indices) & Half-B (odd sample indices) intron read sums
// AND exact per-cluster sample counts with cluster read denominator >= 5 (`nge5_A`, `nge5_B`),
// matching OviSplice `parse_leafcutter_excision.cpp` / `s01_build_sheep_15head_usage.py` 1:1.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct IntronRec {
    int16_t cid;
    int32_t i_s;
    int32_t i_e;
    uint32_t clu_idx;
    int64_t sum_a;
    int64_t sum_b;
};

static inline int16_t parse_cattle_cid(const char* s, size_t len) {
    if (len >= 3 && (s[0] == 'c' || s[0] == 'C') && s[1] == 'h' && s[2] == 'r') {
        s += 3;
        len -= 3;
    }
    if (len == 1 && (s[0] == 'X' || s[0] == 'x')) return 30;
    if (len == 1 && s[0] >= '1' && s[0] <= '9') return static_cast<int16_t>(s[0] - '0');
    if (len == 2 && s[0] >= '1' && s[0] <= '2' && s[1] >= '0' && s[1] <= '9') {
        int v = (s[0] - '0') * 10 + (s[1] - '0');
        return (v >= 1 && v <= 29) ? static_cast<int16_t>(v) : 0;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: parse_cattle_gct <out.tsv>\n");
        return 1;
    }
    const char* out_path = argv[1];

    // Large stdio buffer for fast pipe ingestion
    std::vector<char> io_buf(1 << 22);
    std::setvbuf(stdin, io_buf.data(), _IOFBF, io_buf.size());

    char* line = nullptr;
    size_t cap = 0;
    ssize_t nread = 0;

    // Line 1: #1.2
    if (getline(&line, &cap, stdin) < 0) return 1;
    // Line 2: nrows \t ncols
    if (getline(&line, &cap, stdin) < 0) return 1;
    // Line 3: NAME \t Description \t sample_0 \t sample_1 ...
    nread = getline(&line, &cap, stdin);
    if (nread < 0) return 1;

    int n_samples = 0;
    {
        int col = 0;
        for (ssize_t i = 0; i < nread; ++i) {
            if (line[i] == '\t') ++col;
        }
        // Total columns = col + 1; first 2 are NAME and Description
        n_samples = (col + 1) - 2;
    }
    if (n_samples <= 0) {
        std::fprintf(stderr, "Invalid n_samples=%d\n", n_samples);
        return 1;
    }

    std::unordered_map<std::string, uint32_t> clu_map;
    clu_map.reserve(131072);
    std::vector<std::string> clu_names;
    std::vector<std::vector<uint16_t>> clu_sample_den;
    std::vector<IntronRec> introns;
    introns.reserve(300000);

    std::vector<uint32_t> row_vals(n_samples);

    while ((nread = getline(&line, &cap, stdin)) > 0) {
        // Parse col 0: NAME = chr:i_s:i_e:clu_id
        const char* p = line;
        const char* end = line + nread;
        const char* tab1 = static_cast<const char*>(std::memchr(p, '\t', end - p));
        if (!tab1) continue;

        // Parse chr:i_s:i_e:clu_id inside [p, tab1)
        const char* c1 = static_cast<const char*>(std::memchr(p, ':', tab1 - p));
        if (!c1) continue;
        int16_t cid = parse_cattle_cid(p, c1 - p);
        if (cid == 0) continue;

        const char* c2 = static_cast<const char*>(std::memchr(c1 + 1, ':', tab1 - (c1 + 1)));
        if (!c2) continue;
        const char* c3 = static_cast<const char*>(std::memchr(c2 + 1, ':', tab1 - (c2 + 1)));
        if (!c3) continue;

        int32_t i_s = std::atoi(c1 + 1);
        int32_t i_e = std::atoi(c2 + 1);

        // Skip col 1 (Description)
        const char* tab2 = static_cast<const char*>(std::memchr(tab1 + 1, '\t', end - (tab1 + 1)));
        if (!tab2) continue;

        const char* cur = tab2 + 1;
        int64_t sum_a = 0;
        int64_t sum_b = 0;
        int s_idx = 0;
        while (cur < end && s_idx < n_samples) {
            uint32_t v = 0;
            while (cur < end && *cur >= '0' && *cur <= '9') {
                v = v * 10u + static_cast<uint32_t>(*cur - '0');
                ++cur;
            }
            row_vals[s_idx] = v;
            if ((s_idx & 1) == 0) sum_a += v;
            else sum_b += v;
            ++s_idx;
            if (cur < end && (*cur == '\t' || *cur == '\r' || *cur == '\n')) ++cur;
        }

        if (sum_a + sum_b == 0) continue;

        // Key cluster by cid + ":" + clu_id
        std::string clu_str(c3 + 1, tab1 - (c3 + 1));
        std::string full_clu = std::to_string(cid) + ":" + clu_str;
        uint32_t c_idx = 0;
        auto it = clu_map.find(full_clu);
        if (it == clu_map.end()) {
            c_idx = static_cast<uint32_t>(clu_names.size());
            clu_map.emplace(full_clu, c_idx);
            clu_names.push_back(clu_str);
            clu_sample_den.emplace_back(n_samples, 0);
        } else {
            c_idx = it->second;
        }

        uint16_t* den_vec = clu_sample_den[c_idx].data();
        for (int k = 0; k < s_idx; ++k) {
            uint32_t nv = static_cast<uint32_t>(den_vec[k]) + row_vals[k];
            den_vec[k] = (nv > 65535u) ? 65535u : static_cast<uint16_t>(nv);
        }

        introns.push_back(IntronRec{cid, i_s, i_e, c_idx, sum_a, sum_b});
    }
    std::free(line);

    // Precompute nge5_a and nge5_b per cluster
    size_t n_clu = clu_names.size();
    std::vector<int32_t> clu_nge5_a(n_clu, 0);
    std::vector<int32_t> clu_nge5_b(n_clu, 0);
    for (size_t ci = 0; ci < n_clu; ++ci) {
        const auto& dv = clu_sample_den[ci];
        int32_t na = 0, nb = 0;
        for (int k = 0; k < n_samples; ++k) {
            if (dv[k] >= 5) {
                if ((k & 1) == 0) ++na;
                else ++nb;
            }
        }
        clu_nge5_a[ci] = na;
        clu_nge5_b[ci] = nb;
    }

    FILE* fw = std::fopen(out_path, "wb");
    if (!fw) return 2;
    std::fprintf(fw, "cid\ti_s\ti_e\tclu_id\tn_samples\tsum_a\tsum_b\tnge5_a\tnge5_b\n");
    for (const auto& r : introns) {
        std::fprintf(
            fw, "%d\t%d\t%d\t%s\t%d\t%lld\t%lld\t%d\t%d\n",
            static_cast<int>(r.cid), r.i_s, r.i_e, clu_names[r.clu_idx].c_str(),
            n_samples, static_cast<long long>(r.sum_a), static_cast<long long>(r.sum_b),
            clu_nge5_a[r.clu_idx], clu_nge5_b[r.clu_idx]
        );
    }
    std::fclose(fw);
    return 0;
}
