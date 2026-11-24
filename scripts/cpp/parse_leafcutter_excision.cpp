// High-speed OpenMP streaming aggregator for SheepGTEx_v0.Splicing.excision.gz.
// Computes Half-A and Half-B (num, den, n_samples_den_ge_5) across all 15 OviSplice heads
// for chr1..chr26 and chrX, strictly excluding the 23 LOCO samples.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

namespace {

constexpr int N_HEADS = 15;
constexpr size_t BATCH_SIZE = 4096;

struct SampleMeta {
    bool active = false;
    uint8_t half = 0;   // 0=A, 1=B
    int8_t h1 = -1;     // 1..12 main tissue group head
    int8_t h2 = -1;     // 13..14 muscle stage head
};

struct RowAgg {
    bool keep = false;
    int16_t chrom_id = 0;
    int32_t i_s = 0;
    int32_t i_e = 0;
    std::string clu_id;
    int64_t num[2][N_HEADS] = {};
    int64_t den[2][N_HEADS] = {};
    uint16_t nge5[2][N_HEADS] = {};
};

int head_index_for_group(const std::string& g) {
    if (g == "SkeletalMuscle") return 1;
    if (g == "Liver") return 2;
    if (g == "Adipose") return 3;
    if (g == "SmallIntestine") return 4;
    if (g == "CNS") return 5;
    if (g == "Rumen") return 6;
    if (g == "Pituitary") return 7;
    if (g == "Lung") return 8;
    if (g == "Abomasum") return 9;
    if (g == "Gonad") return 10;
    if (g == "Heart") return 11;
    if (g == "Skin") return 12;
    return -1;
}

int head_index_for_stage(const std::string& s) {
    if (s == "SkeletalMuscle_Prenatal") return 13;
    if (s == "SkeletalMuscle_Postnatal") return 14;
    return -1;
}

int parse_main_chrom(const char* s, size_t len) {
    if (len < 4 || s[0] != 'c' || s[1] != 'h' || s[2] != 'r') return -1;
    if (len == 4 && s[3] == 'X') return 27;
    if (len == 4 && s[3] >= '1' && s[3] <= '9') return s[3] - '0';
    if (len == 5 && s[3] >= '1' && s[3] <= '2' && s[4] >= '0' && s[4] <= '9') {
        int v = (s[3] - '0') * 10 + (s[4] - '0');
        return (v >= 1 && v <= 26) ? v : -1;
    }
    return -1;
}

std::vector<std::string> split_tab_preserve_empty(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    size_t clean_end = line.size();
    while (clean_end > 0 && (line[clean_end - 1] == '\r' || line[clean_end - 1] == '\n')) {
        --clean_end;
    }
    while (start <= clean_end) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos || tab >= clean_end) {
            out.emplace_back(line.substr(start, clean_end - start));
            break;
        }
        out.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <sample_table.tsv> <out_dir>\n", argv[0]);
        return 2;
    }
    const std::string sample_table_path = argv[1];
    const std::string out_dir = argv[2];
    auto t0 = std::chrono::steady_clock::now();

    // 1. Load sample_table.tsv
    std::unordered_map<std::string, SampleMeta> sample_map;
    size_t st_total = 0, st_excluded = 0, st_kept = 0;
    {
        std::ifstream fin(sample_table_path);
        if (!fin) {
            std::fprintf(stderr, "Failed to open %s\n", sample_table_path.c_str());
            return 1;
        }
        std::string hdr_line;
        if (!std::getline(fin, hdr_line)) return 1;
        auto cols = split_tab_preserve_empty(hdr_line);
        int idx_sample = -1, idx_grp = -1, idx_stg = -1, idx_excl = -1, idx_half = -1;
        for (int i = 0; i < (int)cols.size(); ++i) {
            if (cols[i] == "Sample") idx_sample = i;
            else if (cols[i] == "tissue_group") idx_grp = i;
            else if (cols[i] == "muscle_stage_head") idx_stg = i;
            else if (cols[i] == "exclude_loco") idx_excl = i;
            else if (cols[i] == "split_half") idx_half = i;
        }
        if (idx_sample < 0 || idx_grp < 0 || idx_stg < 0 || idx_excl < 0 || idx_half < 0) {
            std::fprintf(stderr, "Missing expected columns in sample_table.tsv\n");
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            auto f = split_tab_preserve_empty(line);
            if ((int)f.size() <= idx_half) continue;
            ++st_total;
            SampleMeta sm;
            int excl = std::atoi(f[idx_excl].c_str());
            if (excl == 1) {
                sm.active = false;
                ++st_excluded;
            } else {
                sm.active = true;
                sm.half = (f[idx_half] == "B") ? 1 : 0;
                sm.h1 = (int8_t)head_index_for_group(f[idx_grp]);
                sm.h2 = (int8_t)head_index_for_stage(f[idx_stg]);
                ++st_kept;
            }
            sample_map[f[idx_sample]] = sm;
        }
    }
    if (st_total != 8171 || st_excluded != 23 || st_kept != 8148) {
        std::fprintf(stderr, "Unexpected sample_table counts: total=%zu excl=%zu kept=%zu\n",
                     st_total, st_excluded, st_kept);
        return 1;
    }

    // 2. Read header from stdin (pigz -dc stream)
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    static char io_buf[8 * 1024 * 1024];
    std::cin.rdbuf()->pubsetbuf(io_buf, sizeof(io_buf));

    std::string header_line;
    if (!std::getline(std::cin, header_line)) {
        std::fprintf(stderr, "Empty stdin stream\n");
        return 1;
    }
    std::vector<SampleMeta> col_meta;
    size_t overlap_header = 0, active_cols = 0, skipped_loco_cols = 0;
    {
        const char* p = header_line.data();
        const char* end = p + header_line.size();
        while (end > p && (end[-1] == '\r' || end[-1] == '\n')) --end;
        bool first_tok = true;
        while (p < end) {
            while (p < end && (*p == ' ' || *p == '\t')) ++p;
            if (p >= end) break;
            const char* q = p;
            while (q < end && *q != ' ' && *q != '\t') ++q;
            if (first_tok) {
                first_tok = false;
            } else {
                std::string sname(p, q - p);
                auto it = sample_map.find(sname);
                if (it != sample_map.end()) {
                    ++overlap_header;
                    col_meta.push_back(it->second);
                    if (it->second.active) ++active_cols;
                    else ++skipped_loco_cols;
                } else {
                    col_meta.push_back(SampleMeta{});
                }
            }
            p = q;
        }
    }
    const size_t n_cols = col_meta.size();
    if (n_cols != 8171 || overlap_header != 8171 || active_cols != 8148 || skipped_loco_cols != 23) {
        std::fprintf(stderr, "Header mismatch: cols=%zu overlap=%zu active=%zu loco=%zu\n",
                     n_cols, overlap_header, active_cols, skipped_loco_cols);
        return 1;
    }

    // 3. Open output files
    FILE* f_meta = std::fopen((out_dir + "/intron_meta.tsv").c_str(), "w");
    FILE* f_num  = std::fopen((out_dir + "/intron_num_ab.bin").c_str(), "wb");
    FILE* f_den  = std::fopen((out_dir + "/intron_den_ab.bin").c_str(), "wb");
    FILE* f_nge5 = std::fopen((out_dir + "/intron_nge5_ab.bin").c_str(), "wb");
    if (!f_meta || !f_num || !f_den || !f_nge5) {
        std::fprintf(stderr, "Failed to open output files in %s\n", out_dir.c_str());
        return 1;
    }
    std::fprintf(f_meta, "chrom_id\ti_s\ti_e\tclu_id\n");

    size_t total_rows_seen = 0;
    size_t skipped_non_main = 0;
    size_t kept_main_rows = 0;
    uint64_t parse_errors = 0;
    uint64_t col_mismatch_errors = 0;

    std::vector<std::string> batch;
    batch.reserve(BATCH_SIZE);
    std::vector<RowAgg> results;

    auto process_batch = [&]() {
        const size_t n = batch.size();
        results.assign(n, RowAgg{});
        uint64_t b_parse_err = 0;
        uint64_t b_col_err = 0;

        #pragma omp parallel for schedule(dynamic, 32) reduction(+:b_parse_err, b_col_err)
        for (size_t k = 0; k < n; ++k) {
            const std::string& ln = batch[k];
            const char* p = ln.data();
            const char* end = p + ln.size();
            while (end > p && (end[-1] == '\r' || end[-1] == '\n')) --end;
            while (p < end && (*p == ' ' || *p == '\t')) ++p;
            if (p >= end) continue;

            // Parse col 0: chrom:start:end:clu_id
            const char* iid_end = p;
            while (iid_end < end && *iid_end != ' ' && *iid_end != '\t') ++iid_end;
            const char* c1 = (const char*)std::memchr(p, ':', iid_end - p);
            if (!c1) { ++b_parse_err; continue; }
            int cid = parse_main_chrom(p, c1 - p);
            if (cid < 0) continue;  // skip NW_... scaffold quickly

            const char* c2 = (const char*)std::memchr(c1 + 1, ':', iid_end - (c1 + 1));
            const char* c3 = c2 ? (const char*)std::memchr(c2 + 1, ':', iid_end - (c2 + 1)) : nullptr;
            if (!c2 || !c3) { ++b_parse_err; continue; }

            int32_t i_s = 0, i_e = 0;
            for (const char* q = c1 + 1; q < c2; ++q) i_s = i_s * 10 + (*q - '0');
            for (const char* q = c2 + 1; q < c3; ++q) i_e = i_e * 10 + (*q - '0');

            RowAgg& ra = results[k];
            ra.keep = true;
            ra.chrom_id = (int16_t)cid;
            ra.i_s = i_s;
            ra.i_e = i_e;
            ra.clu_id.assign(c3 + 1, iid_end - (c3 + 1));

            p = iid_end;
            size_t col = 0;
            while (p < end && col < n_cols) {
                while (p < end && (*p == ' ' || *p == '\t')) ++p;
                if (p >= end) break;

                // Fast path 1: exact '0/0'
                if (p + 2 < end && p[0] == '0' && p[1] == '/' && p[2] == '0' &&
                    (p + 3 == end || p[3] == ' ' || p[3] == '\t')) {
                    p += 3;
                    ++col;
                    continue;
                }
                const SampleMeta& cm = col_meta[col];
                // Fast path 2: excluded LOCO sample
                if (!cm.active) {
                    while (p < end && *p != ' ' && *p != '\t') ++p;
                    ++col;
                    continue;
                }

                uint64_t num = 0, den = 0;
                const char* q = p;
                while (q < end && *q >= '0' && *q <= '9') {
                    num = num * 10 + (uint64_t)(*q - '0');
                    ++q;
                }
                if (q == p || q >= end || *q != '/') {
                    ++b_parse_err;
                    while (q < end && *q != ' ' && *q != '\t') ++q;
                    p = q;
                    ++col;
                    continue;
                }
                ++q;
                const char* d_start = q;
                while (q < end && *q >= '0' && *q <= '9') {
                    den = den * 10 + (uint64_t)(*q - '0');
                    ++q;
                }
                if (q == d_start || (q < end && *q != ' ' && *q != '\t')) {
                    ++b_parse_err;
                    while (q < end && *q != ' ' && *q != '\t') ++q;
                    p = q;
                    ++col;
                    continue;
                }
                p = q;
                ++col;

                const uint8_t hlf = cm.half;
                const uint16_t ge5 = (den >= 5) ? 1 : 0;
                ra.num[hlf][0] += (int64_t)num;
                ra.den[hlf][0] += (int64_t)den;
                ra.nge5[hlf][0] += ge5;
                if (cm.h1 >= 0) {
                    ra.num[hlf][cm.h1] += (int64_t)num;
                    ra.den[hlf][cm.h1] += (int64_t)den;
                    ra.nge5[hlf][cm.h1] += ge5;
                }
                if (cm.h2 >= 0) {
                    ra.num[hlf][cm.h2] += (int64_t)num;
                    ra.den[hlf][cm.h2] += (int64_t)den;
                    ra.nge5[hlf][cm.h2] += ge5;
                }
            }
            if (col != n_cols) {
                ++b_col_err;
            }
        }

        parse_errors += b_parse_err;
        col_mismatch_errors += b_col_err;
        for (size_t k = 0; k < n; ++k) {
            const RowAgg& ra = results[k];
            if (!ra.keep) {
                ++skipped_non_main;
                continue;
            }
            ++kept_main_rows;
            std::fprintf(f_meta, "%d\t%d\t%d\t%s\n",
                         (int)ra.chrom_id, ra.i_s, ra.i_e, ra.clu_id.c_str());
            std::fwrite(ra.num, sizeof(int64_t), 2 * N_HEADS, f_num);
            std::fwrite(ra.den, sizeof(int64_t), 2 * N_HEADS, f_den);
            std::fwrite(ra.nge5, sizeof(uint16_t), 2 * N_HEADS, f_nge5);
        }
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        ++total_rows_seen;
        batch.push_back(std::move(line));
        if (batch.size() >= BATCH_SIZE) {
            process_batch();
            batch.clear();
            if (total_rows_seen % 200000 < BATCH_SIZE) {
                std::fprintf(stderr, "[ExcisionParser] processed=%zu kept=%zu skipped=%zu\n",
                             total_rows_seen, kept_main_rows, skipped_non_main);
            }
        }
    }
    if (!batch.empty()) {
        process_batch();
        batch.clear();
    }

    std::fclose(f_meta);
    std::fclose(f_num);
    std::fclose(f_den);
    std::fclose(f_nge5);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    FILE* f_stat = std::fopen((out_dir + "/parse_stats.json").c_str(), "w");
    std::fprintf(f_stat,
        "{\n"
        "  \"sample_table_total\": %zu,\n"
        "  \"sample_table_excluded_loco\": %zu,\n"
        "  \"sample_table_kept\": %zu,\n"
        "  \"excision_header_samples\": %zu,\n"
        "  \"header_overlap_sample_table\": %zu,\n"
        "  \"active_cols\": %zu,\n"
        "  \"loco_cols_skipped\": %zu,\n"
        "  \"total_rows_seen\": %zu,\n"
        "  \"kept_main_chr_rows\": %zu,\n"
        "  \"skipped_non_main_chr\": %zu,\n"
        "  \"parse_errors\": %llu,\n"
        "  \"col_mismatch_errors\": %llu,\n"
        "  \"elapsed_s\": %.2f\n"
        "}\n",
        st_total, st_excluded, st_kept, n_cols, overlap_header, active_cols, skipped_loco_cols,
        total_rows_seen, kept_main_rows, skipped_non_main,
        (unsigned long long)parse_errors, (unsigned long long)col_mismatch_errors, elapsed_s);
    std::fclose(f_stat);

    std::printf("[ExcisionParser DONE] rows=%zu kept_main=%zu skipped=%zu parse_err=%llu col_err=%llu elapsed=%.2fs\n",
                total_rows_seen, kept_main_rows, skipped_non_main,
                (unsigned long long)parse_errors, (unsigned long long)col_mismatch_errors, elapsed_s);
    return (parse_errors == 0 && col_mismatch_errors == 0) ? 0 : 3;
}
