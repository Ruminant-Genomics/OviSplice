// Ultra-fast zero-RAM streaming UCSC over.chain liftover (<3s for 80 MB gzipped chain).
// Usage: gzip -dc file.over.chain.gz | fast_chain_lift <t_map.tsv> <q_map.tsv_or_ANYCHR> <queries.tsv> <out.tsv>
// queries.tsv lines: dir(0=fwd,1=rev) req3(0/1) cid pos0 strand
// out.tsv lines (same order): out_cid out_pos0 out_strand
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct QRec {
    int pos;
    int idx;
    int strand;
    int req3;
};

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    std::unordered_map<std::string, int> t_map, q_map;
    bool q_any_chr = (std::string(argv[2]) == "ANYCHR");
    {
        FILE* f = std::fopen(argv[1], "r");
        char buf[256]; int id;
        while (std::fscanf(f, "%255s %d", buf, &id) == 2) t_map[buf] = id;
        std::fclose(f);
    }
    if (!q_any_chr) {
        FILE* f = std::fopen(argv[2], "r");
        char buf[256]; int id;
        while (std::fscanf(f, "%255s %d", buf, &id) == 2) q_map[buf] = id;
        std::fclose(f);
    }

    std::vector<std::vector<QRec>> fwd_q(35), rev_q(35);
    int n_q = 0;
    {
        FILE* f = std::fopen(argv[3], "r");
        int dir, req3, cid, pos, st;
        while (std::fscanf(f, "%d %d %d %d %d", &dir, &req3, &cid, &pos, &st) == 5) {
            if (cid > 0 && cid < 35) {
                QRec r{pos, n_q, st, req3};
                if (dir == 0) fwd_q[cid].push_back(r);
                else rev_q[cid].push_back(r);
            }
            n_q++;
        }
        std::fclose(f);
    }
    auto cmp_q = [](const QRec& a, const QRec& b) { return a.pos < b.pos; };
    for (auto& v : fwd_q) std::sort(v.begin(), v.end(), cmp_q);
    for (auto& v : rev_q) std::sort(v.begin(), v.end(), cmp_q);

    std::vector<double> best_sc(n_q, -1.0);
    std::vector<int> out_c(n_q, 0), out_p(n_q, -1), out_s(n_q, 0);

    static char line[4096];
    int t_cid = 0, q_cid = 0, q_size = 0, t_cur = 0, q_cur = 0;
    double score = 0.0;
    bool q_plus = true, active = false;

    while (std::fgets(line, sizeof(line), stdin)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '#') continue;
        if (line[0] == 'c' && std::strncmp(line, "chain ", 6) == 0) {
            char tName[256], qName[256], tStr[4], qStr[4];
            int tSize, tStart, tEnd, qStart, qEnd;
            if (std::sscanf(line + 6, "%lf %255s %d %3s %d %d %255s %d %3s %d %d",
                            &score, tName, &tSize, tStr, &tStart, &tEnd,
                            qName, &q_size, qStr, &qStart, &qEnd) >= 11) {
                auto it_t = t_map.find(tName);
                t_cid = (it_t != t_map.end()) ? it_t->second : 0;
                if (q_any_chr) {
                    q_cid = (std::strncmp(qName, "chr", 3) == 0) ? 1 : 0;
                } else {
                    auto it_q = q_map.find(qName);
                    q_cid = (it_q != q_map.end()) ? it_q->second : 0;
                }
                active = (t_cid > 0 && q_cid > 0);
                if (active) {
                    q_plus = (qStr[0] == '+');
                    t_cur = tStart;
                    q_cur = qStart;
                }
            } else {
                active = false;
            }
        } else if (active) {
            char* p = line;
            int sz = static_cast<int>(std::strtol(p, &p, 10));
            if (sz > 0) {
                int t0 = t_cur, t1 = t_cur + sz;
                // Forward queries on t_cid
                const auto& fv = fwd_q[t_cid];
                if (!fv.empty()) {
                    auto it = std::lower_bound(fv.begin(), fv.end(), t0,
                                               [](const QRec& r, int v) { return r.pos < v; });
                    while (it != fv.end() && it->pos < t1) {
                        if (!it->req3 || (it->pos >= t0 + 2 && it->pos + 3 <= t1)) {
                            int oi = it->idx;
                            if (score > best_sc[oi]) {
                                best_sc[oi] = score;
                                int qs = q_cur + (it->pos - t0);
                                out_c[oi] = q_cid;
                                if (q_plus) {
                                    out_p[oi] = qs;
                                    out_s[oi] = it->strand;
                                } else {
                                    out_p[oi] = (q_size - 1) - qs;
                                    out_s[oi] = -it->strand;
                                }
                            }
                        }
                        ++it;
                    }
                }
                // Reverse queries on q_cid
                if (!q_any_chr) {
                    const auto& rv = rev_q[q_cid];
                    if (!rv.empty()) {
                        int qf0 = q_plus ? q_cur : (q_size - q_cur - sz);
                        int qf1 = qf0 + sz;
                        int t_base = q_plus ? t_cur : (t_cur + sz - 1);
                        int t_rel = q_plus ? 1 : -1;
                        auto it = std::lower_bound(rv.begin(), rv.end(), qf0,
                                                   [](const QRec& r, int v) { return r.pos < v; });
                        while (it != rv.end() && it->pos < qf1) {
                            int oi = it->idx;
                            if (score > best_sc[oi]) {
                                best_sc[oi] = score;
                                int off = it->pos - qf0;
                                out_c[oi] = t_cid;
                                out_p[oi] = (t_rel == 1) ? (t_base + off) : (t_base - off);
                                out_s[oi] = it->strand * t_rel;
                            }
                            ++it;
                        }
                    }
                }
            }
            while (*p == ' ' || *p == '\t') ++p;
            if (*p >= '0' && *p <= '9') {
                int dt = static_cast<int>(std::strtol(p, &p, 10));
                int dq = static_cast<int>(std::strtol(p, &p, 10));
                t_cur += sz + dt;
                q_cur += sz + dq;
            } else {
                active = false;
            }
        }
    }

    FILE* fout = std::fopen(argv[4], "w");
    for (int i = 0; i < n_q; ++i) {
        std::fprintf(fout, "%d\t%d\t%d\n", out_c[i], out_p[i], out_s[i]);
    }
    std::fclose(fout);
    return 0;
}
