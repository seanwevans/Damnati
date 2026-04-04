#include "reporting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "strategy.hpp"

std::vector<std::pair<int, long long>>
sorted_agent_scores(const long long *scores, int count) {
  std::vector<std::pair<int, long long>> ranked;
  if (count <= 0 || scores == nullptr) {
    return ranked;
  }
  ranked.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    ranked.emplace_back(i, scores[i]);
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second > rhs.second;
    }
    return lhs.first < rhs.first;
  });
  return ranked;
}

const char kSummaryHeaderFmt[] =
    "{\"agents\":%d,\"rounds\":%d,\"p_ngram\":%.3f,\"depth\":%d,"
    "\"epsilon\":%.3f,\"gtft\":%.3f,\n";

void print_summary_report(const Config &cfg,
                          const std::vector<AgentParams> &hparams,
                          const long long *scores) {
  const int n = static_cast<int>(hparams.size());
  long long total = 0;
  long long minv = std::numeric_limits<long long>::max();
  long long maxv = std::numeric_limits<long long>::min();
  for (int i = 0; i < n; ++i) {
    total += scores[i];
    minv = std::min(minv, scores[i]);
    maxv = std::max(maxv, scores[i]);
  }

  long double mean = static_cast<long double>(total) /
                     static_cast<long double>(std::max(1, n));
  long double varacc = 0.0L;
  for (int i = 0; i < n; ++i) {
    long double d = static_cast<long double>(scores[i]) - mean;
    varacc += d * d;
  }
  long double variance = varacc / static_cast<long double>(n > 1 ? (n - 1) : 1);
  double stdev = std::sqrt(static_cast<double>(variance));

  std::printf(kSummaryHeaderFmt, n, cfg.rounds, cfg.p_ngram, cfg.depth,
              cfg.epsilon, cfg.gtft_p);
  std::printf(" \"avg_score\":%.3f,\"min\":%lld,\"max\":%lld,\"stdev\":%.3f,\n",
              static_cast<double>(mean), minv, maxv, stdev);

  constexpr std::size_t strategy_count = static_cast<std::size_t>(NGRAM) + 1;
  std::array<long double, strategy_count> sum_by{};
  std::array<int, strategy_count> cnt_by{};
  for (int i = 0; i < n; ++i) {
    sum_by[hparams[i].strat] += static_cast<long double>(scores[i]);
    cnt_by[hparams[i].strat]++;
  }

  std::printf(" \"by_strategy\":{");
  const char *names[] = {"AC",     "AD",     "TFT",  "GTFT", "GRIM",
                         "RANDOM", "PAVLOV", "ALT",  "JOSS", "TESTER",
                         "REPEAT", "S_TFT",  "NGRAM"};
  constexpr std::size_t names_count = sizeof(names) / sizeof(names[0]);
  static_assert(strategy_count == names_count,
                "Strategy names array must match strategy count");
  bool first = true;
  for (std::size_t s = 0; s < strategy_count; ++s) {
    if (cnt_by[s] == 0)
      continue;
    if (!first) {
      std::printf(",");
    }
    long double denom = static_cast<long double>(std::max(1, cnt_by[s]));
    std::printf("\"%s\":{\"mean\":%.3f,\"count\":%d}", names[s],
                static_cast<double>(sum_by[s] / denom), cnt_by[s]);
    first = false;
  }
  std::printf("}}\n");

  int show = dmin(10, n);
  std::vector<std::pair<int, long long>> ranked = sorted_agent_scores(scores, n);
  for (int r = 0; r < show && r < static_cast<int>(ranked.size()); ++r) {
    const auto &entry = ranked[r];
    int idx = entry.first;
    std::printf("agent[%d]: strat=%s score=%lld\n", idx,
                names[hparams[idx].strat], entry.second);
  }
}
