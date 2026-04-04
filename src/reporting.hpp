#pragma once

#include <utility>
#include <vector>

#include "cli.hpp"
#include "core.hpp"

std::vector<std::pair<int, long long>>
sorted_agent_scores(const long long *scores, int count);

extern const char kSummaryHeaderFmt[];

void print_summary_report(const Config &cfg,
                          const std::vector<AgentParams> &hparams,
                          const long long *scores);
