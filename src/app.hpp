#pragma once

#include <vector>

#include "cli.hpp"
#include "core.hpp"

int build_population(const Config &cfg, std::vector<AgentParams> &hparams);
void run_gpu(const Config &cfg);
