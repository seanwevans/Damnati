#pragma once

#include <cstdint>
#include <cstdio>

constexpr int MAX_NGRAM_DEPTH = 15;

struct Config {
  int n_agents = 256;
  int rounds = 200;
  uint64_t seed = 1234567ULL;
  float p_ngram = 0.5f;
  int depth = 3;
  float epsilon = 0.1f;
  float gtft_p = 0.1f;
};

void print_usage(FILE *stream, const char *prog);
void parse_cli(int argc, char **argv, Config &cfg);
