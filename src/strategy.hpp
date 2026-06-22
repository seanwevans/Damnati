#pragma once

#include <cstddef>
#include <cstdint>

#include "core.hpp"

__device__ __host__ int dmin(int a, int b);
__device__ __host__ int dmax(int a, int b);
template <typename T> __device__ __host__ T dclamp(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __host__ std::size_t ngram_span(int depth);
__device__ __host__ uint64_t mix64(uint64_t x);
__device__ __host__ float rng01(uint64_t seed, int i, int j, int r, int who);

struct PlayerState {
  int strat;
  int last = -1;
  int opp_last = -1;
  int defect_seen = 0;
  int depth = 0;
  float epsilon = 0.0f;
  unsigned int state = 0;
  int *counts = nullptr;
  float *q = nullptr;
  float gtft_forget = 0.1f;

  __device__ void init_ngram(int d, float eps, int *cbuf, float *qbuf);
};

__device__ __host__ int encode_pair(int my, int opp);
__device__ __host__ int ngram_choose(PlayerState &p, uint64_t seed, int i,
                                     int j, int r, int who);
__device__ __host__ void ngram_update(PlayerState &p, int my, int opp,
                                      int reward);
__device__ __host__ int choose_action(PlayerState &p, uint64_t seed, int i,
                                      int j, int r, int who);
