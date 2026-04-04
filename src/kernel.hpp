#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core.hpp"

__device__ __host__ __forceinline__ long long isqrt64(long long x);
__device__ __host__ __forceinline__ void
pair_index_to_agents(long long idx, int n_agents, int &i_out, int &j_out);

__global__ void play_all_pairs(const AgentParams *__restrict__ params,
                               int n_agents, int rounds, uint64_t seed,
                               const std::size_t *__restrict__ match_offsets,
                               int *__restrict__ match_counts,
                               float *__restrict__ match_q,
                               long long *__restrict__ scores);

std::size_t compute_match_offsets(const std::vector<AgentParams> &hparams,
                                  std::vector<std::size_t> &match_offsets);
