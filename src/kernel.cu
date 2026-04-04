#include "kernel.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>

#include "strategy.hpp"

__device__ __constant__ int d_payA[4] = {Rw, Sw, Tw, Pw};
__device__ __constant__ int d_payB[4] = {Rw, Tw, Sw, Pw};

__device__ __host__ __forceinline__ long long isqrt64(long long x) {
  long long res = 0;
  long long bit = 1LL << 62;
  while (bit > x)
    bit >>= 2;
  while (bit != 0) {
    if (x >= res + bit) {
      x -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

__device__ __host__ __forceinline__ void
pair_index_to_agents(long long idx, int n_agents, int &i_out, int &j_out) {
  long long disc = -8LL * idx + 4LL * n_agents * (n_agents - 1) - 7LL;
  long long s = isqrt64(disc);
  int i = static_cast<int>(n_agents - 2 - ((s - 1) >> 1));
  long long start = (long long)i * (2LL * n_agents - i - 1) / 2;
  int j = static_cast<int>(i + 1 + (idx - start));
  i_out = i;
  j_out = j;
}

__global__ void play_all_pairs(const AgentParams *__restrict__ params,
                               int n_agents, int rounds, uint64_t seed,
                               const std::size_t *__restrict__ match_offsets,
                               int *__restrict__ match_counts,
                               float *__restrict__ match_q,
                               long long *__restrict__ scores) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)n_agents * (n_agents - 1) / 2;
  if (idx >= total)
    return;

  int i = 0;
  int j = 0;
  pair_index_to_agents(idx, n_agents, i, j);

  AgentParams Ai = params[i];
  AgentParams Bj = params[j];

  PlayerState A;
  A.strat = Ai.strat;
  A.gtft_forget = Ai.gtft_forget;
  PlayerState B;
  B.strat = Bj.strat;
  B.gtft_forget = Bj.gtft_forget;

  constexpr std::size_t invalid = INVALID_OFFSET;
  if (A.strat == NGRAM) {
    assert(match_offsets && match_counts && match_q);
    if (!(match_offsets && match_counts && match_q))
      return;
    std::size_t offset = match_offsets[idx * 2 + 0];
    assert(offset != invalid);
    if (offset == invalid)
      return;
    A.init_ngram(Ai.depth, Ai.epsilon, match_counts + offset, match_q + offset);
  }
  if (B.strat == NGRAM) {
    assert(match_offsets && match_counts && match_q);
    if (!(match_offsets && match_counts && match_q))
      return;
    std::size_t offset = match_offsets[idx * 2 + 1];
    assert(offset != invalid);
    if (offset == invalid)
      return;
    B.init_ngram(Bj.depth, Bj.epsilon, match_counts + offset, match_q + offset);
  }

  long long scoreA = 0;
  long long scoreB = 0;

#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    int a = choose_action(A, seed, i, j, r, 0);
    int b = choose_action(B, seed, i, j, r, 1);

    int code = (a << 1) | b;
    int pa = d_payA[code];
    int pb = d_payB[code];
    scoreA += pa;
    scoreB += pb;

    if (A.strat == NGRAM)
      ngram_update(A, a, b, pa);
    if (B.strat == NGRAM)
      ngram_update(B, b, a, pb);

    A.opp_last = b;
    B.opp_last = a;
    A.last = a;
    B.last = b;
    if (b == D)
      ++A.defect_seen;
    if (a == D)
      ++B.defect_seen;
  }

  atomicAdd(&scores[i], scoreA);
  atomicAdd(&scores[j], scoreB);
}

std::size_t compute_match_offsets(const std::vector<AgentParams> &hparams,
                                  std::vector<std::size_t> &match_offsets) {
  const int n = static_cast<int>(hparams.size());
  long long total_pairs = static_cast<long long>(n) * (n - 1) / 2;
  match_offsets.assign(static_cast<std::size_t>(total_pairs) * 2,
                       INVALID_OFFSET);
  std::size_t offset = 0;
  for (long long idx = 0; idx < total_pairs; ++idx) {
    int i = 0;
    int j = 0;
    pair_index_to_agents(idx, n, i, j);
    if (hparams[i].strat == NGRAM) {
      std::size_t span = ngram_span(hparams[i].depth);
      if (span > std::numeric_limits<std::size_t>::max() - offset) {
        throw std::overflow_error(
            "Error: match-count buffer size exceeds addressable memory range.");
      }
      match_offsets[static_cast<std::size_t>(idx) * 2 + 0] = offset;
      offset += span;
    }
    if (hparams[j].strat == NGRAM) {
      std::size_t span = ngram_span(hparams[j].depth);
      if (span > std::numeric_limits<std::size_t>::max() - offset) {
        throw std::overflow_error(
            "Error: match-count buffer size exceeds addressable memory range.");
      }
      match_offsets[static_cast<std::size_t>(idx) * 2 + 1] = offset;
      offset += span;
    }
  }
  return offset;
}
