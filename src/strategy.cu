#include "strategy.hpp"

#include <cassert>
#include <cstdint>

__device__ __host__ int dmin(int a, int b) { return a < b ? a : b; }
__device__ __host__ int dmax(int a, int b) { return a > b ? a : b; }

__device__ __host__ std::size_t ngram_span(int depth) {
  return (static_cast<std::size_t>(1) << (2 * depth)) * 2;
}

__device__ __host__ uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97f4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return x;
}
__device__ __host__ float rng01(uint64_t seed, int i, int j, int r, int who) {
  uint64_t x = seed;
  x ^= (uint64_t)i * 0xD1342543DE82EF95ULL;
  x ^= (uint64_t)j * 0xA24BAED4963EE407ULL;
  x ^= (uint64_t)r * 0x9E3779B97f4A7C15ULL;
  x ^= (uint64_t)who * 0xC2B2AE3D27D4EB4FULL;
  x = mix64(x);
  return ((x >> 40) & 0xFFFFFF) / 16777216.0f;
}

__device__ void PlayerState::init_ngram(int d, float eps, int *cbuf,
                                        float *qbuf) {
  depth = d;
  epsilon = eps;
  state = 0u;
  counts = cbuf;
  q = qbuf;
}

__device__ __host__ int encode_pair(int my, int opp) { return (my << 1) | opp; }

__device__ __host__ int ngram_choose(PlayerState &p, uint64_t seed, int i,
                                     int j, int r, int who) {
  if (p.depth <= 0) {
    return (p.q[0 * 2 + C] >= p.q[0 * 2 + D]) ? C : D;
  }
  if (rng01(seed, i, j, r, who) < p.epsilon)
    return (rng01(seed, i, j, r, who + 2) < 0.5f ? C : D);
  return (p.q[p.state * 2 + C] >= p.q[p.state * 2 + D]) ? C : D;
}

__device__ __host__ void ngram_update(PlayerState &p, int my, int opp,
                                      int reward) {
  const unsigned int state_bits =
      static_cast<unsigned int>(sizeof(p.state) * 8u);
  const int desired_bits = (p.depth <= 0) ? 0 : 2 * p.depth;
  const unsigned int used_bits =
      (desired_bits <= 0)
          ? 0u
          : static_cast<unsigned int>(
                desired_bits >= static_cast<int>(state_bits) ? state_bits
                                                             : desired_bits);
  const std::uint64_t mask64 =
      (used_bits == 0u) ? 0ULL : ((std::uint64_t{1} << used_bits) - 1ULL);
  const int s = (p.depth == 0) ? 0 : p.state;
  int a = my;
  int cnt = ++p.counts[s * 2 + a];
  float oldq = p.q[s * 2 + a];
  p.q[s * 2 + a] = oldq + (float(reward) - oldq) / float(cnt);
  if (p.depth > 0) {
    std::uint64_t next = (static_cast<std::uint64_t>(p.state) << 2) |
                         static_cast<std::uint64_t>(encode_pair(my, opp));
    next &= mask64;
    p.state = static_cast<unsigned int>(next);
  }
}

__device__ __host__ int choose_action(PlayerState &p, uint64_t seed, int i,
                                      int j, int r, int who) {
  switch ((Strategy)p.strat) {
  case AC:
    return C;
  case AD:
    return D;
  case TFT:
    return (p.opp_last == -1 ? C : p.opp_last);
  case GTFT:
    if (p.opp_last == D && rng01(seed, i, j, r, who) < p.gtft_forget)
      return C;
    return (p.opp_last == -1 ? C : p.opp_last);
  case GRIM:
    return (p.defect_seen > 0 ? D : C);
  case RANDOM:
    return (rng01(seed, i, j, r, who) < 0.5f ? C : D);
  case PAVLOV:
    return (p.last == p.opp_last ? (p.last == -1 ? C : p.last)
                                 : (p.last == -1 ? D : 1 - p.last));
  case ALT:
    return (p.last == -1 ? C : 1 - p.last);
  case JOSS:
    if (rng01(seed, i, j, r, who) < 0.1f)
      return D;
    return (p.opp_last == -1 ? C : p.opp_last);
  case TESTER:
    if (p.defect_seen == 0)
      return D;
    return (p.opp_last == -1 ? C : p.opp_last);
  case REPEAT:
    return (p.last == -1 ? C : p.last);
  case S_TFT:
    return (p.opp_last == -1 ? D : p.opp_last);
  case NGRAM:
    return ngram_choose(p, seed, i, j, r, who);
  default:
    assert(!"Unknown strategy in choose_action");
    return C;
  }
}
