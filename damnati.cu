// damnati.cu — CUDA-accelerated Iterated Prisoner's Dilemma (IPD) tournament
// Build: nvcc -O3 -arch=sm_86 damnati.cu -o damnati
// Run:   ./damnati --agents 512 --rounds 200 --seed 42 --p-ngram 0.6

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <getopt.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Payoffs
constexpr int Rw = 3; // (C,C)
constexpr int Sw = 0; // (C,D)
constexpr int Tw = 5; // (D,C)
constexpr int Pw = 1; // (D,D)

enum Act : int { C = 0, D = 1 };

enum Strategy : int {
  AC = 0,
  AD,
  TFT,
  GTFT,
  GRIM,
  RANDOM,
  PAVLOV,
  ALT,
  JOSS,
  TESTER,
  REPEAT,
  S_TFT,
  NGRAM
};

__device__ __host__ __forceinline__ int dmin(int a, int b) {
  return a < b ? a : b;
}
__device__ __host__ __forceinline__ int dmax(int a, int b) {
  return a > b ? a : b;
}
template <typename T>
__device__ __host__ __forceinline__ T dclamp(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Integer square root using the non-restoring method.
// Returns floor(sqrt(x)) for 0 <= x < 2^63.
__device__ __host__ __forceinline__ long long isqrt64(long long x) {
  long long res = 0;
  long long bit = 1LL << 62; // Start at the highest power-of-four <= x (4^31)
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

struct AgentParams {
  int strat;        // Strategy enum
  float epsilon;    // N-gram ε
  int depth;        // N-gram depth (determines memory size)
  float gtf_forget; // GTFT forgiveness prob
  int *counts;      // Pointer into preallocated visit-count buffer
  float *q;         // Pointer into preallocated Q-value buffer
};

__device__ __constant__ int d_payA[4] = {Rw, Sw, Tw, Pw};
__device__ __constant__ int d_payB[4] = {Rw, Tw, Sw, Pw};

__device__ __host__ __forceinline__ uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97f4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return x;
}
__device__ __host__ __forceinline__ float rng01(uint64_t seed, int i, int j,
                                                int r, int who) {
  uint64_t x = seed;
  x ^= (uint64_t)i * 0xD1342543DE82EF95ULL;
  x ^= (uint64_t)j * 0xA24BAED4963EE407ULL;
  x ^= (uint64_t)r * 0x9E3779B97f4A7C15ULL;
  x ^= (uint64_t)who * 0xC2B2AE3D27D4EB4FULL;
  x = mix64(x);
  return ((x >> 40) & 0xFFFFFF) / 16777216.0f;
}

struct PlayerState {
  int strat;
  int last = -1;
  int opp_last = -1;
  int defect_seen = 0;
  int depth = 0; // current depth
  float epsilon = 0.0f;
  unsigned int state = 0; // encoded history (2 bits/pair)
  int *counts = nullptr;  // visit counts buffer
  float *q = nullptr;     // Q-values buffer
  float gtft_forget = 0.1f;

  __device__ __forceinline__ void init_ngram(int d, float eps, int *cbuf,
                                             float *qbuf) {
    depth = d;
    epsilon = eps;
    state = 0u;
    counts = cbuf;
    q = qbuf;
    int n_states = 1 << (2 * d);
    for (int i = 0; i < n_states * 2; ++i) {
      counts[i] = 0;
      q[i] = 0.0f;
    }
  }
};

__device__ __host__ __forceinline__ int encode_pair(int my, int opp) {
  return (my << 1) | opp;
}

__device__ __host__ __forceinline__ int
ngram_choose(PlayerState &p, uint64_t seed, int i, int j, int r, int who) {
  if (p.depth <= 0) {
    return (p.q[0 * 2 + C] >= p.q[0 * 2 + D]) ? C : D;
  }
  if (rng01(seed, i, j, r, who) < p.epsilon)
    return (rng01(seed, i, j, r, who + 2) < 0.5f ? C : D);
  return (p.q[p.state * 2 + C] >= p.q[p.state * 2 + D]) ? C : D;
}

__device__ __host__ __forceinline__ void ngram_update(PlayerState &p, int my,
                                                      int opp, int reward) {
  const int mask = (p.depth == 0) ? 0 : ((1 << (2 * p.depth)) - 1);
  const int s = (p.depth == 0) ? 0 : p.state;
  int a = my;
  int cnt = ++p.counts[s * 2 + a];
  float oldq = p.q[s * 2 + a];
  p.q[s * 2 + a] = oldq + (float(reward) - oldq) / float(cnt);
  if (p.depth > 0) {
    unsigned int np = ((p.state << 2) | encode_pair(my, opp)) & (unsigned)mask;
    p.state = np;
  }
}

__device__ __host__ __forceinline__ int
choose_action(PlayerState &p, uint64_t seed, int i, int j, int r, int who) {
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
  }
  return C;
}

__global__ void play_all_pairs(const AgentParams *__restrict__ params,
                               int n_agents, int rounds, uint64_t seed,
                               int *__restrict__ scores) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)n_agents * (n_agents - 1) / 2;
  if (idx >= total)
    return;

  // Invert the triangular number T(i) = i*(2n - i - 1)/2 to recover
  // the agent indices (i,j) from the linear pair index `idx` without
  // resorting to floating-point math.  The discriminant inside the square
  // root fits in 64 bits for practical `n_agents`.
  long long disc = -8LL * idx + 4LL * n_agents * (n_agents - 1) - 7LL;
  long long s = isqrt64(disc);
  int i = (int)(n_agents - 2 - ((s - 1) >> 1));
  long long start = (long long)i * (2LL * n_agents - i - 1) / 2;
  int j = (int)(i + 1 + (idx - start));

  AgentParams Ai = params[i];
  AgentParams Bj = params[j];

  PlayerState A;
  A.strat = Ai.strat;
  A.gtft_forget = Ai.gtf_forget;
  PlayerState B;
  B.strat = Bj.strat;
  B.gtft_forget = Bj.gtf_forget;

  if (A.strat == NGRAM)
    A.init_ngram(Ai.depth, Ai.epsilon, Ai.counts, Ai.q);
  if (B.strat == NGRAM)
    B.init_ngram(Bj.depth, Bj.epsilon, Bj.counts, Bj.q);

  int scoreA = 0, scoreB = 0;

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

struct Config {
  int n_agents = 256;
  int rounds = 200;
  uint64_t seed = 1234567ULL;
  float p_ngram = 0.5f; // fraction of N-gram learners
  int depth = 3;        // N-gram depth
  float epsilon = 0.1f; // N-gram ε
  float gtft_p = 0.1f;  // GTFT forgiveness
};

void parse_cli(int argc, char **argv, Config &cfg) {
  static const struct option long_opts[] = {
      {"agents", required_argument, nullptr, 'a'},
      {"rounds", required_argument, nullptr, 'r'},
      {"seed", required_argument, nullptr, 's'},
      {"p-ngram", required_argument, nullptr, 'p'},
      {"depth", required_argument, nullptr, 'd'},
      {"epsilon", required_argument, nullptr, 'e'},
      {"gtft", required_argument, nullptr, 'g'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
    switch (opt) {
    case 'a':
      cfg.n_agents = std::atoi(optarg);
      if (cfg.n_agents <= 0) {
        std::fprintf(stderr, "Error: --agents must be positive.\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 'r':
      cfg.rounds = std::atoi(optarg);
      if (cfg.rounds <= 0) {
        std::fprintf(stderr, "Error: --rounds must be positive.\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 's':
      cfg.seed = std::strtoull(optarg, nullptr, 10);
      break;
    case 'p':
      cfg.p_ngram = std::atof(optarg);
      if (cfg.p_ngram < 0.0f || cfg.p_ngram > 1.0f) {
        std::fprintf(stderr, "Error: --p-ngram must be in [0,1].\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 'd':
      cfg.depth = std::atoi(optarg);
      if (cfg.depth < 0) {
        std::fprintf(stderr, "Error: --depth must be non-negative.\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 'e':
      cfg.epsilon = std::atof(optarg);
      if (cfg.epsilon < 0.0f || cfg.epsilon > 1.0f) {
        std::fprintf(stderr, "Error: --epsilon must be in [0,1].\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 'g':
      cfg.gtft_p = std::atof(optarg);
      if (cfg.gtft_p < 0.0f || cfg.gtft_p > 1.0f) {
        std::fprintf(stderr, "Error: --gtft must be in [0,1].\n");
        std::exit(EXIT_FAILURE);
      }
      break;
    case 'h':
      std::printf("Usage: %s [OPTIONS]\n\n", argv[0]);
      std::printf("Options:\n");
      std::printf("  --agents N    number of agents (>0)\n");
      std::printf("  --rounds R    rounds per match (>0)\n");
      std::printf("  --seed S      RNG seed\n");
      std::printf("  --p-ngram F   fraction of N-gram learners [0,1]\n");
      std::printf("  --depth D     N-gram depth (>=0)\n");
      std::printf("  --epsilon E   exploration rate [0,1]\n");
      std::printf("  --gtft P      GTFT forgiveness [0,1]\n");
      std::printf("\nExample:\n  %s --agents 512 --rounds 200 --p-ngram 0.6 "
                  "--depth 3 --epsilon 0.1 --gtft 0.2\n",
                  argv[0]);
      std::exit(0);
    default:
      break;
    }
  }
}

static const Strategy classics[12] = {AC,     AD,  TFT,  GTFT,   GRIM,   RANDOM,
                                      PAVLOV, ALT, JOSS, TESTER, REPEAT, S_TFT};

static inline void iswap(int &a, int &b) {
  int t = a;
  a = b;
  b = t;
}

static inline uint64_t host_mix64(uint64_t x) {
  x += 0x9E3779B97f4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return x;
}
static void host_shuffle(std::vector<int> &idx, uint64_t seed) {
  uint64_t x = seed ? seed : 0xA5A5A5A5A5A5A5A5ULL;
  for (int i = (int)idx.size() - 1; i > 0; --i) {
    x = host_mix64(x ^ (uint64_t)i);
    int j = (int)(x % (uint64_t)(i + 1));
    if (j < 0)
      j = -j; // paranoia
    iswap(idx[i], idx[j]);
  }
}

void build_population(const Config &cfg, std::vector<AgentParams> &hparams) {
  const int n = cfg.n_agents;
  const int n_ng = int(cfg.p_ngram * n + 0.5f);
  for (int i = 0; i < n; ++i) {
    AgentParams p{};
    if (i < n_ng) {
      p.strat = NGRAM;
      p.depth = cfg.depth;
      p.epsilon = cfg.epsilon;
      p.gtf_forget = cfg.gtft_p;
    } else {
      Strategy s = classics[(i - n_ng) % 12];
      p.strat = s;
      p.depth = 0;
      p.epsilon = 0.0f;
      p.gtf_forget = cfg.gtft_p;
    }
    hparams[i] = p;
  }

  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i)
    idx[i] = i;
  host_shuffle(idx, cfg.seed);
  std::vector<AgentParams> copy = hparams;
  for (int i = 0; i < n; ++i)
    hparams[i] = copy[idx[i]];
}

void run_gpu(const Config &cfg) {
  const int n = cfg.n_agents;
  const int rounds = cfg.rounds;
  const uint64_t seed = cfg.seed;

  std::vector<AgentParams> hparams(n);
  build_population(cfg, hparams);

  // Preallocate buffers for all N-gram agents
  size_t total_states = 0;
  for (int i = 0; i < n; ++i) {
    if (hparams[i].strat == NGRAM) {
      int states = 1 << (2 * hparams[i].depth);
      total_states += (size_t)states * 2;
    }
  }
  std::vector<int> h_counts(total_states, 0);
  std::vector<float> h_q(total_states, 0.0f);
  int *d_counts = nullptr;
  float *d_q = nullptr;
  if (total_states > 0) {
    CUDA_CHECK(cudaMalloc(&d_counts, total_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_q, total_states * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_counts, h_counts.data(), total_states * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_states * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  size_t offset = 0;
  for (int i = 0; i < n; ++i) {
    if (hparams[i].strat == NGRAM) {
      int states = 1 << (2 * hparams[i].depth);
      hparams[i].counts = d_counts + offset;
      hparams[i].q = d_q + offset;
      offset += (size_t)states * 2;
    } else {
      hparams[i].counts = nullptr;
      hparams[i].q = nullptr;
    }
  }

  AgentParams *d_params = nullptr;
  int *d_scores = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_params, n * sizeof(AgentParams)));
  CUDA_CHECK(cudaMallocManaged(&d_scores, n * sizeof(int)));
  std::memcpy(d_params, hparams.data(), n * sizeof(AgentParams));
  std::memset(d_scores, 0, n * sizeof(int));

  long long total_pairs = (long long)n * (n - 1) / 2;
  int threads = 256;
  int blocks = (int)((total_pairs + threads - 1) / threads);

  play_all_pairs<<<blocks, threads>>>(d_params, n, rounds, seed, d_scores);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  long long total = 0;
  int minv = 2147483647;
  int maxv = -2147483647 - 1;
  for (int i = 0; i < n; ++i) {
    total += d_scores[i];
    minv = dmin(minv, d_scores[i]);
    maxv = dmax(maxv, d_scores[i]);
  }

  double mean = double(total) / double(n);
  long double varacc = 0.0L;
  for (int i = 0; i < n; ++i) {
    long double d = (long double)d_scores[i] - (long double)mean;
    varacc += d * d;
  }
  double stdev = std::sqrt(double(varacc / (n > 1 ? (n - 1) : 1)));

  std::printf("{\"agents\":%d,\"rounds\":%d,\"p_ngram\":%.3f,\"depth\":%d,"
              "\"epsilon\":%.3f,\n",
              n, rounds, cfg.p_ngram, cfg.depth, cfg.epsilon);
  std::printf(" \"avg_score\":%.3f,\"min\":%d,\"max\":%d,\"stdev\":%.3f,\n",
              mean, minv, maxv, stdev);

  double sum_by[14] = {0.0};
  int cnt_by[14] = {0};
  for (int i = 0; i < n; ++i) {
    sum_by[hparams[i].strat] += d_scores[i];
    cnt_by[hparams[i].strat]++;
  }

  std::printf(" \"by_strategy\":{");
  const char *names[] = {"AC",     "AD",     "TFT",  "GTFT", "GRIM",
                         "RANDOM", "PAVLOV", "ALT",  "JOSS", "TESTER",
                         "REPEAT", "S_TFT",  "NGRAM"};
  bool first = true;
  for (int s = 0; s <= NGRAM; ++s) {
    if (cnt_by[s] == 0)
      continue;
    if (!first) {
      std::printf(",");
    }
    std::printf("\"%s\":{\"mean\":%.3f,\"count\":%d}", names[s],
                sum_by[s] / (double)dmax(1, cnt_by[s]), cnt_by[s]);
    first = false;
  }
  std::printf("}}\n");

  int show = dmin(10, n);
  for (int i = 0; i < show; ++i) {
    std::printf("agent[%d]: strat=%s score=%d\n", i, names[hparams[i].strat],
                d_scores[i]);
  }

  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_scores));
  if (d_counts)
    CUDA_CHECK(cudaFree(d_counts));
  if (d_q)
    CUDA_CHECK(cudaFree(d_q));
}

#ifndef DAMNATI_NO_MAIN
int main(int argc, char **argv) {
  Config cfg;
  parse_cli(argc, argv, cfg);
  run_gpu(cfg);
  return 0;
}
#endif
