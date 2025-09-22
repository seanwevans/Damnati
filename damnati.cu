// damnati.cu — CUDA-accelerated Iterated Prisoner's Dilemma (IPD) tournament
// Build: nvcc -O3 -arch=sm_86 damnati.cu -o damnati
// Run:   ./damnati --agents 512 --rounds 200 --seed 42 --p-ngram 0.6

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <getopt.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
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

constexpr std::size_t INVALID_OFFSET = static_cast<std::size_t>(-1);

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

__device__ __host__ __forceinline__ std::size_t ngram_span(int depth) {
  return (static_cast<std::size_t>(1) << (2 * depth)) * 2;
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

struct AgentParams {
  int strat;         // Strategy enum
  float epsilon;     // N-gram ε
  int depth;         // N-gram depth (determines memory size)
  float gtft_forget; // GTFT forgiveness prob
  int *counts;       // Pointer into preallocated visit-count buffer
  float *q;          // Pointer into preallocated Q-value buffer
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
  default:
    assert(!"Unknown strategy in choose_action");
    return C;
  }
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

constexpr int MAX_NGRAM_DEPTH = 15;

struct Config {
  int n_agents = 256;
  int rounds = 200;
  uint64_t seed = 1234567ULL;
  float p_ngram = 0.5f; // fraction of N-gram learners
  int depth = 3;        // N-gram depth
  float epsilon = 0.1f; // N-gram ε
  float gtft_p = 0.1f;  // GTFT forgiveness
};

static void print_usage(FILE *stream, const char *prog) {
  std::fprintf(stream, "Usage: %s [OPTIONS]\n\n", prog);
  std::fprintf(stream, "Options:\n");
  std::fprintf(stream, "  --agents N    number of agents (>0)\n");
  std::fprintf(stream, "  --rounds R    rounds per match (>0)\n");
  std::fprintf(stream, "  --seed S      RNG seed\n");
  std::fprintf(stream, "  --p-ngram F   fraction of N-gram learners [0,1]\n");
  std::fprintf(stream, "  --depth D     N-gram depth (>=0)\n");
  std::fprintf(stream, "  --epsilon E   exploration rate [0,1]\n");
  std::fprintf(stream, "  --gtft P      GTFT forgiveness [0,1]\n");
  std::fprintf(stream,
               "\nExample:\n  %s --agents 512 --rounds 200 --p-ngram 0.6 "
               "--depth 3 --epsilon 0.1 --gtft 0.2\n",
               prog);
}

namespace {

std::string format_flag(const char *opt_name) {
  return std::string("--") + opt_name;
}

int parse_int_option(const char *opt_name, const char *value) {
  errno = 0;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  const std::string arg = value;
  const std::string flag = format_flag(opt_name);
  if (end == value || *end != '\0') {
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg +
                             "'.");
  }
  if (errno == ERANGE || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Error: value for " + flag +
                             " is out of range: '" + arg + "'.");
  }
  return static_cast<int>(parsed);
}

uint64_t parse_uint64_option(const char *opt_name, const char *value) {
  errno = 0;
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  const std::string arg = value;
  const std::string flag = format_flag(opt_name);
  if (end == value || *end != '\0') {
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg +
                             "'.");
  }
  if (errno == ERANGE || parsed > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("Error: value for " + flag +
                             " is out of range: '" + arg + "'.");
  }
  return static_cast<uint64_t>(parsed);
}

float parse_float_option(const char *opt_name, const char *value) {
  errno = 0;
  char *end = nullptr;
  double parsed = std::strtod(value, &end);
  const std::string arg = value;
  const std::string flag = format_flag(opt_name);
  if (end == value || *end != '\0') {
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg +
                             "'.");
  }
  if (errno == ERANGE || !std::isfinite(parsed) ||
      parsed < -std::numeric_limits<float>::max() ||
      parsed > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Error: value for " + flag +
                             " is out of range: '" + arg + "'.");
  }
  return static_cast<float>(parsed);
}

} // namespace

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

  opterr = 0;
  optind = 1;
  int opt;
  while ((opt = getopt_long(argc, argv, ":a:r:s:p:d:e:g:h", long_opts,
                            nullptr)) != -1) {
    switch (opt) {
    case 'a':
      cfg.n_agents = parse_int_option("agents", optarg);
      if (cfg.n_agents < 2) {
        throw std::runtime_error("Error: --agents must be at least 2.");
      }
      break;
    case 'r':
      cfg.rounds = parse_int_option("rounds", optarg);
      if (cfg.rounds <= 0) {
        throw std::runtime_error("Error: --rounds must be positive.");
      }
      break;
    case 's':
      cfg.seed = parse_uint64_option("seed", optarg);
      break;
    case 'p':
      cfg.p_ngram = parse_float_option("p-ngram", optarg);
      if (cfg.p_ngram < 0.0f || cfg.p_ngram > 1.0f) {
        throw std::runtime_error("Error: --p-ngram must be in [0,1].");
      }
      break;
    case 'd':
      cfg.depth = parse_int_option("depth", optarg);
      if (cfg.depth < 0 || cfg.depth > MAX_NGRAM_DEPTH) {
        throw std::runtime_error(std::string("Error: --depth must be in [0,") +
                                 std::to_string(MAX_NGRAM_DEPTH) + "].");
      }
      break;
    case 'e':
      cfg.epsilon = parse_float_option("epsilon", optarg);
      if (cfg.epsilon < 0.0f || cfg.epsilon > 1.0f) {
        throw std::runtime_error("Error: --epsilon must be in [0,1].");
      }
      break;
    case 'g':
      cfg.gtft_p = parse_float_option("gtft", optarg);
      if (cfg.gtft_p < 0.0f || cfg.gtft_p > 1.0f) {
        throw std::runtime_error("Error: --gtft must be in [0,1].");
      }
      break;
    case 'h':
      print_usage(stdout, argv[0]);
      std::exit(0);
    case ':': {
      const char *flag_name = nullptr;
      switch (optopt) {
      case 'a':
        flag_name = "--agents";
        break;
      case 'r':
        flag_name = "--rounds";
        break;
      case 's':
        flag_name = "--seed";
        break;
      case 'p':
        flag_name = "--p-ngram";
        break;
      case 'd':
        flag_name = "--depth";
        break;
      case 'e':
        flag_name = "--epsilon";
        break;
      case 'g':
        flag_name = "--gtft";
        break;
      default:
        break;
      }
      std::string flag;
      if (flag_name != nullptr) {
        flag = flag_name;
      } else if (optind > 0 && optind - 1 < argc) {
        flag = argv[optind - 1];
      } else {
        flag = "option";
      }
      throw std::runtime_error("Error: " + flag + " requires a value.");
    }
    case '?': {
      std::string flag = (optind > 0 && optind - 1 < argc)
                             ? std::string(argv[optind - 1])
                             : std::string();
      if (!flag.empty())
        throw std::runtime_error("Error: unrecognized option '" + flag + "'.");
      throw std::runtime_error("Error: unrecognized option.");
    }
    default:
      if (optind > 0 && optind <= argc) {
        std::fprintf(stderr, "Error: unknown option '%s'.\n", argv[optind - 1]);
      } else {
        std::fprintf(stderr, "Error: unknown option encountered.\n");
      }
      print_usage(stderr, argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    throw std::runtime_error("Error: unexpected positional argument '" +
                             std::string(argv[optind]) + "'.");
  }
}

static const Strategy classics[12] = {AC,     AD,  TFT,  GTFT,   GRIM,   RANDOM,
                                      PAVLOV, ALT, JOSS, TESTER, REPEAT, S_TFT};

int build_population(const Config &cfg, std::vector<AgentParams> &hparams) {
  const int n = cfg.n_agents;
  const int n_ng = int(cfg.p_ngram * n + 0.5f);
  for (int i = 0; i < n; ++i) {
    AgentParams p{};
    if (i < n_ng) {
      p.strat = NGRAM;
      p.depth = cfg.depth;
      p.epsilon = cfg.epsilon;
      p.gtft_forget = cfg.gtft_p;
    } else {
      Strategy s = classics[(i - n_ng) % 12];
      p.strat = s;
      p.depth = 0;
      p.epsilon = 0.0f;
      p.gtft_forget = cfg.gtft_p;
    }
    hparams[i] = p;
  }

  std::mt19937_64 rng(cfg.seed);
  std::shuffle(hparams.begin(), hparams.end(), rng);
  return n_ng;
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
      match_offsets[static_cast<std::size_t>(idx) * 2 + 0] = offset;
      offset += span;
    }
    if (hparams[j].strat == NGRAM) {
      std::size_t span = ngram_span(hparams[j].depth);
      match_offsets[static_cast<std::size_t>(idx) * 2 + 1] = offset;
      offset += span;
    }
  }
  return offset;
}

void run_gpu(const Config &cfg) {
  const int n = cfg.n_agents;
  const int rounds = cfg.rounds;
  const uint64_t seed = cfg.seed;

  std::vector<AgentParams> hparams(n);
  int n_ngram = build_population(cfg, hparams);

  long long total_pairs = static_cast<long long>(n) * (n - 1) / 2;
  std::vector<std::size_t> match_offsets;
  std::size_t total_span = 0;
  if (total_pairs > 0 && n_ngram > 0) {
    total_span = compute_match_offsets(hparams, match_offsets);
  }

  int *d_match_counts = nullptr;
  float *d_match_q = nullptr;
  if (total_span > 0) {
    std::size_t counts_bytes = total_span * sizeof(int);
    std::size_t q_bytes = total_span * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_match_counts, counts_bytes));
    CUDA_CHECK(cudaMalloc(&d_match_q, q_bytes));
    CUDA_CHECK(cudaMemset(d_match_counts, 0, counts_bytes));
    CUDA_CHECK(cudaMemset(d_match_q, 0, q_bytes));
  }

  std::size_t *d_match_offsets = nullptr;
  if (!match_offsets.empty()) {
    std::size_t offsets_bytes = match_offsets.size() * sizeof(std::size_t);
    CUDA_CHECK(cudaMallocManaged(&d_match_offsets, offsets_bytes));
    std::memcpy(d_match_offsets, match_offsets.data(), offsets_bytes);
  }

  AgentParams *d_params = nullptr;
  long long *d_scores = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_params, n * sizeof(AgentParams)));
  CUDA_CHECK(cudaMallocManaged(&d_scores, n * sizeof(long long)));
  std::memcpy(d_params, hparams.data(), n * sizeof(AgentParams));
  std::memset(d_scores, 0, n * sizeof(long long));

  if (total_pairs == 0) {
    std::fprintf(
        stderr,
        "Warning: not enough agents to form pairs; skipping kernel launch.\n");
  } else {
    int threads = 256;
    int blocks = (int)((total_pairs + threads - 1) / threads);

    play_all_pairs<<<blocks, threads>>>(d_params, n, rounds, seed,
                                        d_match_offsets, d_match_counts,
                                        d_match_q, d_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  long long total = 0;
  long long minv = std::numeric_limits<long long>::max();
  long long maxv = std::numeric_limits<long long>::min();
  for (int i = 0; i < n; ++i) {
    total += d_scores[i];
    minv = std::min(minv, d_scores[i]);
    maxv = std::max(maxv, d_scores[i]);
  }

  long double mean = static_cast<long double>(total) /
                     static_cast<long double>(std::max(1, n));
  long double varacc = 0.0L;
  for (int i = 0; i < n; ++i) {
    long double d = static_cast<long double>(d_scores[i]) - mean;
    varacc += d * d;
  }
  long double variance = varacc / static_cast<long double>(n > 1 ? (n - 1) : 1);
  double stdev = std::sqrt(static_cast<double>(variance));

  std::printf("{\"agents\":%d,\"rounds\":%d,\"p_ngram\":%.3f,\"depth\":%d,"
              "\"epsilon\":%.3f,\n",
              n, rounds, cfg.p_ngram, cfg.depth, cfg.epsilon);
  std::printf(" \"avg_score\":%.3f,\"min\":%lld,\"max\":%lld,\"stdev\":%.3f,\n",
              static_cast<double>(mean), minv, maxv, stdev);

  constexpr std::size_t strategy_count = static_cast<std::size_t>(NGRAM) + 1;
  std::array<long double, strategy_count> sum_by{};
  std::array<int, strategy_count> cnt_by{};
  for (int i = 0; i < n; ++i) {
    sum_by[hparams[i].strat] += static_cast<long double>(d_scores[i]);
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
  for (int i = 0; i < show; ++i) {
    std::printf("agent[%d]: strat=%s score=%lld\n", i, names[hparams[i].strat],
                d_scores[i]);
  }

  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_scores));
  if (d_match_offsets)
    CUDA_CHECK(cudaFree(d_match_offsets));
  if (d_match_counts)
    CUDA_CHECK(cudaFree(d_match_counts));
  if (d_match_q)
    CUDA_CHECK(cudaFree(d_match_q));
}

#ifndef DAMNATI_NO_MAIN
int main(int argc, char **argv) {
  Config cfg;
  try {
    parse_cli(argc, argv, cfg);
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return EXIT_FAILURE;
  }
  run_gpu(cfg);
  return 0;
}
#endif
