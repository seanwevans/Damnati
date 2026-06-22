#include "app.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "cuda_api.hpp"
#include "kernel.hpp"
#include "reporting.hpp"
#include "strategy.hpp"

static const Strategy classics[12] = {AC,     AD,  TFT,  GTFT,   GRIM,   RANDOM,
                                      PAVLOV, ALT, JOSS, TESTER, REPEAT, S_TFT};

CudaDeviceQueryApi g_cuda_device_query_api{};

std::string describe_device_capability(const cudaDeviceProp &props) {
  return std::string(props.name) + " (compute capability " +
         std::to_string(props.major) + "." + std::to_string(props.minor) + ")";
}

void ensure_runtime_compatibility() {
  int device_count = 0;
  throw_if_cuda_error(g_cuda_device_query_api.get_device_count(&device_count),
                      "cudaGetDeviceCount(&device_count)", __FILE__, __LINE__);
  if (device_count <= 0) {
    throw std::runtime_error(
        "Error: no CUDA device detected. Install an NVIDIA driver/CUDA runtime "
        "and run on a GPU system.");
  }

  cudaDeviceProp first_props{};
  throw_if_cuda_error(
      g_cuda_device_query_api.get_device_properties(&first_props, 0),
      "cudaGetDeviceProperties(&first_props, 0)", __FILE__, __LINE__);
  const std::string selected = describe_device_capability(first_props);
  const bool supported = (first_props.major > kMinComputeCapabilityMajor) ||
                         (first_props.major == kMinComputeCapabilityMajor &&
                          first_props.minor >= kMinComputeCapabilityMinor);
  if (!supported) {
    throw std::runtime_error(
        "Error: selected CUDA device " + selected +
        " is unsupported. Damnati requires compute capability " +
        std::to_string(kMinComputeCapabilityMajor) + "." +
        std::to_string(kMinComputeCapabilityMinor) +
        " or newer. Use a newer GPU or rebuild for a compatible target.");
  }
  std::fprintf(stderr, "CUDA runtime check: selected device %s.\n",
               selected.c_str());
}

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

void run_gpu(const Config &cfg) {
  const int n = cfg.n_agents;
  const int rounds = cfg.rounds;
  const uint64_t seed = cfg.seed;
  constexpr int threads_per_block = 256;

  long long total_pairs = static_cast<long long>(n) * (n - 1) / 2;
  if (total_pairs < 0) {
    throw std::runtime_error("Error: invalid tournament size.");
  }

  const long long max_pairs_supported =
      static_cast<long long>(std::numeric_limits<int>::max()) *
      static_cast<long long>(threads_per_block);
  if (total_pairs > max_pairs_supported) {
    throw std::runtime_error(
        "Error: tournament requires " + std::to_string(total_pairs) +
        " pairings, exceeding the maximum supported " +
        std::to_string(max_pairs_supported) + " for " +
        std::to_string(threads_per_block) + " threads per block.");
  }

  std::vector<AgentParams> hparams(n);
  int n_ngram = build_population(cfg, hparams);

  std::vector<std::size_t> match_offsets;
  std::size_t total_span = 0;
  if (total_pairs > 0 && n_ngram > 0) {
    total_span = compute_match_offsets(hparams, match_offsets);
  }

  CudaPtr d_match_counts;
  CudaPtr d_match_q;
  if (total_span > 0) {
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (total_span > max_size / sizeof(int)) {
      throw std::overflow_error(
          "Error: match-count buffer allocation exceeds size limits.");
    }
    if (total_span > max_size / sizeof(float)) {
      throw std::overflow_error(
          "Error: match-Q buffer allocation exceeds size limits.");
    }
    std::size_t counts_bytes = total_span * sizeof(int);
    std::size_t q_bytes = total_span * sizeof(float);
    throw_if_cuda_error(cuda_malloc(d_match_counts.out(), counts_bytes),
                        "cuda_malloc(d_match_counts.out(), counts_bytes)",
                        __FILE__, __LINE__);
    throw_if_cuda_error(cuda_malloc(d_match_q.out(), q_bytes),
                        "cuda_malloc(d_match_q.out(), q_bytes)", __FILE__,
                        __LINE__);
    throw_if_cuda_error(cuda_memset(d_match_counts.p, 0, counts_bytes),
                        "cuda_memset(d_match_counts.p, 0, counts_bytes)",
                        __FILE__, __LINE__);
    throw_if_cuda_error(cuda_memset(d_match_q.p, 0, q_bytes),
                        "cuda_memset(d_match_q.p, 0, q_bytes)", __FILE__,
                        __LINE__);
    throw_if_cuda_error(
        cuda_memset(d_match_counts.as<int>(), 0, counts_bytes),
        "cuda_memset(d_match_counts.as<int>(), 0, counts_bytes)", __FILE__,
        __LINE__);
    throw_if_cuda_error(cuda_memset(d_match_q.as<float>(), 0, q_bytes),
                        "cuda_memset(d_match_q.as<float>(), 0, q_bytes)",
                        __FILE__, __LINE__);
  }

  CudaPtr d_match_offsets;
  if (!match_offsets.empty()) {
    std::size_t offsets_bytes = match_offsets.size() * sizeof(std::size_t);
    throw_if_cuda_error(
        cuda_malloc_managed(d_match_offsets.out(), offsets_bytes),
        "cuda_malloc_managed(d_match_offsets.out(), offsets_bytes)", __FILE__,
        __LINE__);
    std::memcpy(d_match_offsets.p, match_offsets.data(), offsets_bytes);
  }

  CudaPtr d_params;
  CudaPtr d_scores;
  throw_if_cuda_error(
      cuda_malloc_managed(d_params.out(), n * sizeof(AgentParams)),
      "cuda_malloc_managed(d_params.out(), n * sizeof(AgentParams))", __FILE__,
      __LINE__);
  throw_if_cuda_error(
      cuda_malloc_managed(d_scores.out(), n * sizeof(long long)),
      "cuda_malloc_managed(d_scores.out(), n * sizeof(long long))", __FILE__,
      __LINE__);
  std::memcpy(d_params.p, hparams.data(), n * sizeof(AgentParams));
  std::memset(d_scores.p, 0, n * sizeof(long long));

  if (total_pairs == 0) {
    std::fprintf(
        stderr,
        "Warning: not enough agents to form pairs; skipping kernel launch.\n");
  } else {
    long long blocks_ll =
        (total_pairs + threads_per_block - 1) / threads_per_block;
    if (blocks_ll > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "Error: kernel launch would require " + std::to_string(blocks_ll) +
          " blocks, exceeding the maximum supported value of " +
          std::to_string(std::numeric_limits<int>::max()) + ".");
    }
    int threads = threads_per_block;
    int blocks = static_cast<int>(blocks_ll);

    play_all_pairs<<<blocks, threads>>>(
        d_params.as<AgentParams>(), n, rounds, seed,
        d_match_offsets.as<std::size_t>(), d_match_counts.as<int>(),
        d_match_q.as<float>(), d_scores.as<long long>());
    throw_if_cuda_error(cuda_get_last_error(), "cuda_get_last_error()",
                        __FILE__, __LINE__);
    throw_if_cuda_error(cuda_device_synchronize(), "cuda_device_synchronize()",
                        __FILE__, __LINE__);
  }

  print_summary_report(cfg, hparams, d_scores.as<long long>());
}
