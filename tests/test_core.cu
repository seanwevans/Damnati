#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define DAMNATI_NO_MAIN
#include "../damnati.cu"

namespace {

bool ensure_cuda_device_available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaErrorNoDevice || device_count == 0) {
    INFO("No CUDA devices detected; skipping GPU test");
    SUCCEED("No CUDA devices detected; skipping GPU test");
    return false;
  }
  if (err != cudaSuccess) {
    INFO("cudaGetDeviceCount failed: " << cudaGetErrorString(err));
    SUCCEED("cudaGetDeviceCount failed; skipping GPU test");
    return false;
  }
  return true;
}

} // namespace

TEST_CASE("choose_action outputs expected moves", "[choose]") {
  uint64_t seed = 0ULL;
  PlayerState p{};
  p.last = -1;
  p.opp_last = -1;
  p.defect_seen = 0;
  p.gtft_forget = 0.0f;

  SECTION("AC always cooperates") {
    p.strat = AC;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
  }
  SECTION("AD always defects") {
    p.strat = AD;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("TFT mirrors last move") {
    p.strat = TFT;
    p.opp_last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.opp_last = D;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("GTFT forgives based on probability") {
    p.strat = GTFT;
    p.opp_last = D;
    p.gtft_forget = 1.0f;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.gtft_forget = 0.0f;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("GRIM defects after one defection") {
    p.strat = GRIM;
    p.defect_seen = 0;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.defect_seen = 1;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("RANDOM uses rng") {
    p.strat = RANDOM;
    float rv = rng01(seed, 0, 0, 0, 0);
    int expected = rv < 0.5f ? C : D;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == expected);
  }
  SECTION("PAVLOV repeats if same, else switches") {
    p.strat = PAVLOV;
    p.last = C;
    p.opp_last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.last = C;
    p.opp_last = D;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("ALT alternates starting with C") {
    p.strat = ALT;
    p.last = -1;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("JOSS defects randomly") {
    p.strat = JOSS;
    p.opp_last = C;
    float rv = rng01(seed, 0, 0, 0, 0);
    int expected = rv < 0.1f ? D : C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == expected);
  }
  SECTION("TESTER defects until opponent does") {
    p.strat = TESTER;
    p.defect_seen = 0;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
    p.defect_seen = 1;
    p.opp_last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
  }
  SECTION("REPEAT mirrors own last move") {
    p.strat = REPEAT;
    p.last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
    p.last = D;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
  }
  SECTION("S_TFT starts with D then mirrors") {
    p.strat = S_TFT;
    p.opp_last = -1;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
    p.opp_last = C;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
  }
  SECTION("NGRAM chooses action with higher Q value") {
    p.strat = NGRAM;
    p.depth = 0;
    int counts[2] = {0, 0};
    float q[2] = {1.0f, 2.0f};
    p.counts = counts;
    p.q = q;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == D);
    q[0] = 3.0f;
    q[1] = 2.0f;
    REQUIRE(choose_action(p, seed, 0, 0, 0, 0) == C);
  }
}

TEST_CASE("ngram_update accumulates payoffs", "[ngram]") {
  PlayerState p{};
  p.depth = 1;
  p.state = 0;
  int counts[8] = {0};
  float q[8] = {0.0f};
  p.counts = counts;
  p.q = q;

  ngram_update(p, C, D, 3);
  REQUIRE(counts[0 * 2 + C] == 1);
  REQUIRE(q[0 * 2 + C] == Approx(3.0f));
  REQUIRE(p.state == encode_pair(C, D));

  ngram_update(p, C, C, 5);
  REQUIRE(counts[1 * 2 + C] == 1);
  REQUIRE(q[1 * 2 + C] == Approx(5.0f));
  REQUIRE(p.state == 0);

  ngram_update(p, C, C, 1);
  REQUIRE(counts[0 * 2 + C] == 2);
  REQUIRE(q[0 * 2 + C] == Approx(2.0f));

  SECTION("handles maximum depth mask correctly") {
    PlayerState max_p{};
    max_p.depth = MAX_NGRAM_DEPTH;
    max_p.state = 0u;

    const std::size_t tracked_states = 1u << 10; // limit history states we visit
    std::vector<int> deep_counts(tracked_states * 2, 0);
    std::vector<float> deep_q(tracked_states * 2, 0.0f);
    max_p.counts = deep_counts.data();
    max_p.q = deep_q.data();

    const unsigned int state_bits = static_cast<unsigned int>(sizeof(max_p.state) * 8u);
    const unsigned int used_bits =
        std::min<unsigned int>(static_cast<unsigned int>(max_p.depth * 2), state_bits);
    const std::uint64_t mask64 =
        (used_bits == 0u) ? 0ULL : ((std::uint64_t{1} << used_bits) - 1ULL);
    const unsigned int mask = static_cast<unsigned int>(mask64);

    const std::array<std::pair<int, int>, 5> history = {
        std::make_pair(C, C), std::make_pair(D, C),
        std::make_pair(C, D), std::make_pair(D, D),
        std::make_pair(C, C)};

    for (const auto &[my_act, opp_act] : history) {
      REQUIRE(max_p.state < tracked_states);
      const unsigned int prev_state = max_p.state;
      ngram_update(max_p, my_act, opp_act, 1);
      const unsigned int expected_state = static_cast<unsigned int>(
          (((static_cast<std::uint64_t>(prev_state) << 2) |
            static_cast<std::uint64_t>(encode_pair(my_act, opp_act))) &
           mask64));
      REQUIRE(max_p.state == expected_state);
      REQUIRE(max_p.state <= mask);
    }
  }
}

TEST_CASE("isqrt64 computes floor square roots", "[isqrt64]") {
  struct Case {
    long long input;
    long long expected;
  };
  const Case cases[] = {{0, 0},
                        {1, 1},
                        {2, 1},
                        {3, 1},
                        {4, 2},
                        {7, 2},
                        {8, 2},
                        {15, 3},
                        {16, 4},
                        {24, 4},
                        {25, 5},
                        {(1LL << 62) - 1, 2147483647LL},
                        {(1LL << 62), 2147483648LL}};

  for (const auto &c : cases) {
    CAPTURE(c.input);
    REQUIRE(isqrt64(c.input) == c.expected);
  }
}

TEST_CASE("pair_index_to_agents enumerates all unique pairs", "[pairs]") {
  const int n_agents = 7;
  const long long total_pairs =
      static_cast<long long>(n_agents) * (n_agents - 1) / 2;
  REQUIRE(total_pairs > 0);

  std::vector<std::pair<int, int>> expected;
  expected.reserve(static_cast<std::size_t>(total_pairs));
  for (int i = 0; i < n_agents; ++i) {
    for (int j = i + 1; j < n_agents; ++j) {
      expected.emplace_back(i, j);
    }
  }
  REQUIRE(expected.size() == static_cast<std::size_t>(total_pairs));

  std::vector<char> seen(static_cast<std::size_t>(n_agents * n_agents), 0);
  std::size_t seen_pairs = 0;

  for (long long idx = 0; idx < total_pairs; ++idx) {
    int i_out = -1;
    int j_out = -1;
    pair_index_to_agents(idx, n_agents, i_out, j_out);

    INFO("idx=" << idx);
    REQUIRE(i_out >= 0);
    REQUIRE(j_out >= 0);
    REQUIRE(i_out < n_agents);
    REQUIRE(j_out < n_agents);
    REQUIRE(i_out < j_out);

    const std::pair<int, int> observed{i_out, j_out};
    REQUIRE(observed == expected[static_cast<std::size_t>(idx)]);

    std::size_t key = static_cast<std::size_t>(i_out * n_agents + j_out);
    REQUIRE(seen[key] == 0);
    seen[key] = 1;
    ++seen_pairs;
  }

  REQUIRE(seen_pairs == expected.size());

  int first_i = -1;
  int first_j = -1;
  pair_index_to_agents(0, n_agents, first_i, first_j);
  REQUIRE(first_i == expected.front().first);
  REQUIRE(first_j == expected.front().second);

  int last_i = -1;
  int last_j = -1;
  pair_index_to_agents(total_pairs - 1, n_agents, last_i, last_j);
  REQUIRE(last_i == expected.back().first);
  REQUIRE(last_j == expected.back().second);
}

TEST_CASE("sorted_agent_scores orders by score then index", "[report]") {
  const long long scores[] = {15, 42, 42, -3, 0};
  auto ranked = sorted_agent_scores(scores, 5);
  REQUIRE(ranked.size() == 5);
  REQUIRE(ranked[0].first == 1);
  REQUIRE(ranked[0].second == 42);
  REQUIRE(ranked[1].first == 2);
  REQUIRE(ranked[1].second == 42);
  REQUIRE(ranked[2].first == 0);
  REQUIRE(ranked[2].second == 15);
  REQUIRE(ranked[3].first == 4);
  REQUIRE(ranked[3].second == 0);
  REQUIRE(ranked[4].first == 3);
  REQUIRE(ranked[4].second == -3);
}

TEST_CASE("GPU tournament with multiple N-gram agents is deterministic",
          "[gpu]") {
  if (!ensure_cuda_device_available()) {
    return;
  }

  const int n_agents = 6;
  const int rounds = 20;
  const uint64_t seed = 424242ULL;

  auto make_ngram = [](int depth, float epsilon) {
    AgentParams p{};
    p.strat = NGRAM;
    p.depth = depth;
    p.epsilon = epsilon;
    p.gtft_forget = 0.0f;
    return p;
  };

  auto make_classic = [](Strategy s) {
    AgentParams p{};
    p.strat = s;
    p.depth = 0;
    p.epsilon = 0.0f;
    p.gtft_forget = 0.0f;
    return p;
  };

  std::vector<AgentParams> agents(n_agents);
  agents[0] = make_ngram(1, 0.0f);
  agents[1] = make_ngram(2, 0.05f);
  agents[2] = make_ngram(0, 0.0f);
  agents[3] = make_classic(TFT);
  agents[4] = make_classic(GRIM);
  agents[5] = make_classic(ALT);

  const long long total_pairs = static_cast<long long>(n_agents) *
                                static_cast<long long>(n_agents - 1) / 2;
  REQUIRE(total_pairs > 0);

  std::vector<std::size_t> match_offsets;
  std::size_t total_span = compute_match_offsets(agents, match_offsets);
  REQUIRE(match_offsets.size() == static_cast<std::size_t>(total_pairs) * 2);
  REQUIRE(total_span > 0);

  int *d_match_counts = nullptr;
  float *d_match_q = nullptr;
  CUDA_CHECK(cudaMalloc(&d_match_counts, total_span * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_match_q, total_span * sizeof(float)));

  std::size_t *d_match_offsets = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_match_offsets,
                               match_offsets.size() * sizeof(std::size_t)));
  std::memcpy(d_match_offsets, match_offsets.data(),
              match_offsets.size() * sizeof(std::size_t));

  AgentParams *d_params = nullptr;
  long long *d_scores = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_params, n_agents * sizeof(AgentParams)));
  CUDA_CHECK(cudaMallocManaged(&d_scores, n_agents * sizeof(long long)));
  std::memcpy(d_params, agents.data(), n_agents * sizeof(AgentParams));

  auto run_once = [&](std::vector<long long> &out_scores) {
    CUDA_CHECK(cudaMemset(d_scores, 0, n_agents * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_match_counts, 0, total_span * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_match_q, 0, total_span * sizeof(float)));
    int threads = 64;
    int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    play_all_pairs<<<blocks, threads>>>(d_params, n_agents, rounds, seed,
                                        d_match_offsets, d_match_counts,
                                        d_match_q, d_scores);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out_scores.assign(d_scores, d_scores + n_agents);
  };

  std::vector<long long> first_run;
  std::vector<long long> second_run;
  run_once(first_run);
  run_once(second_run);

  REQUIRE(first_run == second_run);

  CUDA_CHECK(cudaFree(d_match_counts));
  CUDA_CHECK(cudaFree(d_match_q));
  CUDA_CHECK(cudaFree(d_match_offsets));
  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST_CASE("GPU tournament without N-gram agents avoids auxiliary buffers",
          "[gpu][no_ngram]") {
  if (!ensure_cuda_device_available()) {
    return;
  }

  const int n_agents = 4;
  const int rounds = 10;
  const uint64_t seed = 1337ULL;

  std::vector<AgentParams> agents(n_agents);
  const Strategy roster[n_agents] = {AC, AD, TFT, GRIM};
  for (int i = 0; i < n_agents; ++i) {
    agents[i].strat = roster[i];
    agents[i].depth = 0;
    agents[i].epsilon = 0.0f;
    agents[i].gtft_forget = 0.0f;
    agents[i].counts = nullptr;
    agents[i].q = nullptr;
  }

  const long long total_pairs = static_cast<long long>(n_agents) *
                                static_cast<long long>(n_agents - 1) / 2;
  REQUIRE(total_pairs > 0);

  std::vector<std::size_t> match_offsets;
  std::size_t total_span = compute_match_offsets(agents, match_offsets);
  REQUIRE(total_span == 0);
  REQUIRE(match_offsets.size() == static_cast<std::size_t>(total_pairs) * 2);
  for (std::size_t offset : match_offsets) {
    REQUIRE(offset == INVALID_OFFSET);
  }

  AgentParams *d_params = nullptr;
  long long *d_scores = nullptr;
  CUDA_CHECK(cudaMallocManaged(&d_params, n_agents * sizeof(AgentParams)));
  CUDA_CHECK(cudaMallocManaged(&d_scores, n_agents * sizeof(long long)));
  std::memcpy(d_params, agents.data(), n_agents * sizeof(AgentParams));
  CUDA_CHECK(cudaMemset(d_scores, 0, n_agents * sizeof(long long)));

  int threads = 64;
  int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
  play_all_pairs<<<blocks, threads>>>(d_params, n_agents, rounds, seed, nullptr,
                                      nullptr, nullptr, d_scores);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<long long> scores(d_scores, d_scores + n_agents);
  for (long long value : scores) {
    REQUIRE(value >= 0);
  }

  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST_CASE("compute_match_offsets detects span overflow", "[overflow]") {
  std::vector<AgentParams> agents(2);
  for (auto &agent : agents) {
    agent.strat = NGRAM;
    agent.depth = 31;
    agent.epsilon = 0.0f;
    agent.gtft_forget = 0.0f;
  }

  std::vector<std::size_t> match_offsets;
  REQUIRE_THROWS_AS(compute_match_offsets(agents, match_offsets),
                    std::overflow_error);
}

TEST_CASE("run_gpu guards against oversized allocations", "[overflow][gpu]") {
  if (!ensure_cuda_device_available()) {
    return;
  }

  Config cfg;
  cfg.n_agents = 2;
  cfg.rounds = 1;
  cfg.seed = 0ULL;
  cfg.p_ngram = 0.25f; // ensures exactly one N-gram agent after rounding
  cfg.depth = 31;
  cfg.epsilon = 0.0f;
  cfg.gtft_p = 0.0f;

  REQUIRE_THROWS_AS(run_gpu(cfg), std::overflow_error);
}
