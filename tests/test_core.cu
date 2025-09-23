#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define DAMNATI_NO_MAIN
#include "../damnati.cu"

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
