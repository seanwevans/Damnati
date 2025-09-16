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
