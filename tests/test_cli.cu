#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define DAMNATI_NO_MAIN
#include "../damnati.cu"

TEST_CASE("parse_cli rejects invalid depth and unknown options", "[cli]") {
  Config cfg;

  {
    char prog[] = "damnati";
    char depth[] = "--depth";
    char neg[] = "-1";
    char *argv[] = {prog, depth, neg, nullptr};
    int argc = 3;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for negative depth");
    } catch (const std::runtime_error &ex) {
      const std::string expected =
          std::string("Error: --depth must be in [0,") +
          std::to_string(MAX_NGRAM_DEPTH) + "].";
      REQUIRE(std::string(ex.what()) == expected);
    }
  }

  {
    char prog[] = "damnati";
    char agents[] = "--agents";
    char one[] = "1";
    char *argv[] = {prog, agents, one, nullptr};
    int argc = 3;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for too few agents");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()) == "Error: --agents must be at least 2.");
    }
  }

  {
    char prog[] = "damnati";
    char agents[] = "--agents";
    char rounds[] = "--rounds";
    char ten[] = "10";
    char *argv[] = {prog, agents, rounds, ten, nullptr};
    int argc = 4;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for missing agents value");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()) == "Error: --agents requires a value.");
    }
  }

  {
    cfg = Config{};
    char prog[] = "damnati";
    char rounds[] = "--rounds";
    char big[] = "2000000000";
    char *argv[] = {prog, rounds, big, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.rounds == 2000000000);
  }

  {
    char prog[] = "damnati";
    char unknown[] = "--unknown";
    char *argv[] = {prog, unknown, nullptr};
    int argc = 2;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for unknown option");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()).find("unrecognized option") !=
              std::string::npos);
    }
  }

  {
    char prog[] = "damnati";
    char token[] = "extra";
    char *argv[] = {prog, token, nullptr};
    int argc = 2;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for unexpected positional argument");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()) ==
              "Error: unexpected positional argument 'extra'.");
    }
  }
}
