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
      REQUIRE(std::string(ex.what()) == "Error: --depth must be non-negative.");
    }
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
}
