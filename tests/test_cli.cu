#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../src/cli.cu"
#include "../src/strategy.cu"
#include "../src/kernel.cu"
#include "../src/reporting.cu"
#include "../src/app.cu"

TEST_CASE("parse_cli rejects invalid depth and unknown options", "[cli]") {
  {
    Config cfg{};
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
    Config cfg{};
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
    Config cfg{};
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
    Config cfg{};
    char prog[] = "damnati";
    char rounds[] = "--rounds";
    char big[] = "2000000000";
    char *argv[] = {prog, rounds, big, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.rounds == 2000000000);
  }

  {
    Config cfg{};
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
    Config cfg{};
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

TEST_CASE("parse_cli validates boundary values for depth and probabilities",
          "[cli]") {
  {
    Config cfg{};
    char prog[] = "damnati";
    char depth[] = "--depth";
    char zero[] = "0";
    char *argv[] = {prog, depth, zero, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.depth == 0);
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char depth[] = "--depth";
    char max_depth[] = "15";
    char *argv[] = {prog, depth, max_depth, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.depth == MAX_NGRAM_DEPTH);
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char depth[] = "--depth";
    char too_large[] = "16";
    char *argv[] = {prog, depth, too_large, nullptr};
    int argc = 3;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for depth above max");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()) ==
              "Error: --depth must be in [0,15].");
    }
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char epsilon[] = "--epsilon";
    char zero[] = "0";
    char *argv[] = {prog, epsilon, zero, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.epsilon == Approx(0.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char epsilon[] = "--epsilon";
    char one[] = "1";
    char *argv[] = {prog, epsilon, one, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.epsilon == Approx(1.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char epsilon[] = "--epsilon";
    char below[] = "-0.01";
    char *argv[] = {prog, epsilon, below, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --epsilon must be in [0,1].");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char epsilon[] = "--epsilon";
    char above[] = "1.01";
    char *argv[] = {prog, epsilon, above, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --epsilon must be in [0,1].");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char pngram[] = "--p-ngram";
    char zero[] = "0";
    char *argv[] = {prog, pngram, zero, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.p_ngram == Approx(0.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char pngram[] = "--p-ngram";
    char one[] = "1";
    char *argv[] = {prog, pngram, one, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.p_ngram == Approx(1.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char pngram[] = "--p-ngram";
    char below[] = "-0.01";
    char *argv[] = {prog, pngram, below, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --p-ngram must be in [0,1].");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char pngram[] = "--p-ngram";
    char above[] = "1.01";
    char *argv[] = {prog, pngram, above, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --p-ngram must be in [0,1].");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char gtft[] = "--gtft";
    char zero[] = "0";
    char *argv[] = {prog, gtft, zero, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.gtft_p == Approx(0.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char gtft[] = "--gtft";
    char one[] = "1";
    char *argv[] = {prog, gtft, one, nullptr};
    int argc = 3;
    REQUIRE_NOTHROW(parse_cli(argc, argv, cfg));
    REQUIRE(cfg.gtft_p == Approx(1.0f));
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char gtft[] = "--gtft";
    char below[] = "-0.01";
    char *argv[] = {prog, gtft, below, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --gtft must be in [0,1].");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char gtft[] = "--gtft";
    char above[] = "1.01";
    char *argv[] = {prog, gtft, above, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: --gtft must be in [0,1].");
  }
}

TEST_CASE("parse_cli validates seed parsing failures", "[cli]") {
  {
    Config cfg{};
    char prog[] = "damnati";
    char seed[] = "--seed";
    char overflow[] = "18446744073709551616";
    char *argv[] = {prog, seed, overflow, nullptr};
    int argc = 3;
    REQUIRE_THROWS_WITH(parse_cli(argc, argv, cfg),
                        "Error: value for --seed is out of range: "
                        "'18446744073709551616'.");
  }

  {
    Config cfg{};
    char prog[] = "damnati";
    char seed[] = "--seed";
    char invalid[] = "123abc";
    char *argv[] = {prog, seed, invalid, nullptr};
    int argc = 3;
    try {
      parse_cli(argc, argv, cfg);
      FAIL("parse_cli should have thrown for invalid seed token");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()).find("invalid value for --seed") !=
              std::string::npos);
      REQUIRE(std::string(ex.what()).find("123abc") != std::string::npos);
    }
  }
}

TEST_CASE("run_gpu rejects tournaments that exceed launch capacity",
          "[cli][limits]") {
  Config cfg;
  cfg.n_agents = 1048577;
  cfg.p_ngram = 0.0f;
  cfg.rounds = 1;

  try {
    run_gpu(cfg);
    FAIL("run_gpu should have thrown for excessive tournament size");
  } catch (const std::runtime_error &ex) {
    REQUIRE(std::string(ex.what()).find("exceeding the maximum supported") !=
            std::string::npos);
  }
}

namespace {
cudaError_t mock_device_count_no_device(int *device_count) {
  *device_count = 0;
  return cudaSuccess;
}

cudaError_t mock_device_count_one_device(int *device_count) {
  *device_count = 1;
  return cudaSuccess;
}

cudaError_t mock_device_props_unsupported(cudaDeviceProp *props, int device) {
  (void)device;
  std::memset(props, 0, sizeof(cudaDeviceProp));
  std::strncpy(props->name, "Mock GPU", sizeof(props->name) - 1);
  props->major = 5;
  props->minor = 2;
  return cudaSuccess;
}
} // namespace

TEST_CASE("runtime compatibility messages include device details",
          "[cli][compat]") {
  const CudaDeviceQueryApi original = g_cuda_device_query_api;

  SECTION("no CUDA devices present") {
    g_cuda_device_query_api.get_device_count = mock_device_count_no_device;
    g_cuda_device_query_api.get_device_properties =
        mock_device_props_unsupported;

    try {
      ensure_runtime_compatibility();
      FAIL("ensure_runtime_compatibility should fail when no devices exist");
    } catch (const std::runtime_error &ex) {
      REQUIRE(std::string(ex.what()) ==
              "Error: no CUDA device detected. Install an NVIDIA driver/CUDA "
              "runtime and run on a GPU system.");
    }
  }

  SECTION("selected CUDA device is unsupported") {
    g_cuda_device_query_api.get_device_count = mock_device_count_one_device;
    g_cuda_device_query_api.get_device_properties =
        mock_device_props_unsupported;

    try {
      ensure_runtime_compatibility();
      FAIL("ensure_runtime_compatibility should fail for unsupported devices");
    } catch (const std::runtime_error &ex) {
      const std::string message = ex.what();
      REQUIRE(message.find("selected CUDA device Mock GPU (compute capability "
                           "5.2) is unsupported") != std::string::npos);
      REQUIRE(message.find("requires compute capability 6.0 or newer") !=
              std::string::npos);
    }
  }

  g_cuda_device_query_api = original;
}

TEST_CASE("summary header prints gtft probability", "[cli][summary]") {
  std::array<char, 256> buffer{};
  int written = std::snprintf(buffer.data(), buffer.size(), kSummaryHeaderFmt,
                              4, 10, 0.5f, 2, 0.1f, 0.25f);
  REQUIRE(written > 0);
  std::string header(buffer.data(), static_cast<std::size_t>(written));
  REQUIRE(header.find("\"gtft\":0.250") != std::string::npos);
}
