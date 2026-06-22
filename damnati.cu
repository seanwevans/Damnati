#include <cstdio>
#include <cstdlib>
#include <exception>

#include "src/app.hpp"
#include "src/cli.hpp"

int main(int argc, char **argv) {
  Config cfg;
  try {
    parse_cli(argc, argv, cfg);
    ensure_runtime_compatibility();
    run_gpu(cfg);
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
