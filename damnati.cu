#include <cstdio>
#include <cstdlib>
#include <exception>

#include "src/cli.cu"
#include "src/strategy.cu"
#include "src/kernel.cu"
#include "src/reporting.cu"
#include "src/app.cu"

#ifndef DAMNATI_NO_MAIN
int main(int argc, char **argv) {
  Config cfg;
  try {
    parse_cli(argc, argv, cfg);
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return EXIT_FAILURE;
  }
  try {
    run_gpu(cfg);
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return EXIT_FAILURE;
  }
  return 0;
}
#endif
