#include "cli.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <limits>
#include <stdexcept>
#include <string>

void print_usage(FILE *stream, const char *prog) {
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
std::string format_flag(const char *opt_name) { return std::string("--") + opt_name; }

int parse_int_option(const char *opt_name, const char *value) {
  errno = 0;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  const std::string arg = value;
  const std::string flag = format_flag(opt_name);
  if (end == value || *end != '\0') {
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg + "'.");
  }
  if (errno == ERANGE || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Error: value for " + flag + " is out of range: '" + arg + "'.");
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
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg + "'.");
  }
  if (errno == ERANGE || parsed > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("Error: value for " + flag + " is out of range: '" + arg + "'.");
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
    throw std::runtime_error("Error: invalid value for " + flag + ": '" + arg + "'.");
  }
  if (errno == ERANGE || !std::isfinite(parsed) ||
      parsed < -std::numeric_limits<float>::max() ||
      parsed > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Error: value for " + flag + " is out of range: '" + arg + "'.");
  }
  return static_cast<float>(parsed);
}
} // namespace

void parse_cli(int argc, char **argv, Config &cfg) {
  static const struct option long_opts[] = {{"agents", required_argument, nullptr, 'a'},
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
  while ((opt = getopt_long(argc, argv, ":a:r:s:p:d:e:g:h", long_opts, nullptr)) != -1) {
    switch (opt) {
    case 'a':
      cfg.n_agents = parse_int_option("agents", optarg);
      if (cfg.n_agents < 2)
        throw std::runtime_error("Error: --agents must be at least 2.");
      break;
    case 'r':
      cfg.rounds = parse_int_option("rounds", optarg);
      if (cfg.rounds <= 0)
        throw std::runtime_error("Error: --rounds must be positive.");
      break;
    case 's':
      cfg.seed = parse_uint64_option("seed", optarg);
      break;
    case 'p':
      cfg.p_ngram = parse_float_option("p-ngram", optarg);
      if (cfg.p_ngram < 0.0f || cfg.p_ngram > 1.0f)
        throw std::runtime_error("Error: --p-ngram must be in [0,1].");
      break;
    case 'd':
      cfg.depth = parse_int_option("depth", optarg);
      if (cfg.depth < 0 || cfg.depth > MAX_NGRAM_DEPTH)
        throw std::runtime_error(std::string("Error: --depth must be in [0,") +
                                 std::to_string(MAX_NGRAM_DEPTH) + "].");
      break;
    case 'e':
      cfg.epsilon = parse_float_option("epsilon", optarg);
      if (cfg.epsilon < 0.0f || cfg.epsilon > 1.0f)
        throw std::runtime_error("Error: --epsilon must be in [0,1].");
      break;
    case 'g':
      cfg.gtft_p = parse_float_option("gtft", optarg);
      if (cfg.gtft_p < 0.0f || cfg.gtft_p > 1.0f)
        throw std::runtime_error("Error: --gtft must be in [0,1].");
      break;
    case 'h':
      print_usage(stdout, argv[0]);
      std::exit(0);
    case ':': {
      const char *flag_name = nullptr;
      switch (optopt) {
      case 'a': flag_name = "--agents"; break;
      case 'r': flag_name = "--rounds"; break;
      case 's': flag_name = "--seed"; break;
      case 'p': flag_name = "--p-ngram"; break;
      case 'd': flag_name = "--depth"; break;
      case 'e': flag_name = "--epsilon"; break;
      case 'g': flag_name = "--gtft"; break;
      default: break;
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
      std::string flag = (optind > 0 && optind - 1 < argc) ? std::string(argv[optind - 1]) : std::string();
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
    throw std::runtime_error("Error: unexpected positional argument '" + std::string(argv[optind]) + "'.");
  }
}
