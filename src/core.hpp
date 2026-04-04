#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

inline void throw_if_cuda_error(cudaError_t err, const char *expr,
                                const char *file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  std::string message = "CUDA error at ";
  message += file;
  message += ':';
  message += std::to_string(line);
  message += " for ";
  message += expr;
  message += ": ";
  message += cudaGetErrorString(err);
  throw std::runtime_error(message);
}

constexpr int Rw = 3;
constexpr int Sw = 0;
constexpr int Tw = 5;
constexpr int Pw = 1;

enum Act : int { C = 0, D = 1 };

enum Strategy : int {
  AC = 0,
  AD,
  TFT,
  GTFT,
  GRIM,
  RANDOM,
  PAVLOV,
  ALT,
  JOSS,
  TESTER,
  REPEAT,
  S_TFT,
  NGRAM
};

constexpr std::size_t INVALID_OFFSET = static_cast<std::size_t>(-1);

struct AgentParams {
  int strat;
  float epsilon;
  int depth;
  float gtft_forget;
  int *counts;
  float *q;
};

extern __device__ __constant__ int d_payA[4];
extern __device__ __constant__ int d_payB[4];

