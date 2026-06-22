#pragma once

#include <cstddef>
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

struct CudaApi {
  cudaError_t (*malloc_fn)(void **, std::size_t) = cudaMalloc;
  cudaError_t (*malloc_managed_fn)(void **, std::size_t) = cudaMallocManaged;
  cudaError_t (*memset_fn)(void *, int, std::size_t) = cudaMemset;
  cudaError_t (*free_fn)(void *) = cudaFree;
  cudaError_t (*get_last_error_fn)() = cudaGetLastError;
  cudaError_t (*device_synchronize_fn)() = cudaDeviceSynchronize;
};

inline CudaApi &cuda_api() {
  static CudaApi api{};
  return api;
}

inline void set_cuda_api_for_testing(const CudaApi &api) { cuda_api() = api; }
inline void reset_cuda_api_for_testing() { cuda_api() = CudaApi{}; }

inline cudaError_t cuda_malloc(void **ptr, std::size_t size) {
  return cuda_api().malloc_fn(ptr, size);
}
inline cudaError_t cuda_malloc_managed(void **ptr, std::size_t size) {
  return cuda_api().malloc_managed_fn(ptr, size);
}
inline cudaError_t cuda_memset(void *ptr, int value, std::size_t size) {
  return cuda_api().memset_fn(ptr, value, size);
}
inline cudaError_t cuda_free(void *ptr) { return cuda_api().free_fn(ptr); }
inline cudaError_t cuda_get_last_error() {
  return cuda_api().get_last_error_fn();
}
inline cudaError_t cuda_device_synchronize() {
  return cuda_api().device_synchronize_fn();
}

struct CudaPtr {
  void *p = nullptr;

  CudaPtr() = default;
  explicit CudaPtr(void *ptr) : p(ptr) {}
  ~CudaPtr() { reset(); }

  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  CudaPtr(CudaPtr &&other) noexcept : p(other.p) { other.p = nullptr; }
  CudaPtr &operator=(CudaPtr &&other) noexcept {
    if (this != &other) {
      reset();
      p = other.p;
      other.p = nullptr;
    }
    return *this;
  }

  void reset(void *next = nullptr) {
    if (p != nullptr) {
      (void)cuda_free(p);
    }
    p = next;
  }

  void **out() { return &p; }

  template <typename T> T *as() const { return static_cast<T *>(p); }
  explicit operator bool() const { return p != nullptr; }
};

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
