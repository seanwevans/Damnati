#pragma once

#include <string>
#include <vector>

#include "cli.hpp"
#include "core.hpp"

int build_population(const Config &cfg, std::vector<AgentParams> &hparams);
using CudaGetDeviceCountFn = cudaError_t (*)(int *);
using CudaGetDevicePropertiesFn = cudaError_t (*)(cudaDeviceProp *, int);

struct CudaDeviceQueryApi {
  CudaGetDeviceCountFn get_device_count = cudaGetDeviceCount;
  CudaGetDevicePropertiesFn get_device_properties = cudaGetDeviceProperties;
};

extern CudaDeviceQueryApi g_cuda_device_query_api;

void ensure_runtime_compatibility();
void run_gpu(const Config &cfg);

constexpr int kMinComputeCapabilityMajor = 6;
constexpr int kMinComputeCapabilityMinor = 0;

using CudaGetDeviceCountFn = cudaError_t (*)(int *);
using CudaGetDevicePropertiesFn = cudaError_t (*)(cudaDeviceProp *, int);

struct CudaDeviceQueryApi {
  CudaGetDeviceCountFn get_device_count = cudaGetDeviceCount;
  CudaGetDevicePropertiesFn get_device_properties = cudaGetDeviceProperties;
};

extern CudaDeviceQueryApi g_cuda_device_query_api;

std::string describe_device_capability(const cudaDeviceProp &props);
void ensure_runtime_compatibility();
