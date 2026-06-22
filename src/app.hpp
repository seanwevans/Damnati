#pragma once

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
