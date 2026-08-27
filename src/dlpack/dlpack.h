#ifndef FASTEMBEDR_DLPACK_DLPACK_H_
#define FASTEMBEDR_DLPACK_DLPACK_H_

/*
 * SPDX-FileCopyrightText: Copyright DMLC contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal DLPack C ABI header for the optional RAPIDS cuVS backend.
 *
 * Reduced from dmlc/dlpack v1.0 commit
 * bbd2f4d32427e548797929af08cfe2a9cbb3cf12,
 * include/dlpack/dlpack.h. fastEmbedR retains only the stable tensor structs
 * and enum names needed to pass dense matrices through the cuVS C API. The
 * complete Apache-2.0 license is installed as LICENSES/DLPACK-LICENSE.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DLPACK_MAJOR_VERSION 1
#define DLPACK_MINOR_VERSION 0

typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLOpaqueHandle = 3,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}
#endif

#endif
