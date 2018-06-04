//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include "ml_status.h"

typedef uint8_t MLBool;

// TODO - calling convention for former case
#if defined(__GNUC__)
#define ML_API(name) virtual MLStatus name
#define ML_API_IMP(name) MLStatus name
#define ML_API_(returnType, name) virtual returnType name
#define ML_API_IMP_(returnType, name) returnType name
#else
#define ML_API(name) virtual MLStatus __stdcall name
#define ML_API_IMP(name) MLStatus __stdcall name
#define ML_API_(returnType, name) virtual returnType __stdcall name
#define ML_API_IMP_(returnType, name) returnType __stdcall name
#endif

// Attribute types with numeric values matching the ONNX specification
enum class MLAttributeType {
  kUndefined = 0,
  kFloat = 2,
  kInt = 3,
  kString = 4,
  kFloatArray = 7,
  kIntArray = 8,
  kStringArray = 9
};

enum class MLTensorDataType {
  kUndefined = 0,
  kFloat = 1,
  kUInt8 = 2,
  kInt8 = 3,
  kUInt16 = 4,
  kInt16 = 5,
  kInt32 = 6,
  kInt64 = 7,
  kString = 8,
  kBool = 9,
  kFloat16 = 10,
  kDouble = 11,
  kUInt32 = 12,
  kUInt64 = 13,
  kComplex64 = 14,
  kComplex128 = 15
};

class IMLOpKernelInfo {
 public:
  // Gets the count of elements in an attribute
  ML_API(GetAttributeElementCount)
  (
      MLAttributeType type,
      const char* name,
      uint32_t* element_count) const noexcept = 0;

  // Gets the array of values in a numeric attribute
  ML_API(GetAttribute)
  (
      const char* name,
      MLAttributeType type,
      uint32_t element_count,
      uint32_t element_byte_size,
      void* value) const noexcept = 0;

  // Gets the length of an element within a UTF-8 string attribute,
  // including null termination
  ML_API(GetStringAttributeElementLength)
  (
      const char* name,
      uint32_t element_index,
      uint32_t* attribute_element_length) const noexcept = 0;

  // Gets the contents of an element within a UTF-8 string attribute.  The size
  // includes null termination.
  ML_API(GetStringAttributeElement)
  (
      const char* name,
      uint32_t element_index,
      uint32_t attribute_element_length,
      char* attribute_element) const noexcept = 0;

  // Returns a handle whose type varies based on the kernel type.
  // For D3D operator kernels, this may return an IUnknown supporting QueryInterface to
  // ID3D12GraphicsCommandList1.
  ML_API_(const void*, GetExecutionHandle)
  () const noexcept = 0;
};

// Tensors methods used by implementations of IMLOpKernel::Compute
class IMLOpTensor {
 public:
  ML_API(GetDimensionCount)
  (uint32_t* dimensions) const = 0;

  ML_API(GetDimensions)
  (
      int64_t* dimensions,
      uint32_t dimension_count) const noexcept = 0;

  ML_API_(MLTensorDataType, GetTensorDataType)
  () const noexcept = 0;

  // Whether the tensor's memory is CPU-addressible.  This is controlled
  // by the registration parameters of the kernel.
  ML_API_(MLBool, IsCPUData)
  () const noexcept = 0;

  // Whether the tensor's memory is a handle type, such as an interface object.
  // This is controlled by the registration parameters of the kernel.
  // This returns false for tensors with blobs of raw CPU or device memory.  If
  // this returns true, then the caller may cast or offset the pointer returned
  // by GetData().
  ML_API_(MLBool, IsDataHandle)
  () const noexcept = 0;

  // Returns a pointer whose type varies  based on the kernel type.
  // For D3D kernels this returns a pointer to an IUnknown supporting QueryInterface
  // to ID3D12Resource.
  ML_API_(void*, GetData)
  () noexcept = 0;
  ML_API_(const void*, GetData)
  () const noexcept = 0;
};

enum class MLEdgeType {
  kUndefined = 0,
  kTensor = 1,
  kMap = 2,
  kSequence = 3
};

class IMLOpKernelContext {
 public:
  ML_API(GetInputEdgeType)
  (uint32_t input_index, MLEdgeType* edge_type) const noexcept = 0;
  ML_API(GetOutputEdgeType)
  (uint32_t output_index, MLEdgeType* edge_type) const noexcept = 0;

  ML_API(GetInputTensor)
  (uint32_t input_index, const IMLOpTensor** tensor) const noexcept = 0;
  ML_API(GetOutputTensor)
  (uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept = 0;

  ML_API_(uint32_t, GetInputCount)
  () const noexcept = 0;
  ML_API_(uint32_t, GetOutputCount)
  () const noexcept = 0;

  ML_API(AllocateTemporaryData)
  (uint64_t size, void** data) const = 0;
  ML_API(FreeTemporaryData)
  (void* data) const = 0;
};

class IMLOpKernel {
 public:
  ML_API_(void, Release)
  () noexcept = 0;

  // Allocates and computes the outputs of the kernel.  The same IMLOpKernelInfo
  // is provided as to the Initialize method.  Tensors within the input and output
  // arrays have fully packed strides and have NCHW channel ordering.
  //
  // D3D kernels must assume each tensor is initially in the UAV state and should ensure
  // they are in the UAV state when returning.  Kernels must not depend on previous state set
  // within the command list.  The command list is executed on a compute queue,
  // and must contain only compute work.
  //
  // D3D kernels should cache pipeline state objects which they use within the command list
  // using ID3D12Object::SetPrivateDataInterface.
  ML_API(Compute)
  (const IMLOpKernelInfo* info, IMLOpKernelContext* context) noexcept = 0;
};

typedef MLStatus (*IMLOpKernelCreateFn)(const IMLOpKernelInfo& kernelInfo, IMLOpKernel** opKernel);
