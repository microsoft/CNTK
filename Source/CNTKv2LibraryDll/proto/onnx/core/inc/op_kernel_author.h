//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include "core/common/ml_status.h"

// Disable formatting, which is incorrect for ML_API macros
// clang-format off

namespace onnxruntime {

// TODO - calling convention
#if defined(__GNUC__)
#define ML_API(name) virtual MLStatus name
#define ML_API_IMP(name) MLStatus name
#define ML_API_(returnType, name) virtual returnType name
#define ML_API_IMP_(returnType, name) returnType name
#define ML_CALLBACK_API(name) MLStatus(*name)
#else
#define ML_API(name) virtual MLStatus __stdcall name
#define ML_API_IMP(name) MLStatus __stdcall name
#define ML_API_(returnType, name) virtual returnType __stdcall name
#define ML_API_IMP_(returnType, name) returnType __stdcall name
#define ML_CALLBACK_API(name) MLStatus(*name)
#endif

#define ML_DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE)                                                                         \
  static_assert(sizeof(ENUMTYPE) == sizeof(uint32_t), "Incompatible enumeration size");                                  \
  inline constexpr ENUMTYPE operator|(ENUMTYPE a, ENUMTYPE b) throw() { return ENUMTYPE(((uint32_t)a) | ((uint32_t)b)); } \
  inline ENUMTYPE& operator|=(ENUMTYPE& a, ENUMTYPE b) throw() { return (ENUMTYPE&)(((uint32_t&)a) |= ((uint32_t)b)); }   \
  inline constexpr ENUMTYPE operator&(ENUMTYPE a, ENUMTYPE b) throw() { return ENUMTYPE(((uint32_t)a) & ((uint32_t)b)); } \
  inline ENUMTYPE& operator&=(ENUMTYPE& a, ENUMTYPE b) throw() { return (ENUMTYPE&)(((uint32_t&)a) &= ((uint32_t)b)); }   \
  inline constexpr ENUMTYPE operator~(ENUMTYPE a) throw() { return ENUMTYPE(~((uint32_t)a)); }                           \
  inline constexpr ENUMTYPE operator^(ENUMTYPE a, ENUMTYPE b) throw() { return ENUMTYPE(((uint32_t)a) ^ ((uint32_t)b)); } \
  inline ENUMTYPE& operator^=(ENUMTYPE& a, ENUMTYPE b) throw() { return (ENUMTYPE&)(((uint32_t&)a) ^= ((uint32_t)b)); }

static_assert(sizeof(bool) == 1, "Unsupported size for bool");

// Attribute types with numeric values matching the ONNX specification
enum class MLAttributeType : uint32_t {
  kUndefined = 0,
  kFloat = 2,
  kInt = 3,
  kString = 4,
  kFloatArray = 7,
  kIntArray = 8,
  kStringArray = 9
};

enum class MLTensorDataType : uint32_t {
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

union MLFloat16 {
  uint16_t val;
  
  explicit MLFloat16(uint16_t x) : val(x) {}
  MLFloat16() : val(0) {}
};

inline bool operator==(const MLFloat16& left, const MLFloat16& right)
{
  return left.val == right.val;
}

inline bool operator!=(const MLFloat16& left, const MLFloat16& right)
{
  return left.val != right.val;
}

struct MLMapType {
  MLTensorDataType data_type;
  MLTensorDataType value_type;
};

enum class MLEdgeClass : uint32_t {
  kUndefined = 0,
  kTensor = 1,
  kMap = 2,
  kTensorSequence = 3,
  kMapSequence = 4,
};

// Edge information used by schema during inferencing and provided to operator
// kernel factory methods.
struct MLEdgeType {
  MLEdgeClass edge_class;

  union {
    MLTensorDataType tensor_data_type;
    MLMapType map_type;

    int64_t reserved;
  };
};

// Operator information used by kernel creation methods and inferencing functions
class IMLOperatorAttributes {
 public:
  // Gets the count of elements in an attribute.  May be used to determine if an
  // attribute of any type exists.
  ML_API(GetAttributeElementCount)(
      MLAttributeType type,
      const char* name,
      uint32_t* element_count) const noexcept = 0;

  // Gets the array of values in a numeric attribute
  ML_API(GetAttribute)(
      const char* name,
      MLAttributeType type,
      uint32_t element_count,
      uint32_t element_byte_size,
      void* value) const noexcept = 0;

  // Gets the length of an element within a UTF-8 string attribute,
  // including null termination
  ML_API(GetStringAttributeElementLength)(
      const char* name,
      uint32_t element_index,
      uint32_t* attribute_element_length) const noexcept = 0;

  // Gets the contents of an element within a UTF-8 string attribute.  The size
  // includes null termination.
  ML_API(GetStringAttributeElement)(
      const char* name,
      uint32_t element_index,
      uint32_t attribute_element_length,
      char* attribute_element) const noexcept = 0;
};

// Shape information used by kernel implementations
class IMLOpKernelTensorShapeInfo {
 public:
  ML_API(GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept = 0;
  ML_API(GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept = 0;

  // HasOutputShapeInfo returns false if and only if the kernel was registered with
  // kProducesDynamicOutputTensorSize. Otherise, shape inference functions are required
  // to have been provided by the kernel registration.
  ML_API_(bool, HasOutputShapeInfo)() const noexcept = 0;
  ML_API(GetOutputTensorDimensionCount)(uint32_t output_index, uint32_t* dimension_count) const noexcept = 0;
  ML_API(GetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, int64_t* dimensions) const noexcept = 0;
};

// Operator information provided to operator kernel factory methods.
class IMLOpKernelInfo : public IMLOperatorAttributes {
 public:
  ML_API_(uint32_t, GetInputCount)() const noexcept = 0;
  ML_API_(uint32_t, GetOutputCount)() const noexcept = 0;

  ML_API(GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept = 0;
  ML_API(GetOutputEdgeType)(uint32_t output_index, MLEdgeType* edge_type) const noexcept = 0;

  // HasTensorShapeInfo returns false if and only if the kernel is registered using
  // MLOpKernelOptions::kAllowDynamicInputTensorSizes.  If this flag is specified and upstream
  // shapes are known when the kernel is created, HasTensorShapeInfo still returns false.
  ML_API_(bool, HasTensorShapeInfo)() const noexcept = 0;
  ML_API(GetTensorShapeInfo)(const IMLOpKernelTensorShapeInfo** shapeInfo) const noexcept = 0;

  // Returns a handle whose type varies based on the kernel type.
  ML_API_(const void*, GetExecutionHandle)() const noexcept = 0;
};

// Tensors methods used by implementations of IMLOpKernel::Compute
class IMLOpTensor {
 public:
  ML_API_(uint32_t, GetDimensionCount)() const noexcept = 0;

  ML_API(GetDimensions)(
      int64_t* dimensions,
      uint32_t dimension_count) const noexcept = 0;

  ML_API_(MLTensorDataType, GetTensorDataType)() const noexcept = 0;

  // Whether the tensor's memory is CPU-addressible.  This is controlled
  // by the registration parameters of the kernel.
  ML_API_(bool, IsCPUData)() const noexcept = 0;

  // Whether the tensor's memory is a handle type, such as an interface object.
  // This is controlled by the registration parameters of the kernel.
  // This returns false for tensors with blobs of raw CPU or device memory.  If
  // this returns true, then the caller may cast or offset the pointer returned
  // by GetData().
  ML_API_(bool, IsDataHandle)() const noexcept = 0;

  // Returns a pointer whose type varies  based on the kernel type.
  ML_API_(void*, GetData)() noexcept = 0;
  ML_API_(const void*, GetData)() const noexcept = 0;

  // Whether this tensor is an unused optional input/output tensors
  ML_API_(bool, IsUnused)() const noexcept = 0;

  // TODO - Methods to access strings stored within tensors
};

// Context used by IMLOpKernel::Compute
class IMLOpKernelContext {
 public:
  ML_API(GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept = 0;

  // If the kernel is registered without a shape inference method, then the overload of
  // GetOutputTensor consuming the tensor's shape must be called.
  ML_API(GetOutputTensor)(uint32_t output_index, IMLOpTensor** tensor) noexcept = 0;

  ML_API(GetOutputTensor)(
      uint32_t output_index,
      const int64_t* dimension_sizes,
      uint32_t dimensions,
      IMLOpTensor** tensor) noexcept = 0;

  // TODO - methods to query maps and sequences

  // Allocate and free intermediate resources.  The allocation will automatically
  // be maintained as necessary until after the IMLOpKernel::Compute returns and
  // any GPU work scheduled during that routine completes.
  ML_API(AllocateTemporaryData)(uint64_t size, void** data) const = 0;
  ML_API(FreeTemporaryData)(void* data) const = 0;

  // Returns a handle whose type varies based on the kernel type.
  ML_API_(const void*, GetExecutionHandle)() const noexcept = 0;
};

class IMLOpKernel {
 public:
  ML_API_(void, Release)() noexcept = 0;

  // Computes the outputs of the kernel.  This may be called multiple times
  // simultaneously within the same instance of the class.  Implementations
  // of this method must be thread-safe.
  ML_API(Compute)(IMLOpKernelContext* context) noexcept = 0;
};

enum class MLFormalParameterOptions : uint32_t {
  kSingle = 0,
  kOptional = 1,
  kVariadic = 2,
};

enum class MLFormalParameterTypeFormat {
  // The type is defined using MLEdgeType
  kEdgeType = 0,

  // The type is a string which is part of the operator definition and described in its schema
  kLabel = 1,
};

struct MLFormalParameter {
  MLFormalParameterOptions options;

  MLFormalParameterTypeFormat type_format;
  union {
    const char* type_label;
    MLEdgeType edge_type;
  };
};

struct MLTypeConstraint {
  const char* type_label;
  const MLEdgeType* allowed_types;
  uint32_t allowed_type_count;
};

class IMLShapeInferenceContext : public IMLOperatorAttributes {
 public:
  ML_API_(uint32_t, GetInputCount)() const noexcept = 0;
  ML_API_(uint32_t, GetOutputCount)() const noexcept = 0;

  ML_API(GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept = 0;
  ML_API(GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept = 0;
  ML_API(GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept = 0;

  ML_API(SetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, const int64_t* dimensions) noexcept = 0;
};

class IMLTypeInferenceContext : public IMLOperatorAttributes {
 public:
  ML_API_(uint32_t, GetInputCount)() const noexcept = 0;
  ML_API_(uint32_t, GetOutputCount)() const noexcept = 0;

  ML_API(GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept = 0;
  ML_API(SetOutputEdgeType)(uint32_t output_index, const MLEdgeType* edge_type) const noexcept = 0;
};

// Inference function to compute the output types. This should be used in cases where
// MLSchemaDefinition cannot express an operator's type mapping declaratively.
using MLTypeInferenceFunction = MLStatus (*)(void *, IMLTypeInferenceContext *);

// Inference function to compute sizes of output tensors.
// All input tensors provided to the shape inference callback will have well defined sizes.
// If upstream operators cannot determine their output shape before computation, then this
// will be called only after their computation.
using MLShapeInferenceFunction = MLStatus (*)(void *, IMLShapeInferenceContext *);

struct MLAttribute {
  const char* name;
  MLAttributeType type;
  bool required;
};

// Attribute name and value pairs.  Used to supply default attribute values.
struct MLAttributeNameValue {
  const char* name;
  MLAttributeType type;
  uint32_t value_count;

  union {
    const int64_t* ints;
    const char* const* strings;
    const float* floats;
  };
};

// Definitions of operators which are independent of kernel implementations
struct MLSchemaDefinition {
  const char* name;

  // The operator set version at which this operator was introduced with most recent change
  // For example, ONNX 1.2 exposes up to version 7 of the operator set for the ONNX domain.
  int operator_set_since_version;

  const MLFormalParameter* inputs;
  uint32_t input_count;

  const MLFormalParameter* outputs;
  uint32_t output_count;

  const MLTypeConstraint* type_constraints;
  uint32_t type_constraint_count;

  // The provided context is passed to the function
  MLTypeInferenceFunction type_inference_function;
  void* type_inference_function_context;

  const MLAttribute* attributes;
  uint32_t attribute_count;

  // Default attributes, used for validation.  Default attributes provided
  // when registering kernels must be consistent.  Only the defaults provided
  // in schema registrations are used to automatically set missing values.
  const MLAttributeNameValue* default_attributes;
  uint32_t default_attribute_count;

  // Optional shape inference function, used for validation.
  // This may be the same function as provided to MLOpKernelDefinition.
  // The provided context is passed to the function.
  MLShapeInferenceFunction shape_inference_function;
  void* shape_inference_function_context;
};

struct MLOperatorSetId {
  // The domain of the operator, for example, "ai.onnx.ml", or an empty string
  // for the ONNX domain.
  const char* domain;

  int version;
};

struct MLOpKernelDefinition {
  const char* domain;
  const char* name;

  // The operator version at which this kernel becomes valid.  The maximum valid
  // version of the kernel is inferred based on registrations of schema for operator
  // sets containing breaking changes.
  int operator_set_since_version;

  // Type of kernel, for example "CPUExecutionProvider"
  const char* execution_provider_name;

  MLTypeConstraint* type_constraints;
  uint32_t type_constraint_count;

  // Default attributes, used for automatically setting missing values.
  // Default attributes provided in schema registrations must be consistent.
  // Only the defaults provided in kernel registrations are used to automatically
  // set missing values.
  const MLAttributeNameValue* default_attributes;
  uint32_t default_attribute_count;

  // Optional shape inference function, used for validation and memory planning.
  // This may be the same function as provided to MLSchemaDefinition.
  // If this is provided, IMLOpKernelContext::GetOutputTensor may be called
  // while not providing the output tensor shape.  The provided context is 
  // passed to shape_inference_function.
  MLShapeInferenceFunction shape_inference_function;
  void* shape_inference_function_context;
};

// TODO - Make this store a context value or allow interfaces to be registered
using IMLOpKernelCreateFn = MLStatus (*)(const IMLOpKernelInfo &, IMLOpKernel **);

enum class MLOpKernelOptions : uint32_t {
  kNone = 0,

  // Whether the shapes of input tensors are allowed to vary across invocations
  // of an operator kernel instance.  If this is not set, kernel instances may query input
  // tensor shapes during creation, and front-load initialization work which depends
  // on those shapes.  Setting this may improve performance in some cases by enabling
  // a kernel instance to be re-used with different input sizes, but caches accumulated
  // by kernels during computation must be managed in a thread-safe fashion.
  kAllowDynamicInputShapes = 1,
};

ML_DEFINE_ENUM_FLAG_OPERATORS(MLOpKernelOptions)

// Operator and kernel registrations. Registrations may be overridden by subsequent registrations
// of the same operator.
class IMLOperatorRegistry {
 public:
  // The operator set registration must provide schema for all operators that have changed since
  // the specified baseline version.
  ML_API(RegisterOpSetFromSchema)(
      const MLOperatorSetId* opSetId,
      int baseline_version,
      const MLSchemaDefinition* const* schema,
      uint32_t schema_count) const noexcept = 0;

  ML_API(RegisterOpKernel)(
      const MLOpKernelDefinition* op_kernel,
      MLOpKernelOptions options,
      IMLOpKernelCreateFn op_kernel_factory) const noexcept = 0;
};

} // namespace onnxruntime
