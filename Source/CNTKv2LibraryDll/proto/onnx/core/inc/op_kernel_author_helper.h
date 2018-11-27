//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include "core/inc/op_kernel_author.h"
#include <limits>
#include <string>
#include <vector>
#include <memory>

// Disable formatting, which is incorrect for ML_API macros
// clang-format off
namespace onnxruntime {
using MLConstStringParam = const char*;

class MLOpKernelContext;

// TODO - Consider using this directly in onnxruntime and merging error handling
class MLStatusException : public std::exception {
 public:
  MLStatusException(const MLStatus& status) : status_(status) {
  }

  MLStatus GetStatus() const noexcept {
    return status_;
  }

  const char* what() const noexcept override {
    return MLStatusToString(status_);
  }

 private:
  MLStatus status_;
};

#define ML_CHECK_STATUS(x)        \
  {                               \
    if ((x) != MLStatus::OK) {    \
      throw MLStatusException(x); \
    }                             \
  }

// TODO - consume error code to be returned upon failure
#define ML_CHECK_BOOL(x)                       \
  {                                            \
    if ((x) == false) {                        \
      throw MLStatusException(MLStatus::FAIL); \
    }                                          \
  }

//
// Traits for numeric attribute types
//
template <typename T>
struct MLTypeTraits {
};

template <>
struct MLTypeTraits<float> {
  static const MLAttributeType AttributeType = MLAttributeType::kFloat;
  static const MLAttributeType AttributeVectorType = MLAttributeType::kFloatArray;
  static const MLTensorDataType TensorType = MLTensorDataType::kFloat;
};

template <>
struct MLTypeTraits<int32_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kInt32;
};

template <>
struct MLTypeTraits<uint8_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kUInt8;
};

template <>
struct MLTypeTraits<int8_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kInt8;
};

template <>
struct MLTypeTraits<uint16_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kUInt16;
};

template <>
struct MLTypeTraits<int16_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kInt16;
};

template <>
struct MLTypeTraits<int64_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kInt64;
  static const MLAttributeType AttributeType = MLAttributeType::kInt;
  static const MLAttributeType AttributeVectorType = MLAttributeType::kIntArray;
};

template <>
struct MLTypeTraits<bool> {
  static const MLTensorDataType TensorType = MLTensorDataType::kBool;
};

// TODO - non-primitive traits classes: string, float16, complex64, complex128

template <>
struct MLTypeTraits<double> {
  static const MLTensorDataType TensorType = MLTensorDataType::kDouble;
};

template <>
struct MLTypeTraits<uint32_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kUInt32;
};

template <>
struct MLTypeTraits<uint64_t> {
  static const MLTensorDataType TensorType = MLTensorDataType::kUInt64;
};

template <>
struct MLTypeTraits<MLFloat16> {
  static const MLTensorDataType TensorType = MLTensorDataType::kFloat16;
};

//
// Wrappers for ABI objects consumed by kernels.
// These wrappers provide typesafe methods which use STL types and convert
// return values to exceptions.
//

class MLOpKernelTensorShapeInfo {
 public:
  MLOpKernelTensorShapeInfo(const IMLOpKernelTensorShapeInfo* impl) : impl_(impl) {}

  uint32_t GetInputTensorDimensionCount(uint32_t input_index) const {
    uint32_t ret;
    ML_CHECK_STATUS(impl_->GetInputTensorDimensionCount(input_index, &ret));
    return ret;
  }

  std::vector<int64_t> GetInputTensorShape(uint32_t input_index) const {
    std::vector<int64_t> ret;
    uint32_t dimension_count = GetInputTensorDimensionCount(input_index);
    ret.resize(dimension_count);

    ML_CHECK_STATUS(impl_->GetInputTensorShape(input_index, dimension_count, ret.data()));
    return ret;
  }

  bool HasOutputShapeInfo() const noexcept {
    return impl_->HasOutputShapeInfo();
  }

  uint32_t GetOutputTensorDimensionCount(uint32_t output_index) const {
    uint32_t ret;
    ML_CHECK_STATUS(impl_->GetOutputTensorDimensionCount(output_index, &ret));
    return ret;
  }

  std::vector<int64_t> GetOutputTensorShape(uint32_t output_index) const {
    std::vector<int64_t> ret;
    uint32_t dimension_count = GetOutputTensorDimensionCount(output_index);
    ret.resize(dimension_count);

    ML_CHECK_STATUS(impl_->GetOutputTensorShape(output_index, dimension_count, ret.data()));
    return ret;
  }

  const IMLOpKernelTensorShapeInfo* GetInterface() const { return impl_; }

 protected:
  const IMLOpKernelTensorShapeInfo* impl_ = nullptr;
};

class MLOperatorAttributes {
 public:
  MLOperatorAttributes(const IMLOperatorAttributes* impl) : impl_(impl) {
  }

  uint32_t GetAttributeElementCount(
      MLAttributeType type, MLConstStringParam name) const {
    uint32_t element_count;
    ML_CHECK_STATUS(impl_->GetAttributeElementCount(type, name, &element_count));
    return element_count;
  }

  bool HasAttribute(MLAttributeType type, MLConstStringParam name) const noexcept {
    return GetAttributeElementCount(type, name) > 0;
  }

  //
  // Templatized methods to query numeric attributes using MLTypeTraits
  //
  template <typename T>
  T GetAttribute(MLConstStringParam name) const {
    T value;

    ML_CHECK_STATUS(impl_->GetAttribute(
        name,
        MLTypeTraits<T>::AttributeType,
        1,
        sizeof(T),
        &value));

    return value;
  }

  template <typename T>
  std::vector<T> GetAttributeVector(MLConstStringParam name) const {
    uint32_t count = GetAttributeElementCount(MLTypeTraits<T>::AttributeVectorType, name);
    std::vector<T> values(count);

    ML_CHECK_STATUS(impl_->GetAttribute(
        name,
        MLTypeTraits<T>::AttributeVectorType,
        count,
        sizeof(T),
        values.data()));

    return values;
  }

  std::string GetAttribute(MLConstStringParam name) const {
    return GetAttributeElement(name, 0);
  }

  std::vector<std::string> GetAttributeVector(MLConstStringParam name) const {
    uint32_t count = GetAttributeElementCount(MLAttributeType::kStringArray, name);
    std::vector<std::string> values;
    values.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
      values[i] = GetAttributeElement(name, i);
    }

    return values;
  }

  std::string GetAttributeElement(MLConstStringParam name, uint32_t element_index) const {
    uint32_t length = 0;
    ML_CHECK_STATUS(impl_->GetStringAttributeElementLength(name, element_index, &length));

    // Construct a string by copying a character array.  The copy can be removed with C++17
    // using the non-const std::basic_string::data method.
    std::vector<char> temp(length);
    ML_CHECK_STATUS(impl_->GetStringAttributeElement(name, element_index, length, temp.data()));
    std::string value(temp.data());
    return value;
  }

 private:
  const IMLOperatorAttributes* impl_;
};

class MLOpKernelInfo : public MLOperatorAttributes {
 public:
  MLOpKernelInfo(const IMLOpKernelInfo* impl) : MLOperatorAttributes(impl), impl_(impl) {}

  // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
  const IMLOpKernelInfo* GetInterface() const noexcept {
    return impl_;
  }

  const void* GetExecutionHandle() const noexcept {
    return impl_->GetExecutionHandle();
  }

  uint32_t GetInputCount() const noexcept {
    return impl_->GetInputCount();
  }

  uint32_t GetOutputCount() const noexcept {
    return impl_->GetOutputCount();
  }

  MLEdgeType GetInputEdgeType(uint32_t input_index) const {
    MLEdgeType ret;
    ML_CHECK_STATUS(impl_->GetInputEdgeType(input_index, &ret));

    return ret;
  }

  MLEdgeType GetOutputEdgeType(uint32_t output_index) const {
    MLEdgeType ret = {};
    ML_CHECK_STATUS(impl_->GetOutputEdgeType(output_index, &ret));

    return ret;
  }

  bool HasTensorShapeInfo() const noexcept {
    return impl_->HasTensorShapeInfo();
  }

  MLOpKernelTensorShapeInfo GetTensorShapeInfo() const {
    const IMLOpKernelTensorShapeInfo* ret = nullptr;
    ML_CHECK_STATUS(impl_->GetTensorShapeInfo(&ret));
    return {ret};
  }

 private:
  const IMLOpKernelInfo* impl_;
};

class MLShapeInferenceContext : public MLOperatorAttributes {
 public:
  MLShapeInferenceContext(IMLShapeInferenceContext* impl) : MLOperatorAttributes(impl), impl_(impl) {}

  // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
  const IMLShapeInferenceContext* GetInterface() const noexcept {
    return impl_;
  }

  uint32_t GetInputCount() const noexcept {
    return impl_->GetInputCount();
  }

  uint32_t GetOutputCount() const noexcept {
    return impl_->GetOutputCount();
  }

  MLEdgeType GetInputEdgeType(uint32_t input_index) const {
    MLEdgeType ret;
    ML_CHECK_STATUS(impl_->GetInputEdgeType(input_index, &ret));

    return ret;
  }

  uint32_t GetInputTensorDimensionCount(uint32_t input_index) const {
    uint32_t ret;
    ML_CHECK_STATUS(impl_->GetInputTensorDimensionCount(input_index, &ret));
    return ret;
  }

  std::vector<int64_t> GetInputTensorShape(uint32_t input_index) const {
    std::vector<int64_t> ret;
    uint32_t dimension_count = GetInputTensorDimensionCount(input_index);
    ret.resize(dimension_count);

    ML_CHECK_STATUS(impl_->GetInputTensorShape(input_index, dimension_count, ret.data()));
    return ret;
  }

  void SetOutputTensorShape(uint32_t output_index, const std::vector<int64_t>& output_dimensions) {
    ML_CHECK_STATUS(impl_->SetOutputTensorShape(output_index, static_cast<uint32_t>(output_dimensions.size()), output_dimensions.data()));
  }

 private:
  IMLShapeInferenceContext* impl_;
};

class MLTypeInferenceContext : public MLOperatorAttributes {
 public:
  MLTypeInferenceContext(IMLTypeInferenceContext* impl) : MLOperatorAttributes(impl),impl_(impl) {}

  // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
  const IMLTypeInferenceContext* GetInterface() const noexcept {
    return impl_;
  }

  uint32_t GetInputCount() const noexcept {
    return impl_->GetInputCount();
  }

  uint32_t GetOutputCount() const noexcept {
    return impl_->GetOutputCount();
  }

  MLEdgeType GetInputEdgeType(uint32_t input_index) const {
    MLEdgeType type;
    ML_CHECK_STATUS(impl_->GetInputEdgeType(input_index, &type));

    return type;
  }

  void SetOutputEdgeType(uint32_t output_index, const MLEdgeType* edge_type) const {
    ML_CHECK_STATUS(impl_->SetOutputEdgeType(output_index, edge_type));
  }

 private:
  IMLTypeInferenceContext* impl_;
};

class MLOpTensor {
 public:
  MLOpTensor(IMLOpTensor* impl) : impl_(impl) {}

  // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
  const IMLOpTensor* GetInterface() const noexcept {
    return impl_;
  }

  IMLOpTensor* GetInterface() noexcept {
    return impl_;
  }

  // Need default constructor for usage in STL containers.
  MLOpTensor() = default;
  MLOpTensor(const MLOpTensor&) = default;
  MLOpTensor(MLOpTensor&&) = default;
  MLOpTensor& operator=(const MLOpTensor&) = default;
  // TODO rename to shape to match other methods
  uint32_t GetDimensionCount() const {
    return impl_->GetDimensionCount();
  }

  const std::vector<int64_t>& GetDimensions() const {
    if (dimensions_cache_.empty()) {
      uint32_t dimension_count = GetDimensionCount();
      const_cast<MLOpTensor*>(this)->dimensions_cache_.resize(dimension_count);
      ML_CHECK_STATUS(impl_->GetDimensions(const_cast<MLOpTensor*>(this)->dimensions_cache_.data(), dimension_count));
    }

    return dimensions_cache_;
  }

  MLTensorDataType GetTensorDataType() const noexcept {
    return impl_->GetTensorDataType();
  }

  bool IsCPUData() const noexcept {
    return impl_->IsCPUData();
  }

  bool IsDataHandle() const noexcept {
    return impl_->IsDataHandle();
  }

  // Return data as an explicitly typed array, verifying the requested type
  // is the actual data type in the tensor.
  template <typename T>
  T* GetData() {
    ML_CHECK_BOOL(GetTensorDataType() == MLTypeTraits<T>::TensorType);
    ML_CHECK_BOOL(!IsDataHandle());

    return static_cast<T*>(impl_->GetData());
  }

  template <typename T>
  const T* GetData() const {
    ML_CHECK_BOOL(GetTensorDataType() == MLTypeTraits<T>::TensorType);
    ML_CHECK_BOOL(!IsDataHandle());

    return static_cast<const T*>(impl_->GetData());
  }

  // Return as raw bytes, regardless of underlying type, which is useful when
  // needing to agnostically copy memory.
  const void* GetByteData() const {
    ML_CHECK_BOOL(!IsDataHandle());

    return impl_->GetData();
  }

  void* GetByteData() {
    ML_CHECK_BOOL(!IsDataHandle());

    return impl_->GetData();
  }

  void* GetDataHandle() {
    ML_CHECK_BOOL(IsDataHandle());

    return impl_->GetData();
  }

  const void* GetDataHandle() const {
    ML_CHECK_BOOL(IsDataHandle());

    return impl_->GetData();
  }

  bool IsUnused() const noexcept {
    return impl_->IsUnused();
  }

 private:
  IMLOpTensor* impl_;

  std::vector<int64_t> dimensions_cache_;
};

class MLTemporaryDataDeleter {
 public:
  MLTemporaryDataDeleter()  {}
  MLTemporaryDataDeleter(const MLOpKernelContext* context)
      : context_(context) {}

  void operator()(void* p) const;

 private:
  const MLOpKernelContext* context_{nullptr};
};

using MLTemporaryDataUniquePtr = std::unique_ptr<void, MLTemporaryDataDeleter>;

class MLOpKernelContext {
 public:
  MLOpKernelContext(IMLOpKernelContext* impl) : impl_(impl) {}

  // Retrieve the underlying ABI compatible interface from the wrapper, for cases of interop
  // between components or different DLLs where the caller needs to pass the unwrapped class
  // across a boundary. e.g. Operator implementations may use the helper classes so that
  // they can use exceptions without checking every return value, but then they need to pass
  // results onward to a different component which expects the lower level currency.
  IMLOpKernelContext* GetInterface() const noexcept {
    return impl_;
  }

  const MLOpTensor GetInputTensor(uint32_t input_index) const {
    const IMLOpTensor* tensor = nullptr;
    ML_CHECK_STATUS(impl_->GetInputTensor(input_index, &tensor));
    return const_cast<IMLOpTensor*>(tensor);
  }

  MLOpTensor GetOutputTensor(uint32_t output_index) const {
    IMLOpTensor* tensor = nullptr;
    ML_CHECK_STATUS(impl_->GetOutputTensor(output_index, &tensor));
    return const_cast<IMLOpTensor*>(tensor);
  }

  MLOpTensor GetOutputTensor(uint32_t output_index, const std::vector<int64_t> dimension_sizes) const {
    IMLOpTensor* tensor = nullptr;
    ML_CHECK_STATUS(impl_->GetOutputTensor(output_index, dimension_sizes.data(), static_cast<uint32_t>(dimension_sizes.size()), &tensor));
    return MLOpTensor(tensor);
  }

  MLTemporaryDataUniquePtr AllocateTemporaryData(uint64_t size) const {
    void* data = nullptr;
    ML_CHECK_STATUS(impl_->AllocateTemporaryData(size, &data));
    return MLTemporaryDataUniquePtr(data, this);
  }

  const void* GetExecutionHandle() const noexcept {
    return impl_->GetExecutionHandle();
  }

 private:
  IMLOpKernelContext* impl_ = nullptr;
};

inline void MLTemporaryDataDeleter::operator()(void* p) const {
  if (context_)
    context_->GetInterface()->FreeTemporaryData(p);
}

// Helper class for operator implementations, templatized by the
// implementation type. This class converts ABI types to wrappers,
// supports STL types, and converts exceptions to return values.
template <class T>
class MLOpKernel : public IMLOpKernel, public T {
 public:
  static ML_API_IMP(CreateInstance)(const IMLOpKernelInfo& info, IMLOpKernel** op_kernel) noexcept {
    try {
      *op_kernel = new MLOpKernel(MLOpKernelInfo(&info));
      return MLStatus::OK;
    } catch (const MLStatusException& ex) {
      return ex.GetStatus();
    } catch (const std::exception& /*ex*/) {
      return MLStatus::FAIL;
    }
  }

  MLOpKernel(const MLOpKernelInfo& info) : T(info) {
  }

  virtual ~MLOpKernel() {
  }

  ML_API_IMP_(void, Release)() noexcept override {
    delete this;
  }

  ML_API_IMP(Compute)(IMLOpKernelContext* context) noexcept override {
    try {
      T::Compute(MLOpKernelContext(context));

      return MLStatus::OK;
    } catch (const MLStatusException& ex) {
      return ex.GetStatus();
    } catch (const std::exception& /*ex*/) {
      return MLStatus::FAIL;
    }
  }

  using T::Compute;
};

} // namespace onnxruntime
