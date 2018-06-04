//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include "op_kernel_author.h"
#include <limits>
#include <string>
#include <vector>
#include <memory>

typedef const char* MLConstStringParam;

class MLOpKernelContext;

// TODO - Consider using this directly in Lotus and merging error handling
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

//
// Wrappers for ABI objects consumed by kernels.
// These wrappers provide typesafe methods which use STL types and convert
// return values to exceptions.
//

class MLOpKernelInfo {
 public:
  MLOpKernelInfo(const IMLOpKernelInfo* impl) : impl_(impl) {}

  // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
  const IMLOpKernelInfo* GetInterface() const noexcept {
    return impl_;
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

  const void* GetExecutionHandle() const noexcept {
    return impl_->GetExecutionHandle();
  }

 private:
  const IMLOpKernelInfo* impl_;
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

  uint32_t GetDimensionCount() const {
    uint32_t dimension_count = 0;

    ML_CHECK_STATUS(impl_->GetDimensionCount(&dimension_count));
    return dimension_count;
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

 private:
  IMLOpTensor* impl_;

  std::vector<int64_t> dimensions_cache_;
};

class MLTemporaryDataDeleter {
 public:
  MLTemporaryDataDeleter() : context_(nullptr) {}
  MLTemporaryDataDeleter(const MLOpKernelContext* context)
      : context_(context) {}

  void operator()(void* p) const;

 private:
  const MLOpKernelContext* context_;
};

typedef std::unique_ptr<void, MLTemporaryDataDeleter> MLTemporaryDataUniquePtr;

class MLOpKernelContext {
 public:
  MLOpKernelContext(IMLOpKernelContext* impl) : impl_(impl) {}

  // Retrieve the underlying ABI compatible interface from the wrapper, for cases of interop
  // between components or different DLLs where the caller needs to pass the unwrapped class
  // across a boundary. e.g. Operator implementations may use the helper classes so that
  // they can use exceptions without checking every return value, but then they need to pass
  // results onward to a different component which expects the lower level currency.
  IMLOpKernelContext* GetInterface() noexcept {
    return impl_;
  }

  const IMLOpKernelContext* GetInterface() const noexcept {
    return impl_;
  }

  MLEdgeType GetInputEdgeType(uint32_t input_index) const {
    MLEdgeType edge_type;
    ML_CHECK_STATUS(impl_->GetInputEdgeType(input_index, &edge_type));
    return edge_type;
  }

  MLEdgeType GetOutputEdgeType(uint32_t output_index) const {
    MLEdgeType edge_type;
    ML_CHECK_STATUS(impl_->GetInputEdgeType(output_index, &edge_type));
    return edge_type;
  }

  const MLOpTensor GetInputTensor(uint32_t input_index) const {
    ML_CHECK_BOOL(GetInputEdgeType(input_index) == MLEdgeType::kTensor);

    const IMLOpTensor* tensor = nullptr;
    ML_CHECK_STATUS(impl_->GetInputTensor(input_index, &tensor));
    return const_cast<IMLOpTensor*>(tensor);
  }

  MLOpTensor GetOutputTensor(uint32_t output_index, const std::vector<int64_t> dimension_sizes) const {
    ML_CHECK_BOOL(GetOutputEdgeType(output_index) == MLEdgeType::kTensor);

    IMLOpTensor* tensor = nullptr;
    ML_CHECK_STATUS(impl_->GetOutputTensor(output_index, dimension_sizes.data(), static_cast<uint32_t>(dimension_sizes.size()), &tensor));
    return MLOpTensor(tensor);
  }

  uint32_t GetInputCount() const noexcept {
    return impl_->GetInputCount();
  }

  uint32_t GetOutputCount() const noexcept {
    return impl_->GetOutputCount();
  }

  MLTemporaryDataUniquePtr AllocateTemporaryData(uint64_t size) const {
    void* data = nullptr;
    ML_CHECK_STATUS(impl_->AllocateTemporaryData(size, &data));
    return MLTemporaryDataUniquePtr(data, this);
  }

  const IMLOpKernelContext* GetImpl() const { return impl_; }

 private:
  IMLOpKernelContext* impl_;
};

inline void MLTemporaryDataDeleter::operator()(void* p) const {
  if (context_)
    context_->GetImpl()->FreeTemporaryData(p);
}

// Helper class for operator implementations, templatized by the
// implementation type. This class converts ABI types to wrappers,
// supports STL / GSL types, and converts exceptions to return values.
template <class T>
class MLOpKernel : public IMLOpKernel, public T {
 public:
  static ML_API_IMP(CreateInstance)(const IMLOpKernelInfo& info, IMLOpKernel** opKernel) noexcept {
    try {
      *opKernel = new MLOpKernel(MLOpKernelInfo(&info));
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

  ML_API_IMP_(void, Release)
  () noexcept override {
    delete this;
  }

  ML_API_IMP(Compute)
  (
      const IMLOpKernelInfo* info,
      IMLOpKernelContext* context) noexcept override {
    try {
      T::Compute(
          MLOpKernelInfo(info),
          MLOpKernelContext(context));

      return MLStatus::OK;
    } catch (const MLStatusException& ex) {
      return ex.GetStatus();
    } catch (const std::exception& /*ex*/) {
      return MLStatus::FAIL;
    }
  }

  using T::Compute;
};
