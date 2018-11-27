// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
//std::copy only works for the same type(input/output must have the same type)
//TODO(@chasun): remove std::copy from DEFINE_UNPACK_TENSOR
#pragma warning(disable : 4244)
#endif
#include "core/framework/tensorutils.h"
#include "core/framework/allocator.h"

#include <algorithm>

#include "core/graph/onnx_protobuf.h"

#include "gsl/pointers"
#include "gsl/span"

#include "core/inc/op_kernel_author.h"

GSL_SUPPRESS(type .1)  // allow use of reinterpret_cast for this special case
inline bool IsLittleEndianOrder() noexcept {
  static int n = 1;
  return (*reinterpret_cast<char*>(&n) == 1);
}

template <typename T>
static void UnpackTensorWithRawData(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ T* p_data) {
  // allow this low level routine to be somewhat unsafe. assuming it's thoroughly tested and valid
  GSL_SUPPRESS(type)       // type.1 reinterpret-cast; type.4 C-style casts; type.5 'T result;' is uninitialized;
  GSL_SUPPRESS(bounds .1)  // pointer arithmetic
  GSL_SUPPRESS(f .23)      // buff and temp_bytes never tested for nullness and could be gsl::not_null
  {
    auto& raw_data = tensor.raw_data();
    auto buff = raw_data.c_str();
    const size_t type_size = sizeof(T);

    if (IsLittleEndianOrder()) {
      memcpy((void*)p_data, (void*)buff, raw_data.size() * sizeof(char));
    } else {
      for (size_t i = 0; i < raw_data.size(); i += type_size, buff += type_size) {
        T result;
        const char* temp_bytes = reinterpret_cast<char*>(&result);
        for (size_t j = 0; j < type_size; ++j) {
          memcpy((void*)&temp_bytes[j], (void*)&buff[type_size - 1 - i], sizeof(char));
        }
        p_data[i] = result;
      }
    }
  }
}

namespace onnxruntime {
namespace utils {
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                                             \
  template <>                                                                                                             \
  Status TensorUtils::UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ T* p_data, int64_t expected_size) { \
    if (nullptr == p_data) {                                                                                              \
      const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.field_size();                         \
      if (size == 0)                                                                                                      \
        return Status::OK();                                                                                              \
      else                                                                                                                \
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                     \
    }                                                                                                                     \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                                \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                       \
    }                                                                                                                     \
    if (tensor.has_raw_data()) {                                                                                          \
      if (tensor.raw_data().size() != ((expected_size) * sizeof(T)))                                                      \
        return Status(common::ONNXRUNTIME, common::FAIL,                                                                  \
                      "UnpackTensor: the pre-allocated size does not match the raw data size");                           \
      UnpackTensorWithRawData(tensor, p_data);                                                                            \
      return Status::OK();                                                                                                \
    }                                                                                                                     \
    if (tensor.field_size() != expected_size)                                                                             \
      return Status(common::ONNXRUNTIME, common::FAIL,                                                                    \
                    "UnpackTensor: the pre-allocated size does not match the size in proto");                             \
    const auto span = gsl::make_span(p_data, expected_size);                                                              \
    auto& data = tensor.field_name();                                                                                     \
    std::copy(data.cbegin(), data.cend(), span.begin());                                                                  \
    return Status::OK();                                                                                                  \
  }

//TODO: uint32 uint64 complex64 complex128
//TODO: int16_t/uint16_t/float16 is confusing right now
DEFINE_UNPACK_TENSOR(float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR(double, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR(uint8_t, ONNX_NAMESPACE::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int16_t, ONNX_NAMESPACE::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(uint16_t, ONNX_NAMESPACE::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR(uint64_t, ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR(uint32_t, ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

template <>
Status TensorUtils::UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                                 /*out*/ std::string* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0)
      return Status::OK();
    else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_STRING != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.string_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const auto data = gsl::make_span(p_data, expected_size);

  auto& string_data = tensor.string_data();
  std::copy(string_data.cbegin(), string_data.cend(), data.begin());

  return Status::OK();
}
template <>
Status TensorUtils::UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                                 /*out*/ bool* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(bool))
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor, p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const auto data = gsl::make_span(p_data, expected_size);
  std::copy(tensor.int32_data().cbegin(), tensor.int32_data().cend(), data.begin());

  return Status::OK();
}
template <>
Status TensorUtils::UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                                 /*out*/ MLFloat16* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(uint16_t))
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor, p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  const auto data = gsl::make_span(p_data, expected_size);
  for (int i = 0; i < expected_size; i++)
    data[i] = MLFloat16(gsl::narrow<uint16_t>(tensor.int32_data()[i]));

  return Status::OK();
}

#define CASE_PROTO_TRACE(X, Y)                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                    \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) { \
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");    \
    }                                                                                     \
    break;

template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");
    }
    if (!IAllocator::CalcMemSizeForArray(size, static_cast<size_t>(dims[i]), &size)) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Invalid TensorProto");
    }
  }
  switch (tensor_proto.data_type()) {
    CASE_PROTO_TRACE(FLOAT, float);
    CASE_PROTO_TRACE(DOUBLE, double);
    CASE_PROTO_TRACE(BOOL, bool);
    CASE_PROTO_TRACE(INT8, int8_t);
    CASE_PROTO_TRACE(INT16, int16_t);
    CASE_PROTO_TRACE(INT32, int32_t);
    CASE_PROTO_TRACE(INT64, int64_t);
    CASE_PROTO_TRACE(UINT8, uint8_t);
    CASE_PROTO_TRACE(UINT16, uint16_t);
    CASE_PROTO_TRACE(UINT32, uint32_t);
    CASE_PROTO_TRACE(UINT64, uint64_t);
    CASE_PROTO_TRACE(FLOAT16, MLFloat16);
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING:
    default:
      return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

template common::Status GetSizeInBytesFromTensorProto<256>(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);
}  // namespace utils
}  // namespace onnxruntime
