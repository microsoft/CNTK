#include "core/common/CommonSTD.h"
#include "core/graph/tensorutils.h"

#include <algorithm>
// #include "gsl/span"

namespace Lotus {
namespace Utils {
Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor,
                                 /*out*/ std::string* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0)
      return Status::OK();
    else
      return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_STRING != tensor.data_type()) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }

  if (tensor.string_data_size() != expected_size)
    return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  for (auto& elem : tensor.string_data()) {
    *p_data++ = elem;
  }

  return Status::OK();
}

Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor,
                                 /*out*/ bool* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    else
      return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(bool))
      return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor, p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  for (auto& elem : tensor.int32_data()) {
    *p_data++ = elem;
  }

  return Status::OK();
}

Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor,
                                 /*out*/ uint16_t* p_data,
                                 int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = tensor.has_raw_data() ? tensor.raw_data().size() : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();
    else
      return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    return Status(StatusCategory::LOTUS, StatusCode::INVALID_ARGUMENT);
  }

  if (tensor.has_raw_data()) {
    if (tensor.raw_data().size() != (expected_size) * sizeof(uint16_t))
      return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                    "UnpackTensor: the pre-allocate size does not match the raw data size");

    UnpackTensorWithRawData(tensor, p_data);
    return Status::OK();
  }

  if (tensor.int32_data_size() != expected_size)
    return Status(StatusCategory::LOTUS, StatusCode::FAIL,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");
  for (auto& elem : tensor.int32_data()) {
    *p_data++ = (uint16_t)elem;
  }

  return Status::OK();
}
}  // namespace Utils
}  // namespace Lotus
