#include "core/graph/tensorutils.h"

#include <algorithm>
#include "gsl/span"

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

  const auto data = gsl::make_span(p_data, expected_size);

  auto& string_data = tensor.string_data();
  std::copy(string_data.cbegin(), string_data.cend(), data.begin());

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

  const auto data = gsl::make_span(p_data, expected_size);
  std::copy(tensor.int32_data().cbegin(), tensor.int32_data().cend(), data.begin());

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

  const auto data = gsl::make_span(p_data, expected_size);
  for (int i = 0; i < expected_size; i++)
    data[i] = gsl::narrow_cast<uint16_t>(tensor.int32_data()[i]);

  return Status::OK();
}
}  // namespace Utils
}  // namespace Lotus
