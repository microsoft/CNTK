#include "proto/onnx/core/common/CommonSTD.h"
#include "proto/onnx/core/graph/tensorutils.h"

#include <algorithm>
// #include "gsl/span"

namespace ONNX
{
namespace Utils
{
Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor,
                                 /*out*/ std::string* p_data,
                                 int64_t expected_size)
{
    if (onnx::TensorProto_DataType_STRING != tensor.data_type() || nullptr == p_data)
    {
        return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
    }

    if (tensor.string_data_size() != expected_size)
        return Status(StatusCategory::ONNX, StatusCode::FAIL,
                      "UnpackTensor: the pre-allocate size does not match the size in proto");

    for (auto& elem : tensor.string_data()) {
        *p_data++ = elem;
    }

    return Status::OK();
}

Status TensorUtils::UnpackTensor(const onnx::TensorProto& tensor,
                                 /*out*/ bool* p_data,
                                 int64_t expected_size)
{
    if (onnx::TensorProto_DataType_BOOL != tensor.data_type() || nullptr == p_data)
    {
        return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
    }

    if (tensor.has_raw_data())
    {
        if (tensor.raw_data().size() != (expected_size) * sizeof(bool))
            return Status(StatusCategory::ONNX, StatusCode::FAIL,
                          "UnpackTensor: the pre-allocate size does not match the raw data size");

        UnpackTensorWithRawData(tensor, p_data);
        return Status::OK();
    }

    if (tensor.int32_data_size() != expected_size)
        return Status(StatusCategory::ONNX, StatusCode::FAIL,
                      "UnpackTensor: the pre-allocate size does not match the size in proto");

    for (auto& elem : tensor.int32_data()) {
        *p_data++ = elem != 0;
    }

    return Status::OK();
}
} // namespace Utils
} // namespace ONNX
