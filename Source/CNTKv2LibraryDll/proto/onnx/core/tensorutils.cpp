#include "tensorutils.h"

namespace ONNXIR
{
    using namespace Common;

    namespace Utils
    {
        Status TensorUtils::UnpackTensor(const onnx::TensorProto& p_tensor, /*out*/std::string* p_data, int64_t p_expected_size)
        {
            if (onnx::TensorProto_DataType_STRING != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            if (p_tensor.string_data_size() != p_expected_size)
                return Status(StatusCategory::ONNX, StatusCode::FAIL, \
                    "UnpackTensor: the pre-allocate size does not match the size in proto");

            for (auto& elem : p_tensor.string_data())
            {
                *p_data++ = elem;
            }
            return Status::OK();
        }

        Status TensorUtils::UnpackTensor(const onnx::TensorProto& p_tensor, /*out*/bool* p_data, int64_t p_expected_size)
        {
            if (onnx::TensorProto_DataType_BOOL != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            if (p_tensor.has_raw_data())
            {
                if (p_tensor.raw_data().size() != (p_expected_size) * sizeof(bool))
                    return Common::Status(Common::StatusCategory::ONNX, Common::StatusCode::FAIL,
                        "UnpackTensor: the pre-allocate size does not match the raw data size");
                UnpackTensorWithRawData(p_tensor, p_data);
                return Common::Status::OK();
            }

            if (p_tensor.int32_data_size() != p_expected_size)
                return Status(StatusCategory::ONNX, StatusCode::FAIL, \
                    "UnpackTensor: the pre-allocate size does not match the size in proto");

            for (auto& elem : p_tensor.int32_data())
            {
                *p_data++ = elem != 0;
            }
            return Status::OK();
        }
    }
}