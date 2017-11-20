#include "tensorutils.h"

namespace ONNXIR
{
    using namespace Common;

    namespace Utils
    {
        bool TensorUtils::IsLittleEndianOrder()
        {
            int n = 1;
            return (*(char*)&n == 1);
        }

        Status TensorUtils::UnpackTensor(const ONNXIR::TensorProto& p_tensor, /*out*/ std::vector<std::string>* p_data)
        {
            if (ONNXIR::TensorProto_DataType_STRING != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            p_data->clear();
            for (auto& elem : p_tensor.string_data())
            {
                p_data->push_back(elem);
            }
            return Status::OK();
        }

        Status TensorUtils::UnpackTensor(const ONNXIR::TensorProto& p_tensor, /*out*/ std::vector<float>* p_data)
        {
            if (ONNXIR::TensorProto_DataType_FLOAT != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            p_data->clear();
            if (p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return Status::OK();
            }

            for (auto elem : p_tensor.float_data())
            {
                p_data->push_back(elem);
            }
            return Status::OK();
        }

        Status TensorUtils::UnpackTensor(const ONNXIR::TensorProto& p_tensor, /*out*/ std::vector<int32_t>* p_data)
        {
            if (ONNXIR::TensorProto_DataType_INT32 != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            p_data->clear();
            if (p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return Status::OK();
            }

            for (auto elem : p_tensor.int32_data())
            {
                p_data->push_back(elem);
            }
            return Status::OK();
        }

        Status TensorUtils::UnpackTensor(const ONNXIR::TensorProto& p_tensor, /*out*/ std::vector<bool>* p_data)
        {
            if (ONNXIR::TensorProto_DataType_BOOL != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            p_data->clear();
            if (p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return Status::OK();
            }

            for (auto elem : p_tensor.int32_data())
            {
                p_data->push_back(elem != 0);
            }
            return Status::OK();
        }

        Status TensorUtils::UnpackTensor(const ONNXIR::TensorProto& p_tensor, /*out*/ std::vector<int64_t>* p_data)
        {
            if (ONNXIR::TensorProto_DataType_INT64 != p_tensor.data_type()
                || nullptr == p_data)
            {
                return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT);
            }

            p_data->clear();
            if (p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return Status::OK();
            }

            for (auto elem : p_tensor.int64_data())
            {
                p_data->push_back(elem);
            }
            return Status::OK();
        }
    }
}
