//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

using namespace CNTK;

class UserTimesFunction final : public Function
{
public:
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return AsComposite(MakeSharedObject<UserTimesFunction>(leftOperand, rightOperand, name));
    }

    UserTimesFunction(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, Dictionary(), name)
    {}

private:
    static void MatrixMultiply(const NDArrayViewPtr& leftMatrix, const NDArrayViewPtr& rightMatrix, NDArrayViewPtr& outputMatrix, bool transposeRight = false)
    {
        auto GetNumRowsAndCols = [](const NDShape& shape, bool transpose = false) {
            auto numRows = shape[0];
            auto numCols = shape[shape.Rank() - 1];
            if (transpose)
                std::swap(numRows, numCols);

            return std::make_pair(numRows, numCols);
        };

        size_t leftNumRows, leftNumCols;
        std::tie(leftNumRows, leftNumCols) = GetNumRowsAndCols(leftMatrix->Shape());

        size_t rightNumRows, rightNumCols;
        std::tie(rightNumRows, rightNumCols) = GetNumRowsAndCols(rightMatrix->Shape(), transposeRight);

        auto numOutRows = leftNumRows;
        auto K = leftNumCols;
        auto numOutCols = rightNumCols;

        assert(!leftMatrix->IsSparse() && !rightMatrix->IsSparse() && !outputMatrix->IsSparse());
        assert(K == rightNumRows);
        assert((outputMatrix->Shape()[0] == numOutRows) && (outputMatrix->Shape()[1] == numOutCols));
        outputMatrix->SetValue(0.0f);

        // The operands values are in column major layout
        auto Offset = [](size_t rowIdx, size_t colIdx, const NDShape& matrixShape, bool transpose = false) {
            if (transpose)
                std::swap(rowIdx, colIdx);

            return (colIdx * matrixShape[0]) + rowIdx;
        };

        auto leftBuffer = leftMatrix->DataBuffer<float>();
        auto rightBuffer = rightMatrix->DataBuffer<float>();
        auto outBuffer = outputMatrix->WritableDataBuffer<float>();
        for (size_t j = 0; j < numOutCols; ++j)
            for (size_t k = 0; k < K; ++k)
                for (size_t i = 0; i < numOutRows; ++i)
                    outBuffer[Offset(i, j, outputMatrix->Shape())] += leftBuffer[Offset(i, k, leftMatrix->Shape())] * rightBuffer[Offset(k, j, rightMatrix->Shape(), transposeRight)];
    }

    BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/) override
    {
        auto leftOperandData = inputValues[0]->Data();
        auto rightOperandData = inputValues[1]->Data();

        // Allocate outputValue if needed
        auto& outputValue = outputs[this->Output()];
        if (outputValue == nullptr)
        {
            auto numOutRows = leftOperandData->Shape()[0];
            auto numOutCols = rightOperandData->Shape()[rightOperandData->Shape().Rank() - 1];
            outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ numOutRows , numOutCols }), computeDevice));
        }

        auto outputData = outputValue->Data();
        MatrixMultiply(leftOperandData, rightOperandData, outputData);

        // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
        return MakeSharedObject<BackPropState>(this->shared_from_this(), computeDevice, std::unordered_map<Variable, ValuePtr>({ {Inputs()[1], inputValues[1] } }));
    }

    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        auto leftInputVariable = Inputs()[0];
        auto rightInputVariable = Inputs()[1];
        if (backPropagatedGradientValuesForInputs.find(rightInputVariable) != backPropagatedGradientValuesForInputs.end())
            std::runtime_error("UserTimesFunction does not support computing gradient wrt right operand");

        auto rightInputData = state->SavedForwardPropValues().at(rightInputVariable)->Data();

        // Allocate input gradient Value if needed
        auto& inputGradientValue = backPropagatedGradientValuesForInputs[leftInputVariable];
        if (inputGradientValue == nullptr)
            inputGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, leftInputVariable.Shape(), state->Device()));

        auto rootGradientData = rootGradientValues.at(this->Output())->Data();
        auto inputGradientData = inputGradientValue->Data();

        MatrixMultiply(rootGradientData, rightInputData, inputGradientData, /*transposeRight =*/ true);
    }

    const std::wstring& OpName() const override
    {
        static const std::wstring opName = L"UserTimesOp";
        return opName;
    }

    Dictionary Serialize() const override { NOT_IMPLEMENTED; }
    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }

    void InferOutputs(std::vector<Variable>& outputs) override
    {
        auto leftOperand = Inputs()[0];
        auto rightOperand = Inputs()[1];

        if (leftOperand.Shape().Rank() != 2)
            std::runtime_error("Left operand must be 2D");

        if (rightOperand.Shape().Rank() != 1)
            std::runtime_error("Right operand must be 1D");

        if (!leftOperand.DynamicAxes().empty())
            std::runtime_error("Left operand must not have dynamic axes (i.e. should not be minibatch data, but be a Parameter of fixed size)");

        outputs.push_back(OutputVariable(NDShape({ leftOperand.Shape()[0] }), leftOperand.GetDataType(), rightOperand.DynamicAxes()));
    }
};
