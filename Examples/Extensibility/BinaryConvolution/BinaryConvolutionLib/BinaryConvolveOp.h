//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This file contains an implementation of single bit binarization using an optimized halide function call

#include "CNTKLibrary.h"
#include "convolve_wrapper.h"

using namespace CNTK;

class BinaryConvolveFunction final : public Function
{
public:
    // initialization function, creates an object for the user function 
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const Dictionary& attributes, const std::wstring& name)
    {   
        return AsComposite(MakeSharedObject<BinaryConvolveFunction>(leftOperand, rightOperand, attributes, name));
    }   

    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {   
        return Create(leftOperand, rightOperand, Dictionary(), name);
    } 

    // declares our function as a subset of the Function class and maps the dictionary values in
    BinaryConvolveFunction(const Variable& leftOperand, const Variable& rightOperand, const Dictionary& attributes, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, Dictionary(attributes), name), Attr(Dictionary(attributes))
    {} 

private:
    // simple convolve function that pulls out raw data buffers and passes them into our halide function
    static void Convolve(const NDArrayViewPtr& weights, const NDArrayViewPtr& input, const int size, const int stride, const int pad, const int w, const int h, const int channels, const int num_filters, NDArrayViewPtr& output)
    {
        auto weightBuffer = weights->DataBuffer<float>();
        auto inputBuffer = input->DataBuffer<float>();
        auto outBuffer = output->WritableDataBuffer<float>();
        invoke_halide_convolve(weightBuffer, inputBuffer, num_filters, size, channels, pad, stride, w, h, outBuffer); 
    }

    // forward function definition, needs to parse the data and call into the Convolve function
    BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/) override
    {
        // pull out the kernel data from inputValues
        auto leftOperandData = inputValues[0]->Data();
        // pull out the activation data from inputValues
        auto rightOperandData = inputValues[1]->Data();
        // determine the number of filters in the input
        auto kernelRank = leftOperandData->Shape().Rank();
        long unsigned int num_filters;
        if (kernelRank >= 4) {
            num_filters = leftOperandData->Shape()[3];
        } else {
            num_filters = 1; 
        }
        // extract some basic information that is needed by halide
        auto channels = leftOperandData->Shape()[2];
        auto w = rightOperandData->Shape()[0];
        auto h = rightOperandData->Shape()[1];

        auto pad = Attr[padkey].Value<bool>();
        auto size = Attr[sizekey].Value<int>();
        auto stride = Attr[stridekey].Value<int>();

        // Allocate outputValue if needed
        auto& outputValue = outputs[this->Output()];
        if (outputValue == nullptr)
        {
            auto numOutCols = pad == 0 ? (w - size)/stride + 1 : (w - 1)/stride + 1;
            auto numOutRows = pad == 0 ? (h - size)/stride + 1 : (h - 1)/stride + 1;
            outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ numOutRows , numOutCols, num_filters }), computeDevice));
        }
        
        // extract the output data
        auto outputData = outputValue->Data();
        // pass everything to Halide to compute the result, outputs are directly stored in the outputData buffer
        Convolve(leftOperandData, rightOperandData, size, stride, pad, w, h, channels, num_filters, outputData);

        // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
        return MakeSharedObject<BackPropState>(this->shared_from_this(), computeDevice, std::unordered_map<Variable, ValuePtr>({ {Inputs()[1], inputValues[1] } }));
    }

    // backprop currently not implemented, simply throw an error
    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        std::runtime_error("Binary Convolution does not currently support backprop");
    }

    const std::wstring& OpName() const override
    {
        static const std::wstring opName = L"BinaryConvolveOp";
        return opName;
    }

    Dictionary Serialize() const override { NOT_IMPLEMENTED; }
    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }
    // create a dictionary of attributes with a few specific keys
    const Dictionary Attr;
    const wchar_t* padkey = L"padding";
    const wchar_t* stridekey = L"stride";
    const wchar_t* sizekey = L"size";

    // Compute the dimensions of the output variable and return the proper shape and dynamic axes
    void InferOutputs(std::vector<Variable>& outputs) override
    {
        // Pull out the inputs to the function, left is kernels right is activations
        auto leftOperand = Inputs()[0];
        auto rightOperand = Inputs()[1];

        auto kernelRank = leftOperand.Shape().Rank();
        long unsigned int num_filters;
        // determine the number of filters 
        if (kernelRank >= 4) {
            num_filters = leftOperand.Shape()[3];
        } else {
            num_filters = 1; 
        }
        auto w = rightOperand.Shape()[0];
        auto h = rightOperand.Shape()[1];

        auto pad = Attr[padkey].Value<bool>();
        auto size = Attr[sizekey].Value<int>();
        auto stride = Attr[stridekey].Value<int>();

        // compute the output dimensions
        auto numOutCols = !pad ? (w - size)/stride + 1 : (w - 1)/stride + 1;
        auto numOutRows = !pad ? (h - size)/stride + 1 : (h - 1)/stride + 1;
        // return the appropriate output shape 
        outputs.push_back(OutputVariable(NDShape({ numOutRows, numOutCols, num_filters }), leftOperand.GetDataType(), rightOperand.DynamicAxes()));
    }

    FunctionPtr Clone(const std::vector<Variable>& clonedInputs) override
    {
        return Create(clonedInputs[0], clonedInputs[1], this->Attributes(), this->Name());
    }
};
