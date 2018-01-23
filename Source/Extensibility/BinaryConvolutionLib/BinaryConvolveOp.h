//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// This file contains an implementation of single bit binarization using an optimized halide function call

#include "halide_binary_convolve.h"
#include "CNTKLibrary.h"

using namespace CNTK;

int convolutional_out_size(int x, int size, int stride, bool pad)
{
    if (!pad) x -= size;
    else x -= 1;
    return x/stride + 1;
}

void binarize_array(const float *input, int size, int64_t *binary)
{
    for (int i = 0; i < size; ++i) {
        int index = i;
        int block = index/64;
        int bit = index%64;
        float input_val = input[index];
        if (input_val > 0) {
            binary[block] |= ((uint64_t) 1 << bit);
        } else {
            binary[block] &= ~((uint64_t) 1 << bit);
        }
    }
}

float pad_mask_check_pixel(int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return false;
    return true;
}

void get_pad_mask(int channels,  int height,  int width,
     int ksize,  int stride, int pad, int64_t* pad_mask)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int filter_size = ksize*ksize*channels;
    int bit;
    int block;
    // pad just indicates that you want your windows to fit in nicely, add however many 0s as is needed (ksize/2) to make that happen,
    // means pad should either be 1 or 0 in cfg file
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int output_size = height_col * width_col;
    for (c = 0; c < output_size; ++c) {
        int block_start = c * ((filter_size - 1)/64 + 1);
        int w_offset = (c*stride) % width_col;
        int h_offset = ((c*stride) / width_col) % height_col;
        for (h = 0; h < channels; ++h) {
            for (w = 0; w < (ksize*ksize); ++w) {
                int im_row = h_offset + (w / ksize);
                int im_col = w_offset + (w % ksize);
                int col_offset = (h * ksize*ksize) + w;
                // note that data col is an array of uint64 values, find which uint64 has the bit we want to set
                block = block_start + (col_offset/64);
                // now find the bit in that block that needs to be set
                bit = col_offset % 64;
                // finally, set or clear that bit
                if (pad_mask_check_pixel(height, width, channels, im_row, im_col, h, pad)) {
                    pad_mask[block] |= ((uint64_t) 1 << bit);
                } else {
                    pad_mask[block] &= ~((uint64_t) 1 << bit);
                }
            }
        }
    }
}

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
    {
        w = Attr[w_key].Value<int>();
        h = Attr[h_key].Value<int>();
        size = Attr[size_key].Value<int>();
        stride = Attr[stride_key].Value<int>();
        pad = Attr[pad_key].Value<bool>();
        channels = Attr[channels_key].Value<int>();
        filters = Attr[filters_key].Value<int>(); 
        out_h = convolutional_out_size(h, size, stride, pad);
        out_w = convolutional_out_size(w, size, stride, pad);
        const NDArrayViewPtr& weight_array = leftOperand.GetValue();
        weight_data = weight_array->DataBuffer<float>();
        binary_weights = (int64_t *) malloc(((size*size*channels)/64)*filters*sizeof(int64_t));
        pad_mask = (int64_t *) malloc((size*size*channels/64)*out_h*out_w*sizeof(int64_t));
        binarize_array(weight_data, size*size*channels*filters, binary_weights);
        Executor = new HalideBinaryConvolve(binary_weights, pad_mask, w, h, channels, filters, size, stride, pad);
    } 

private:
    // simple convolve function that pulls out raw data buffers and passes them into our halide function
    void Convolve(const NDArrayViewPtr& input, NDArrayViewPtr& output)
    {
        auto inputBuffer = input->DataBuffer<float>();
        auto outBuffer = output->WritableDataBuffer<float>();
        Executor->realize(inputBuffer, outBuffer);
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

        // Allocate outputValue if needed
        auto& outputValue = outputs[this->Output()];
        if (outputValue == nullptr)
        {
            auto numOutCols = !pad ? (w - size)/stride + 1 : (w - 1)/stride + 1;
            auto numOutRows = !pad ? (h - size)/stride + 1 : (h - 1)/stride + 1;
            outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ (long unsigned int) numOutRows, (long unsigned int) numOutCols, (long unsigned int) filters }), computeDevice));
        }
        
        // extract the output data
        auto outputData = outputValue->Data();
        // pass everything to Halide to compute the result, outputs are directly stored in the outputData buffer
        Convolve(rightOperandData, outputData);

        // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
        return MakeSharedObject<BackPropState>(this->shared_from_this(), computeDevice, std::unordered_map<Variable, ValuePtr>({ {Inputs()[1], inputValues[1] } }));
    }

    // backprop currently not implemented, simply throw an error
    void Backward(const BackPropStatePtr& state,
                  const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                  std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override
    {
        state; rootGradientValues; backPropagatedGradientValuesForInputs; 
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
    const wchar_t* pad_key = L"padding";
    const wchar_t* stride_key = L"stride";
    const wchar_t* size_key = L"size";
    const wchar_t* w_key = L"w";
    const wchar_t* h_key = L"h";
    const wchar_t* channels_key = L"channels";
    const wchar_t* filters_key = L"filters";
    bool pad;
    int stride;
    int size;
    int w;
    int h;
    int channels;
    int filters;
    int out_w;
    int out_h;
    int64_t *binary_weights;
    int64_t *pad_mask;
    const float *weight_data;
    HalideBinaryConvolve *Executor;

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
            num_filters = (long unsigned int)leftOperand.Shape()[3];
        } else {
            num_filters = 1; 
        }
        auto w = rightOperand.Shape()[0];
        auto h = rightOperand.Shape()[1];

        auto pad = Attr[pad_key].Value<bool>();
        auto size = Attr[size_key].Value<int>();
        auto stride = Attr[stride_key].Value<int>();

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
