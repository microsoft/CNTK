//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#pragma warning(disable : 4127) // conditional expression is constant

namespace Microsoft { namespace MSR { namespace CNTK {

// RawType - input type to the quantizer
// QuantizedType - output type of the quantizer
template <class RawType, class QuantizedType>
class IQuantizerBase 
{
public:
    virtual void Quantize(const RawType* input, QuantizedType* output, size_t arraySize) = 0;
    virtual void Dequantize(const QuantizedType* input, RawType* output, size_t arraySize) = 0;

protected:
    static int rangeMax;
};

template <class RawType, class QuantizedType>
class SymmetricQuantizer : public IQuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizer;
    RawType m_invQuantizer;
public:
    // elements - collection to be quantized
    // extraBits decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    // Higher extraBits will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    // For quantization with shorts, recommended value of extraBits is 1 or 2.
    SymmetricQuantizer(RawType* elements, size_t elementsSize, size_t extraBits)
    {
        if (elementsSize == 0)
        {
            LogicError("The sequence to be quantized is empty.");
        }
        RawType absMax = FindAbsMax(elements, elementsSize);
        SymmetricQuantizer(absMax, extraBits);
    }

    // absoluteMax - the range of the quantizer (normally represents maximum absolute value of the values in the collection to be quantized).
    // extraBits - see comment in another ctor
    SymmetricQuantizer(RawType absoluteMax, size_t extraBits)
    {
        RawType shiftedMax = absoluteMax * (1 << extraBits);
        if (shiftedMax == 0)
        {
            LogicError("The absolute max element in the sequence to be quantized is 0.");
        }
        m_quantizer = rangeMax / shiftedMax;
        m_invQuantizer = shiftedMax / rangeMax;
    }

    virtual void Quantize(const RawType* input, QuantizedType* output, size_t inputSize)
    {
        for (size_t i = 0; i < inputSize; i++)
        {
            output[i] = (QuantizedType) (input[i] * m_quantizer);
        }
    }

    virtual void Dequantize(const QuantizedType* input, RawType* output, size_t inputSize)
    {
        for (size_t i = 0; i < inputSize; i++)
        {
            output[i] = (RawType)(input[i] * m_invQuantizer);
        }
    }

private: 
    // Find absolute maximum value
    RawType FindAbsMax(RawType* elements, size_t elementsSize)
    {
        // in constructor we asserted that arraySize > 0
        RawType maxElem, minElem = elements[0];
        for (size_t i = 0; i < elementsSize; i++)
        {
            maxElem = std::max(maxElem, elements[i]);
            minElem = std::min(minElem, elements[i]);
        }

        return std::max(maxElem, std::abs(minElem));
    }
};

int IQuantizerBase<float, short>::rangeMax = SHRT_MAX;
int IQuantizerBase<double, short>::rangeMax = SHRT_MAX;

}}}