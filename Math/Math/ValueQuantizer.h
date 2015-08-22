#pragma once 
#ifndef __VALLUE_QUANTIZER_H__
#define __VALLUE_QUANTIZER_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace Microsoft { namespace MSR { namespace CNTK {
    
    #ifdef __device__  // this can be used in CUDA; if this is not defined, then we are compiling in a non-CUDA context
    #define cudacode       __device__           // CUDA: we assume we ONLY run these functions on CUDA (otherwise we'd need to mess with specifiers of matrixref)
    #define cudasharedcode __device__ __host__  // shared on both CUDA and CPU; note that such functions cannot call into __device__ only functions like matrixref::operator(,)
    #undef assert
    #define assert(c)
    #else
    #define cudacode  // non-CUDA context: defines to nothing
    #define cudasharedcode
    //#define QUANTUSEPPL
    #endif

    #ifdef QUANTUSEPPL
    #include <ppl.h>    // in non-CUDA: also use PPL lib
    #endif

    template <typename ElemType> 
    class QuantizedWordHelper;

    template<>
    class QuantizedWordHelper<float>
    {
    public:
        typedef unsigned int ValueType;
        typedef int ValueTypeSigned;
        static_assert(sizeof(float) == sizeof(ValueType), "Quantized word size != size of ElemType=float");
    };

    template<>
    class QuantizedWordHelper<double>
    {
    public:
        typedef unsigned long long ValueType;
        typedef long long ValueTypeSigned;
        static_assert(sizeof(double) == sizeof(ValueType), "Quantized word size != size of ElemType=double");
    };

    template<class ElemType>
    class ValueQuantizer
    {
    public:
        typedef typename QuantizedWordHelper<ElemType>::ValueType QWord;
        typedef typename QuantizedWordHelper<ElemType>::ValueType QWordVal;
        typedef typename QuantizedWordHelper<ElemType>::ValueTypeSigned QWordValSigned;
        static const size_t QWordNumBits = 8 * sizeof(QWord);

    public:
        cudasharedcode ValueQuantizer(size_t ldNbits, ElemType lower, ElemType upper);

        template<bool ZeroThresholdFor1Bit>
        cudasharedcode QWordVal Quantize(ElemType u) const;

        cudasharedcode ElemType Unquantize(QWordVal u) const;

        template<bool ZeroThresholdFor1Bit>
        cudasharedcode bool Quantize1(ElemType u) const;

        static cudasharedcode ElemType Unquantize1(bool u, ElemType val0, ElemType val1);

        //how many bits we are quanatizing to
        cudasharedcode size_t NBits() const
        {
            return Nbits;          
        }
        
        //max value of quantize value; 2^Nbits
        cudasharedcode QWordVal QuanRangeEnd() const
        {
            return rangeend;
        } 
        
        static size_t ld(size_t v);
        
    protected:   
        cudasharedcode QWordVal QuantizeToFullQWord(ElemType u) const;

    protected:
        // NBits must be power of two
        size_t ldNbits;
        size_t Nbits;

        QWordVal rangeend;
        
        // quantization range
        ElemType quantimin;
        ElemType quantimax;
        
        // quantization threshold for 1-bit case
        ElemType quantimid;
    
        // precomputed factor for quantizing
        ElemType qfactor;
        
        // and for unquantizing
        ElemType ufactor;
    };
}}}
#endif 