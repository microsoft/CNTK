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

    // quantized values are stored in groups of 'qbwords'     = unsigned ints 
    // (which happen to memory-align with 'float' as used in 'quantizedcolumn' structure)
    typedef unsigned int QBWord;                            // one word of storage containing multiple bits
    static const size_t qbwordbits = 8 * sizeof (QBWord);   // number of bits in a qbword (32)
        
    // options for handling the mean for 1-bit quantization
    
    // force 1-bit quant to threshold against 0 rather than the midpoint between lower and upper, but use 
    #define ZERO_THRESHOLD_FOR_1BIT 
    // in 1-bit quantization, compute the quantization range boundaries including the residual values
    //#define INCLUDE_RESIDUE_FOR_QUANTIZATION_RANGE 

    template<class ElemType>
    class ValueQuantizer
    {
    public:
        cudasharedcode ValueQuantizer(size_t ldNbits, ElemType lower, ElemType upper);
        cudasharedcode unsigned int Quantize(ElemType u) const;                  
        cudasharedcode ElemType Unquantize(unsigned int u) const;         

        cudasharedcode bool Quantize1(ElemType u) const;
        static cudasharedcode ElemType Unquantize1(bool u, ElemType val0, ElemType val1);

        //how many bits we are quanatizing to
        cudasharedcode size_t NBits() const
        {
            return Nbits;          
        }
        
        //max value of quantize value; 2^Nbits
        cudasharedcode size_t QuanRangeEnd() const
        {
            return rangeend;         
        } 
        
        static size_t ld(size_t v);
        
    protected:   
        cudasharedcode unsigned int Quantize32(ElemType u) const;

    protected:
        /*const*/ size_t ldNbits;   // must be power of two
        /*const*/ size_t Nbits;     // now we quantized to 4 bits i.e. [0, 16)
        /*const*/ unsigned int rangeend;
        
        // quantization range
        /*const*/ ElemType quantimin;
        /*const*/ ElemType quantimax;
        
    #ifndef ZERO_THRESHOLD_FOR_1BIT
        // quantization threshold for 1-bit case
        /*const*/ ElemType quantimid;              
    #endif
    
        // precomputed factor for quantizating
        /*const*/ ElemType qfactor;    
        
        // and for unquantizing
        /*const*/ ElemType ufactor;    
    };
}}}
#endif 