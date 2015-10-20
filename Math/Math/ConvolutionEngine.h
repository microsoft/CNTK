//
// <copyright file="ConvolutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

// REVIEW alexeyk: this seems to be repeated all over the CNTKMathDll.
#ifdef    _WIN32
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#else    // no DLLs on Linux
#define    MATH_API 
#endif

#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // REVIEW alexeyk: this is a temp class until we have generic tensor suport in CNTK.
    class ConvolutionTensor4D
    {
    public:
        size_t w() const { return m_w; }
        size_t h() const { return m_h; }
        size_t c() const { return m_c; }
        size_t n() const { return m_n; }
        void setN(size_t n) { m_n = n; }
    public:
        ConvolutionTensor4D(size_t w = 1, size_t h = 1, size_t c = 1, size_t n = 1)
        {
            m_w = w;
            m_h = h;
            m_c = c;
            m_n = n;
        }
    public:
        virtual ~ConvolutionTensor4D() = default;
        // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
        //ConvolutionTensor4D(const ConvolutionTensor4D&) = delete;
        //ConvolutionTensor4D& operator=(const ConvolutionTensor4D&) = delete;
        // Have to implement move ctor explicitly as VS2013 does not support default move ctors.
        //ConvolutionTensor4D(const ConvolutionTensor4D&&);
        //ConvolutionTensor4D& operator=(ConvolutionTensor4D&&);

    private:
        size_t m_w;
        size_t m_h;
        size_t m_c;
        size_t m_n;
    };

    // ConvolutionOptions describes properties specific to convolution application.
    // It is not part of the tensor class to avoid confusion between tensor-specific and convolution-specific terms having the same names.
    // For example, the stride in ConvolutionOptions means step size between subsequent convolution applications and not stride in the tensor storage of a particular dimension.
    // The class might be extended with other specific properties, like LRN constants etc.
    class ConvolutionOptions
    {
    public:
        // Horizontal stride (in w-dimension).
        size_t wStride() const { return m_wStride; }
        // Vertical stride (in h-dimension).
        size_t hStride() const { return m_hStride; }
        bool padding() const { return m_padding; }
    public:
        ConvolutionOptions(const ConvolutionTensor4D& inT, const ConvolutionTensor4D& filterT, 
            size_t wStride = 1, size_t hStride = 1, bool padding = false)
        {
            UNUSED(inT);
            UNUSED(filterT);
            m_wStride = wStride;
            m_hStride = hStride;
            m_padding = padding;
        }

    public:
        virtual ~ConvolutionOptions() = default;
        // Deleting copy ctor/assignment as derived objects may contain non-copyable state.
        //ConvolutionOptions(const ConvolutionOptions&) = delete;
        //ConvolutionOptions& operator=(const ConvolutionOptions&) = delete;

    private:
        size_t m_wStride;
        size_t m_hStride;
        bool m_padding;
    };

    template<class ElemType>
    class MATH_API ConvolutionEngine
    {
    public:
        using Tensor4D = ConvolutionTensor4D;
        using Tensor4DPtr = std::unique_ptr<Tensor4D>;
        using Mat = Matrix<ElemType>;

        ConvolutionEngine() {}
        virtual ~ConvolutionEngine() = default;

        static std::unique_ptr<ConvolutionEngine<ElemType>> Create(DEVICEID_TYPE deviceId, size_t maxTempMemSizeInSamples);

    public:
        virtual void Forward(const Tensor4D& inT, const Mat& in, const Tensor4D& filterT, const Mat& filter, const ConvolutionOptions& convOpt,
            const Tensor4D& outT, Mat& out) = 0;
        //virtual void BackwardData() = 0;
        //virtual void BackwardFilter() = 0;

        virtual Tensor4DPtr CreateConvTensor(size_t w = 1, size_t h = 1, size_t c = 1, size_t n = 1) = 0;
        //virtual Tensor4DPtr CreateFilterTensor() = 0;
        //virtual Tensor4DPtr CreatePoolingTensor() = 0;
        //virtual Tensor4DPtr CreateLrnTensor() = 0;

    public:
        ConvolutionEngine(const ConvolutionEngine&) = delete;
        ConvolutionEngine& operator=(const ConvolutionEngine&) = delete;
        ConvolutionEngine(ConvolutionEngine&&) = delete;
        ConvolutionEngine& operator=(ConvolutionEngine&&) = delete;
    };
}}}
