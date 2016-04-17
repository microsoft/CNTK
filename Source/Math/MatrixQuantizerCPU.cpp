#include "stdafx.h"
#include "MatrixQuantizerCPU.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
MatrixQuantizerCPU<ElemType>::MatrixQuantizerCPU()
    : MatrixQuantizerImpl<ElemType>(CPUDEVICE)
{
}

template <class ElemType>
void MatrixQuantizerCPU<ElemType>::QuantizeAsync(const Matrix<ElemType>& inMatrix, const Matrix<ElemType>& inResidual, QuantizedMatrix<ElemType>& outQMatrix, Matrix<ElemType>& outResidual, bool zeroThresholdFor1Bit)
{
    // The outQMatrix should be on the CPU
    // TODO: Support transferring the quantization output to a quantized matrix on the GPU
    assert(outQMatrix.GetDeviceId() == CPUDEVICE);

    size_t nBits = outQMatrix.GetNumBits();

    size_t nRow = inMatrix.GetNumRows();
    size_t nCol = inMatrix.GetNumCols();

    // Verify that the different matrix parameters have matching dimensions
    assert((outQMatrix.GetNumRows() == nRow) && (outQMatrix.GetNumCols() == nCol));
    assert((inResidual.GetNumRows() == nRow) && (inResidual.GetNumCols() == nCol));
    assert((outResidual.GetNumRows() == nRow) && (outResidual.GetNumCols() == nCol));

    const size_t ldNbits = ValueQuantizer<ElemType>::ld(nBits);
#ifdef QUANTUSEPPL
    Concurrency::parallel_for((size_t) 0, us.cols(), [&](size_t j)
#else
    for (size_t j = 0; j < nCol; j++)
#endif
                              {
                                  auto& qcol = *(outQMatrix.GetQuantizedColumn(j));
                                  if (zeroThresholdFor1Bit)
                                  {
                                      // Explicit use of 'template' keyword is needed to compile with GCC
                                      ColumnQuantizer<ElemType>::template ComputeRangeStatColj<true>(inMatrix.Data(), inResidual.Data(), (long) nRow, j, nBits, qcol.lower, qcol.upper);
                                  }
                                  else
                                  {
                                      // Explicit use of 'template' keyword is needed to compile with GCC
                                      ColumnQuantizer<ElemType>::template ComputeRangeStatColj<false>(inMatrix.Data(), inResidual.Data(), (long) nRow, j, nBits, qcol.lower, qcol.upper);
                                  }

                                  ColumnQuantizer<ElemType> q(ldNbits, qcol.lower, qcol.upper);
                                  if (zeroThresholdFor1Bit)
                                  {
                                      // Explicit use of 'template' keyword is needed to compile with GCC
                                      q.template Quantize<true>(inMatrix.Data(), inResidual.Data(), (long) nRow, j, qcol.bits, outResidual.Data());
                                  }
                                  else
                                  {
                                      // Explicit use of 'template' keyword is needed to compile with GCC
                                      q.template Quantize<false>(inMatrix.Data(), inResidual.Data(), (long) nRow, j, qcol.bits, outResidual.Data());
                                  }
                              }
#ifdef QUANTUSEPPL
                              );
#endif
}

template <class ElemType>
void MatrixQuantizerCPU<ElemType>::WaitQuantizeAsyncDone()
{
    // TODO: Currently this is a no-op since the actual quantization is synchronous
}

// unquantize an entire matrix, calling unquantize() for each column
template <class ElemType>
void MatrixQuantizerCPU<ElemType>::UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add /*= false*/)
{
    // The inQMatrix and hould be on the CPU
    assert(inQMatrix.GetDeviceId() == CPUDEVICE);
    assert(outMatrix.GetDeviceId() == CPUDEVICE);

    size_t nBits = inQMatrix.GetNumBits();
    size_t nRow = inQMatrix.GetNumRows();
    size_t nCol = inQMatrix.GetNumCols();

    // Verify that the different matrix parameters have matching dimensions
    assert((outMatrix.GetNumRows() == nRow) && (outMatrix.GetNumCols() == nCol));

    const size_t ldNbits = ValueQuantizer<ElemType>::ld(nBits);
#ifdef QUANTUSEPPL
    Concurrency::parallel_for((size_t) 0, us.cols(), [&](size_t j)
#else
    for (size_t j = 0; j < nCol; j++)
#endif
                              {
                                  const auto& qcol = *(inQMatrix.GetQuantizedColumn(j));
                                  ColumnQuantizer<ElemType> q(ldNbits, qcol.lower, qcol.upper);
                                  q.Unquantize(outMatrix.Data(), (long) nRow, j, qcol.bits, add);
                              }
#ifdef QUANTUSEPPL
                              );
#endif
}

template <class ElemType>
void MatrixQuantizerCPU<ElemType>::WaitUnquantizeAsyncDone()
{
    // TODO: Currently this is a no-op since the actual quantization is synchronous
}

//The explicit instantiation part will make the linker happy
template class MatrixQuantizerCPU<float>;
template class MatrixQuantizerCPU<double>;
} } }
