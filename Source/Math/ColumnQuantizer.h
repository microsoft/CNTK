#ifndef __COLUMN_QUANTIZER_H__
#define __COLUMN_QUANTIZER_H__
#include "ValueQuantizer.h"
#include <math.h>

#pragma warning(disable : 4127) // conditional expression is constant

namespace Microsoft { namespace MSR { namespace CNTK {

#define ColMIDX(i, j, numRow) (((j) * (numRow)) + (i)) // 0 based indexing for column major

// ---------------------------------------------------------------------------
// Class to perform columnwise quantization/unquantization
//
// The quantization of a column is performed in 2 steps
// a) Compute the values used for unquantizing/reconstructing the quantized values. This is done by computing a pair of
//    values that specify the range of reconstructed unquantized values, such that the aggregate qunatization error is minimized.
// b) Perform the actual quantization by quantizing each value in the column to an integer of the size of
//    the specified number of bits and then packing these integer bits into the quantized matrix storage
// ---------------------------------------------------------------------------

template <class ElemType>
class ColumnQuantizer
{
    typedef typename ValueQuantizer<ElemType>::QWord QWord;
    typedef typename ValueQuantizer<ElemType>::QWordVal QWordVal;
    static const size_t QWordNumBits = ValueQuantizer<ElemType>::QWordNumBits;

public:
    cudacode ColumnQuantizer(size_t logNbits, ElemType lower, ElemType upper)
        : valQ(logNbits, lower, upper)
    {
    }

    // compute #QWords per column of a given height
    static size_t QWordsPerCol(size_t rows, size_t Nbits)
    {
        const size_t valsPerQWord = QWordNumBits / Nbits;
        return (rows + valsPerQWord - 1) / valsPerQWord;
    }

    size_t QWordsPerCol(size_t rows) const
    {
        return QWordsPerCol(rows, valQ.NBits());
    }

    // quantize a matrix column into qcoldata
    //  The current value of 'inResidual' is added to the matrix, and 'outResidual' gets updated with the new residual;
    //  inResidual = outResidual is allowed (intended)
    template <bool ZeroThresholdFor1Bit>
    cudacode void Quantize(const ElemType* inMat, const ElemType* inResidual, long M, size_t j, QWord* qColBits, ElemType* outResidual) const
    {
        // we loop over QWord values
        // E.g. there are 35 ints for a 1100-dim column (at 1-bit quantization).
        // For better CUDA memory collating, we interleave memory such that computing consecutive ints triggers consecutive memory accesses
        // (although for the CPU side, it breaks caching; we could do in-place op)
        // E.g., int  0 accesses elements 0, 35, 70, etc.
        // while int  1 accesses elements 1, 36, 71, etc
        // up to int 34 accesses elements 34, 69, 104, etc.
        const size_t numQWordsPerCol = QWordsPerCol(M);
        for (size_t iQWord = 0; iQWord < numQWordsPerCol; iQWord++)
        {
            qColBits[iQWord] = QuantizeOneQWord<ZeroThresholdFor1Bit>(inMat, inResidual, M, iQWord, M, numQWordsPerCol, j, outResidual);
        }
    }

    // unquantize a matrix column from qcoldata
    // If 'add' then add to the existing content of the matrix (this is a common thing to do; saves a buffer).
    cudacode void Unquantize(ElemType* outMat, long M, size_t j, const QWord* qColBits, bool add) const
    {
        // loop over QWord values
        const size_t numQWordsPerCol = QWordsPerCol(M);
        for (size_t iQWord = 0; iQWord < numQWordsPerCol; iQWord++)
        {
            UnquantizeOneQWord(outMat, M, iQWord, M, numQWordsPerCol, j, qColBits[iQWord], add);
        }
    }

    // workaround for not being able to declare a default argument for lambda parameters
    template <bool ZeroThresholdFor1Bit>
    static cudacode void ComputeRangeStatColj(const ElemType* inMat, const ElemType* inResidual, long M, size_t j, size_t bits, ElemType& lower, ElemType& upper)
    {
        /*dummy reducers do nothing in linear CPU version*/
        ComputeRangeStatColjSubset<ZeroThresholdFor1Bit>(inMat, inResidual, M, j, bits, lower, upper, 0, 1, [](ElemType&)
                                                         {
                                                         },
                                                         [](unsigned int&)
                                                         {
                                                         });
    }

public:
    // quantize the value in  inMat[rowStart,colIdx],  inMat[rowStart + rowStride,colIdx],inMat[rowStart + rowStride*,colIdx]  ... and pack them into a QWord
    // Question: note that it is somewhat un-intuitional, but this memory access pattern is efficient for GPU?
    template <bool ZeroThresholdFor1Bit>
    cudacode QWord QuantizeOneQWord(
        const ElemType* inMat, const ElemType* inResidual,
        long M,
        size_t rowStart, size_t rowEnd, size_t rowStride,
        size_t colIdx,
        ElemType* outResidual) const
    {
        QWord bitBuf = 0;

        if ((valQ.NBits() == 1) && (inResidual == outResidual) /*in-place*/)
        {
            ElemType val0 = valQ.Unquantize(0);
            ElemType val1 = valQ.Unquantize(1);
            size_t ij = ColMIDX(rowStart, colIdx, M);
            const ElemType* usibj = inMat + ij;
            const ElemType* usibjend = usibj + (rowEnd - rowStart);
            ElemType* resibj = outResidual + ij;
            // we know that the range covers at most the number of bits in a 'QWord'
            for (QWord bitmask = 1; usibj < usibjend; bitmask <<= 1, usibj += rowStride, resibj += rowStride)
            {
                // quantize   --we access element (i,j) through the three increasing pointers
                ElemType val = *usibj + *resibj;

                // Explicit use of 'template' keyword is needed to compile with GCC
                bool qval = valQ.template Quantize1<ZeroThresholdFor1Bit>(val);
                if (qval)
                {
                    bitBuf |= bitmask;
                }

                // compute residual
                ElemType uval = valQ.Unquantize1(qval, val0, val1);
                *resibj = val - uval;
            }
        }
        else
        {
            // number of bits in a QWord
            size_t i = rowStart;
            for (size_t k = 0; (k < QWordNumBits) && (i < rowEnd); k += valQ.NBits(), i += rowStride)
            {
                // quantize
                size_t ij = ColMIDX(i, colIdx, M);
                ElemType val = inMat[ij] + inResidual[ij];
                QWordVal qval = valQ.Quantize<ZeroThresholdFor1Bit>(val);

                // compute residual
                ElemType uval = valQ.Unquantize(qval);
                ElemType r = val - uval;
                outResidual[ij] = r;
                bitBuf = bitBuf | (qval << k);
            }
        }
        return bitBuf;
    }

    // unquantize one QWord of a quantized matrix column
    cudacode void UnquantizeOneQWord(
        ElemType* us, long M,
        size_t rowStart, size_t rowEnd, size_t rowStride,
        size_t colIdx, QWord bitBuf, bool add) const
    {
        // special case for 1 bit
        if (valQ.NBits() == 1)
        {
            ElemType val0 = valQ.Unquantize(0);
            ElemType val1 = valQ.Unquantize(1);
            size_t ij = ColMIDX(rowStart, colIdx, M);
            ElemType* usibj = us + ij;
            const ElemType* usibjend = usibj + (rowEnd - rowStart);
            for (; usibj < usibjend; usibj += rowStride)
            {
                // get value
                // bitbuf is shifted in-place
                bool qval = (bitBuf & 1) != 0;

                // and get bitbuf into next position
                bitBuf >>= 1;

                // unquantize
                ElemType val = ValueQuantizer<ElemType>::Unquantize1(qval, val0, val1);
                if (add)
                {
                    val += *usibj;
                }

                *usibj = val;
            }
        }
        else
        {
            // (rangeend MUST be a power of two; ensured by constructing off ldNbits)
            const QWordVal bitmask = valQ.QuanRangeEnd() - 1;
            size_t i = rowStart;
            for (size_t k = 0; (k < QWordNumBits) && (i < rowEnd); k += valQ.NBits(), i += rowStride)
            {
                // get value
                const QWordVal qval = (bitBuf >> k) & bitmask; // % 2^Nbits

                // unquantize
                ElemType val = valQ.Unquantize(qval);
                size_t ij = ColMIDX(i, colIdx, M);
                if (add)
                {
                    val += us[ij];
                }

                us[ij] = val;
            }
        }
    }

    // determine quantization range of one column
    // This code is written so that it can run in parallel threads on CUDA for collated memory access;
    // set 'subsets' to >1 and pass cross-thread reducer functions for 'float' and 'size_t' (which would reduce through using CUDA __shared__ memory).
    // TODO: further opportunity for speed-up: use 'mean' from last round for 1-bit and stddev calc
    template <bool ZeroThresholdFor1Bit, class F1, class F2>
    static cudacode void ComputeRangeStatColjSubset(
        const ElemType* inMat,
        const ElemType* inResidual, long M,
        size_t j,
        size_t bits,
        ElemType& lower, ElemType& upper,
        size_t subset, size_t subsets,
        F1 allReduceElem, F2 allReduceUint)
    {
        // quantization range, cut off after how many standard deviations (make this a parameter if we care)
        size_t rows = M;

        // compute mean
        // computing the mean is expensive; we assume there is no reason for asymmetry and thus a zero mean
        // an initial experiment showed that this is significantly worse (36.0 vs. 37.7% frame acc) at the start, but seems to recover nearly (minor gap)
        // thought:
        //  - we could set the threshold at 0
        //  - but keep the quantization values for 0 and 1 separate
        // i.e.
        //  - do not symmetrize/pool the quantization values for 0 and 1
        //  - but hard-code the quantization threshold to be 0 instead of the mean of the two bounds
        // This should give us the best of all--fast operation yet ability to be asymmetric within a column
        ElemType mean = 0.0f;
        if (!ZeroThresholdFor1Bit || (bits != 1))
        {
            ElemType meanacc = 0.0f;
            // (subset: compute subset sum)
            for (size_t i = subset; i < rows; i += subsets)
            {
                size_t ij = ColMIDX(i, j, M);
                meanacc += inMat[ij] + inResidual[ij];
            }
            // multi-subset (CUDA): reduce to one thread
            allReduceElem(meanacc);
            mean = meanacc / rows;
        }

        if (bits == 1)
        {
            // 1-bit case:
            // We want to minimize the (squared) reconstruction error within the two levels.
            // I.e. we should reconstruct to the respective means of each level.
            // To be able to express the range by two floats, we approximate the level threshold as the av. of the two level means.
            // compute the two level means
            ElemType meanacc0 = 0.0f, meanacc1 = 0.0f;
            unsigned int num0 = 0, num1 = 0;
            // (subset: compute subset sum)
            for (size_t i = subset; i < rows; i += subsets)
            {
                size_t ij = ColMIDX(i, j, M);
                ElemType val = inMat[ij] + inResidual[ij];
                if (val < mean)
                {
                    meanacc0 += val;
                    num0++;
                }
                else
                {
                    meanacc1 += val;
                    num1++;
                }
            }

            // multi-subset (CUDA): reduce to one thread
            allReduceElem(meanacc0);
            allReduceElem(meanacc1);
            allReduceUint(num0);
            allReduceUint(num1);

            ElemType radius;
            ElemType newmean;
            if (!ZeroThresholdFor1Bit)
            {
                // we minimize the error jointly across positive and negative numbers to make things
                // symmetrical around the mean (which may be non-zero) tying the two sides
                ElemType devacc0 = (num0 * mean) - meanacc0;
                ElemType devacc1 = meanacc1 - (num1 * mean);

                // both deviations tied, to ensure consistent mean
                ElemType dev = (devacc0 + devacc1) / rows;
                radius = 2.0f * dev;
                newmean = mean;
            }
            else
            {
                // we keep two separate reconstruction values to allow for asymmetries--but we
                // instead hard-code that the threshold is 0

                // happens for all-zero columns which do exist (mean0 is 0 in that case)
                if (num0 == 0)
                    num0 = 1;
                if (num1 == 0)
                    num1 = 1;
                ElemType mean0 = meanacc0 / num0;
                ElemType mean1 = meanacc1 / num1;

                // approximate by using their average as the threshold between 0 and 1
                // with these values, bits (0,1) which mean values (0.5,1.5) will reconstruct to mean0/1
                newmean = 0.5f * (mean0 + mean1);
                radius = 2.0f * (mean1 - newmean);
            }

            if (subset == 0)
            {
                lower = newmean - radius;
                upper = newmean + radius;
            }
        }
        else
        {
            ElemType stddevs = 5.0f;
            // >1 bit:
            // We linearly quantize between 'stddevs' standard deviations.
            ElemType varacc = 0.0f;
            // (subset: compute subset sum)
            for (size_t i = subset; i < rows; i += subsets)
            {
                size_t ij = ColMIDX(i, j, M);
                ElemType val = inMat[ij] + inResidual[ij];
                varacc += (val - mean) * (val - mean);
            }
            // multi-subset (CUDA): reduce to one thread
            allReduceElem(varacc);
            ElemType stddev = sqrt(varacc / rows);
            if (subset == 0)
            {
                // stddevs = how many stddevs from the mean until outside of quantization range
                lower = mean - (stddevs * stddev);
                upper = mean + (stddevs * stddev);
            }
        }
    }

private:
    ValueQuantizer<ElemType> valQ;

    template <typename T>
    friend class QuantizedMatrix;
};
}
}
}
#endif
