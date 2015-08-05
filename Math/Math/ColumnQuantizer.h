#ifndef __COLUMN_QUANTIZER_H__
#define __COLUMN_QUANTIZER_H__
#include "ValueQuantizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    
    #define ColMIDX(i,j,numRow) (((j)*(numRow))+(i)) // 0 based indexing for column major

    // ---------------------------------------------------------------------------
    // quantization of one column
    // ---------------------------------------------------------------------------    

    template<class ElemType>
    class ColumnQuantizer
    {
    public:
        cudacode ColumnQuantizer(size_t logNbits, ElemType lower, ElemType upper) 
        : valQ(logNbits, lower, upper)
        {        
        }
        
        // compute #qbwords per column of a given height
        static size_t QbwordsPerCol(size_t rows, size_t Nbits) 
        {
            // how many quantized values fit into one qbword (32 in 1-bit case)
            const size_t valsperqbword = QBWordBits / Nbits;  
            
            // how many qbwords do we need to store the column
            return (rows + valsperqbword - 1) / valsperqbword;  
        }
        
        size_t QbwordsPerCol(size_t rows) const
        {
            return QbwordsPerCol(rows, valQ.NBits());        
        }

        // quantize a matrix column into qcoldata
        //  The current value of 'inResidual' is added to the matrix, and 'outResidual' gets updated with the new residual;
        //  inResidual = outResidual is allowed (intended)
        cudacode void Quantize(const ElemType* inMat, const ElemType* inResidual, long M, size_t j, QBWord* qcolbits, ElemType* outResidual) const;

        // unquantize a matrix column from qcoldata
        // If 'add' then add to the existing content of the matrix (this is a common thing to do; saves a buffer).
        cudacode void Unquantize(ElemType* outMat, long M, size_t j, const QBWord* QBWord, bool add) const;

        // workaround for not being able to declare a default argument for lambda parameters
        static cudacode void ComputeRangeStatColj(const ElemType* inMat, const ElemType* inResidual, long M, size_t j, size_t bits, ElemType& lower, ElemType& upper)
        {
            /*dummy reducers do nothing in linear CPU version*/
            ComputeRangeStatColjSubset(inMat, inResidual, M, j, bits, lower, upper, 0, 1, [](ElemType&){}, [](unsigned int&){});
        }

    public:
        
        //quantized the value in  inMat[rowStart,colIdx],  inMat[rowStart + rowStride,colIdx],inMat[rowStart + rowStride*,colIdx]  ... and pack them into a Qbword
        //Question: note that it is somewhat un-intuitional, but this memory access pattern is efficient for GPU?
        cudacode QBWord QuantizeOneQbword(
            const ElemType* inMat, const ElemType* inResidual,
            long M, 
            size_t rowStart, size_t rowEnd, size_t rowStride,
            size_t colIdx, 
            ElemType* outResidual) const; 

        // unquantize one qbword of a quantized matrix column
        cudacode void UnquantizeOneQbword(
            ElemType* us, long M, 
            size_t rowStart, size_t rowEnd, size_t rowStride,
            size_t colIdx, QBWord bitbuf, bool add) const;

        // determine quantization range of one column
        // This code is written so that it can run in parallel threads on CUDA for collated memory access;
        // set 'subsets' to >1 and pass cross-thread reducer functions for 'float' and 'size_t' (which would reduce through using CUDA __shared__ memory).
        // TODO: further opportunity for speed-up: use 'mean' from last round for 1-bit and stddev calc
        template<class F1, class F2>
        static cudacode void ComputeRangeStatColjSubset(
            const ElemType* inMat,
            const ElemType* inResidual, long M,
            size_t j,
            size_t bits,
            ElemType& lower, ElemType& upper,
            size_t subset, size_t subsets,
            F1 allReduceElem, F2 allReduceUint);

    public:
        // quantized values are stored in groups of 'qbwords' = unsigned ints (which happen to memory-align with 'float' as used in 'quantizedcolumn' structure)
        static const size_t  QBWordBits = 8 * sizeof (QBWord);   // number of bits in a qbword (32)

    private:
        ValueQuantizer<ElemType> valQ;
    };

    //-------------------definitions----------------
    // compute one qbword value of a quantized matrix column
    template<class ElemType>
    cudacode QBWord ColumnQuantizer<ElemType>::QuantizeOneQbword(
        const ElemType* inMat,
        const ElemType* inResidual,
        long M,
        size_t rowStart,
        size_t rowEnd,
        size_t rowStride,
        size_t j,
        ElemType* outResidual) const
    {
        QBWord bitBuf = 0;

        if ((valQ.NBits() == 1) && (inResidual == outResidual)/*in-place*/)
        {
            ElemType val0 = valQ.Unquantize(0);
            ElemType val1 = valQ.Unquantize(1);
            size_t ij = ColMIDX(rowStart,j,M);
            const ElemType* usibj = inMat + ij;
            const ElemType* usibjend = usibj + (rowEnd - rowStart);
            ElemType* resibj = outResidual + ij;
            for (QBWord bitmask = 1;
                usibj < usibjend; // we know that the range covers at most 'qbwordbits' bits
                bitmask <<= 1, usibj += rowStride, resibj += rowStride)
            {
                // quantize   --we access element (i,j) through the three increasing pointers
                ElemType val = *usibj + *resibj;
                bool qval = valQ.Quantize1(val);
                if (qval)
                {
                    bitBuf |= bitmask;
                }
                
                // compute residual
                ElemType uval = valQ.Unquantize1(qval, val0, val1);
    #undef NORESIDUAL  // set this to test without residual--does it still work?
    #ifdef NORESIDUAL
                *resibj = 0.0f;
    #else
                *resibj = val - uval;
    #endif
            }
        }
        else
        {
            // number of bits in a qbword
            const unsigned int qbWordBits = 8 * sizeof (QBWord);     
            size_t i = rowStart;
            for (size_t k = 0;
                k < qbWordBits && i < rowEnd;
                k += valQ.NBits(), i += rowStride)
            {
                // quantize
                size_t ij = ColMIDX(i,j,M);//col-major 0-based index
                ElemType val = inMat[ij] + inResidual[ij];
                unsigned int qval = valQ.Quantize(val);

                // compute residual
                ElemType uval = valQ.Unquantize(qval);
                ElemType r = val - uval;
                outResidual[ij] = r;
                bitBuf = bitBuf | (qval << k);
            }
        }
        return bitBuf;
    }


    template<class ElemType>
    cudacode void ColumnQuantizer<ElemType>::UnquantizeOneQbword(
        ElemType*us, long M, 
        size_t rowStart, size_t rowEnd, size_t rowStride,
        size_t j, QBWord bitbuf, bool add) const
    {
        // special case for 1 bit
        if (valQ.NBits() == 1)   
        {
            ElemType val0 = valQ.Unquantize(0);
            ElemType val1 = valQ.Unquantize(1);
            size_t ij = ColMIDX(rowStart, j, M);
            ElemType* usibj = us + ij;
            const ElemType* usibjend = usibj + (rowEnd - rowStart);
            for (; usibj < usibjend; usibj += rowStride)
            {
                // get value
                // bitbuf is shifted in-place
                bool qval = (bitbuf & 1) != 0;    
                
                // and get bitbuf into next position
                bitbuf >>= 1;                           
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
            const size_t bitmask = valQ.QuanRangeEnd() - 1;                     
            size_t i = rowStart;
            for (size_t k = 0; k < qbwordbits && i < rowEnd; k += valQ.NBits(), i += rowStride)
            {
                // get value
                const unsigned int qval = (bitbuf >> k) & bitmask;  // % 2^Nbits
                // unquantize
                ElemType val = valQ.Unquantize(qval);
                size_t ij = ColMIDX(i, j, M);
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
    //computerange in dbn.exe
    template<class ElemType>
    template<class F1, class F2>
    cudacode void ColumnQuantizer<ElemType>::ComputeRangeStatColjSubset(
        const ElemType* us,
        const ElemType* inResidual, long M,
        size_t j,
        size_t bits, 
        ElemType& lower, ElemType& upper,
        size_t subset, size_t subsets,
        F1 allReduceElem, F2 allReduceUint)
    {
    #ifdef WIN32
    #ifndef INCLUDE_RESIDUE_FOR_QUANTIZATION_RANGE
        UNREFERENCED_PARAMETER(inResidual);
    #endif
    #endif

        // quantization range, cut off after how many standard deviations (make this a parameter if we care)
        size_t rows = M;
        // compute mean
        // computing the mean is expensive; we assume there is no reason for asymmetry and thus a zero mean
    #if defined (ZERO_THRESHOLD_FOR_1BIT)   
        // an initial experiment showed that this is significantly worse (36.0 vs. 37.7% frame acc) at the start, but seems to recover nearly (minor gap)
        // thought:
        //  - we could set the threshold at 0
        //  - but keep the quantization values for 0 and 1 separate
        // i.e.
        //  - do not symmetrize/pool the quantization values for 0 and 1
        //  - but hard-code the quantization threshold to be 0 instead of the mean of the two bounds
        // This should give us the best of all--fast operation yet ability to be asymmetric within a column
        ElemType mean = 0.0f;
    #else
        ElemType meanacc = 0.0f;
        // (subset: compute subset sum)
        for (size_t i = subset; i < rows; i += subsets)     
        {
            size_t ij = ColMIDX(i, j, M);
            meanacc += us[ij];
    #ifdef INCLUDE_RESIDUE_FOR_QUANTIZATION_RANGE
            meanacc += inResidual[ij];
    #endif
        }
        // multi-subset (CUDA): reduce to one thread
        allReduceElem(meanacc);
        ElemType mean = meanacc / rows;
    #endif

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
                ElemType val = us[ij];
    #ifdef INCLUDE_RESIDUE_FOR_QUANTIZATION_RANGE
                val += inResidual[ij];
    #endif
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
    #ifndef ZERO_THRESHOLD_FOR_1BIT    
            // we minimize the error jointly across positive and negative numbers to make things symmetrical around the mean (which may be non-zero)
            // tying the two sides
            ElemType devacc0 = num0 * mean - meanacc0;
            ElemType devacc1 = meanacc1 - num1 * mean;
            
            // both deviations tied, to ensure consistent mean
            ElemType dev = (devacc0 + devacc1) / rows;   
            ElemType radius = 2.0f * dev;
            ElemType newmean = mean;
    #else       
            // we keep two separate reconstruction values to allow for asymmetries--but we instead hard-code that the threshold is 0
            
            // happens for all-zero columns which do exist (mean0 is 0 in that case)
            if (num0 == 0) num0 = 1;                        
            if (num1 == 0) num1 = 1;
            ElemType mean0 = meanacc0 / num0;
            ElemType mean1 = meanacc1 / num1;
            
            // approximate by using their average as the threshold between 0 and 1
            // with these values, bits (0,1) which mean values (0.5,1.5) will reconstruct to mean0/1
            ElemType newmean = 0.5f * (mean0 + mean1);           
            ElemType radius = 2.0f * (mean1 - newmean);  
    #endif
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
                ElemType val = us[ij];
    #ifdef INCLUDE_RESIDUE_FOR_QUANTIZATION_RANGE
                val += inResidual[ij];
    #endif
                varacc += (val-mean) * (val-mean);
            }
            // multi-subset (CUDA): reduce to one thread
            allReduceElem(varacc);
            ElemType stddev = sqrt(varacc / rows);
            if (subset == 0)
            {
                // stddevs = how many stddevs from the mean until outside of quantization range
                lower = mean - stddevs * stddev;            
                upper = mean + stddevs * stddev;
            }
        }
    }

    template<class ElemType>
    cudacode void ColumnQuantizer<ElemType>::Quantize(const ElemType* inMat, const ElemType* inResidual, long M, size_t j, QBWord* qcolbits, ElemType* outResidual) const
    {
        // we loop over qbword values
        // E.g. there are 35 ints for a 1100-dim column (at 1-bit quantization).
        // For better CUDA memory collating, we interleave memory such that computing consecutive ints triggers consecutive memory accesses
        // (although for the CPU side, it breaks caching; we could do in-place op)
        // E.g., int  0 accesses elements 0, 35, 70, etc.
        // while int  1 accesses elements 1, 36, 71, etc
        // up to int 34 accesses elements 34, 69, 104, etc.
        const size_t numqbwordspercol = QbwordsPerCol (M);
        for (size_t iqbword = 0; iqbword < numqbwordspercol; iqbword++)
        {
            qcolbits[iqbword] = QuantizeOneQbword(inMat, inResidual, M,  iqbword, M, numqbwordspercol, j, outResidual);
        }
    }

    template<class ElemType> 
    cudacode void ColumnQuantizer<ElemType>::Unquantize(ElemType* outMat, long M,  size_t j, const QBWord* qcolbits, bool add) const
    {
        // loop over qbword values
        const size_t numqbwordspercol = QbwordsPerCol (M);
        for (size_t iqbword = 0; iqbword < numqbwordspercol; iqbword++)
        {
            UnquantizeOneQbword (outMat, M, iqbword, M, numqbwordspercol, j, qcolbits[iqbword], add);
        }
    }

}}}
#endif

