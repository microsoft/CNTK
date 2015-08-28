#ifndef __VALLUE_QUANTIZER_CUH__
#define __VALLUE_QUANTIZER_CUH__

#include "stdafx.h"
#include "ValueQuantizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    cudasharedcode
    ValueQuantizer<ElemType>::ValueQuantizer(size_t ldNbits, ElemType lower, ElemType upper) 
    : ldNbits(ldNbits), Nbits(1 << ldNbits), quantimin(lower), quantimax(upper)
    {
        rangeend = ((QWordVal)1) << Nbits;

        // post-fix for incorrect shift for no-quant hack (Nbits=32): << arg is taken mod 32!
        // in this case, it's only used as (rangeend-1) which is now correct (before it was 0!)
        if (Nbits >= (8 * sizeof(rangeend)))
        {
            rangeend = 0;
        }

        // must protect against NaN: interval is 0 -> quantization is futile, just emit 0
        if (((quantimax - quantimin) < 1e-36f) || (rangeend == 0))
        {
            qfactor = ufactor = (ElemType)0.0;
        }
        else
        {
            // precompute this for quantize() (see comment there)
            qfactor = rangeend / (quantimax - quantimin);   
            // and for unquantize()
            ufactor = (quantimax - quantimin) / rangeend;   
        }

        // set the quantization threshold for the special case of 1-bit
        quantimid = 0.5f * (quantimax + quantimin);
    }

    // quantize for full ElemType size bits case (special case that allows to bypass quantization, for testing/debugging purposes)
    template<class ElemType>
    cudasharedcode ValueQuantizer<ElemType>::QWordVal
    ValueQuantizer<ElemType>::QuantizeToFullQWord(ElemType u) const
    {
        assert(Nbits == QWordNumBits);
        
        // we return the bit pattern that encodes the float value
        return *(QWordVal*)&u;
    }

    // quantize one value --special version for 1 bit
    template<class ElemType>
    template<bool ZeroThresholdFor1Bit>
    cudasharedcode bool
    ValueQuantizer<ElemType>::Quantize1(ElemType u) const
    {
        assert (Nbits == 1);
        if (!ZeroThresholdFor1Bit)
        {
            return u >= quantimid;
        }
        else
        {
            return u >= (ElemType)0.0;
        }
    }

    // quantize one value
    // TODO: we can optimize for 1 bit here - very simply use a template arg 'isonebit'
    template<class ElemType>
    template<bool ZeroThresholdFor1Bit>
    cudasharedcode ValueQuantizer<ElemType>::QWordVal
    ValueQuantizer<ElemType>::Quantize(ElemType u) const
    {
        if (Nbits == QWordNumBits)
        {
            return QuantizeToFullQWord(u);
        }
        // TODO: we may need to optimize this by a template arg
        else if (ldNbits == 0)
        {
            return Quantize1<ZeroThresholdFor1Bit>(u) ? 1 : 0;
        }
        else
        {
            if (u <= quantimin)
            {
                return 0;
            }
            else if (u >= quantimax)
            {
                return (rangeend - 1);
            }
            else
            {
                return (QWordVal)((QWordValSigned)((u - quantimin) * qfactor));
            }
        }
    }

    // unquantize one value
    template<class ElemType>
    cudasharedcode  
    ElemType ValueQuantizer<ElemType>::Unquantize(QWordVal u) const
    {
        if (Nbits == QWordNumBits)
        {
            return *(ElemType*)&u;
        }
        
        // Note: in 1-bit case, we want 0.5 -> mean0, 1.5 -> mean1
        return ((u + (ElemType)0.5) * ufactor) + quantimin;
    }

    // unquantize one value  --special case for 1 bit
    template<class ElemType>
    cudasharedcode 
    ElemType ValueQuantizer<ElemType>::Unquantize1(bool u, ElemType val0, ElemType val1)
    {
        return u ? val1 : val0;
    }

    // helper: compute the binary log of a power of two (utility function to convert 'Nbits' into 'ldNbits'
    template<class ElemType>
    size_t ValueQuantizer<ElemType>::ld(size_t v)
    {
        if (v == 1)
        {
            return 0;
        }
        else if (v & 1) // not a power of two
        {
            throw std::runtime_error("ld: 'bits' must be a power of two");
        }
        else
        {
            return 1 + ld (v >> 1);
        }
    }
    
    // Explicit instantiation
    template class ValueQuantizer<float>;
    template class ValueQuantizer<double>;
    template ValueQuantizer<float>::QWordVal ValueQuantizer<float>::Quantize<true>(float u) const;
    template ValueQuantizer<float>::QWordVal ValueQuantizer<float>::Quantize<false>(float u) const;
    template ValueQuantizer<double>::QWordVal ValueQuantizer<double>::Quantize<true>(double u) const;
    template ValueQuantizer<double>::QWordVal ValueQuantizer<double>::Quantize<false>(double u) const;
    template bool ValueQuantizer<float>::Quantize1<true>(float u) const;
    template bool ValueQuantizer<float>::Quantize1<false>(float u) const;
    template bool ValueQuantizer<double>::Quantize1<true>(double u) const;
    template bool ValueQuantizer<double>::Quantize1<false>(double u) const;
}}}
#endif
