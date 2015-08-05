#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    template<typename ElemType>
    struct DistGradHeader
    {
    public:
        size_t numSample;
        ElemType criterion;

        // variable-size array to hold the bits, grouped into 'qbwords'
        int numEvalNode;
        ElemType evalErrors[1];

        static size_t DistGradHeaderSize(size_t nEvalNode)
        {
            return sizeof(DistGradHeader<ElemType>) + (sizeof(ElemType) * (nEvalNode - 1));
        }

        //aggregate header information
        void Aggregate(DistGradHeader<ElemType>* other, bool add = false)
        {
            if (other->numEvalNode != numEvalNode)
            {
                throw  std::runtime_error("mismatched size");
            }
            if (!add)
            {
                memcpy((void*)this, (void*)other, DistGradHeaderSize(numEvalNode));
            }
            else
            {
                criterion += other->criterion;
                numSample += other->numSample;
                for (int i = 0; i < numEvalNode; i++)
                {
                    evalErrors[i] += other->evalErrors[i];
                }
            }
        }
    };
}}}
