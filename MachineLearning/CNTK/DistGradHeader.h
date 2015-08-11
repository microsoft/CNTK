#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    template<typename ElemType>
    struct DistGradHeader
    {
    public:
        size_t numSamples;
        size_t numSamplesWithLabel;
        ElemType criterion;

        // variable-size array
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
                numSamples += other->numSamples;
                numSamplesWithLabel += other->numSamplesWithLabel;
                criterion += other->criterion;
                for (int i = 0; i < numEvalNode; i++)
                {
                    evalErrors[i] += other->evalErrors[i];
                }
            }
        }
    };
}}}
