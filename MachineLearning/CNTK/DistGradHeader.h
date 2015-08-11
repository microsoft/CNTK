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

        static DistGradHeader<ElemType>* Create(int numEvalNode)
        {
            DistGradHeader<ElemType>* header = (DistGradHeader<ElemType>*)new char[DistGradHeaderSize(numEvalNode)];
            header->numEvalNode = numEvalNode;
            return header;
        }

        static void Destroy(DistGradHeader* header)
        {
            delete[]((char*)header);
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

        size_t Size() const
        {
            return DistGradHeaderSize(numEvalNode);
        }

    private:
        static size_t DistGradHeaderSize(size_t nEvalNode)
        {
            return sizeof(DistGradHeader<ElemType>) + (sizeof(ElemType) * (nEvalNode - 1));
        }

        // Disallow construction and destruction since this type contains a variable sized array member
        // and hence must be constructed through the create and destroy functions
        DistGradHeader() = delete;
        ~DistGradHeader() = delete;

        // Disallow copy and move construction/assignment
        DistGradHeader(const DistGradHeader&) = delete;
        DistGradHeader& operator=(const DistGradHeader&) = delete;
        DistGradHeader(DistGradHeader&&) = delete;
        DistGradHeader& operator=(DistGradHeader&&) = delete;
    };
}}}
