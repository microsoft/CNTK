#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    struct DistGradHeader
    {
    public:
        size_t numSamples;
        size_t numSamplesWithLabel;
        double criterion;

        // variable-size array
        int numEvalNode;
        double evalErrors[1];

        static DistGradHeader* Create(int numEvalNode)
        {
            DistGradHeader* header = (DistGradHeader*)new char[DistGradHeaderSize(numEvalNode)];
            header->numEvalNode = numEvalNode;
            return header;
        }

        static void Destroy(DistGradHeader* header)
        {
            delete[]((char*)header);
        }

        //aggregate header information
        void Aggregate(DistGradHeader* other, bool add = false)
        {
            if (other->numEvalNode != numEvalNode)
                RuntimeError("mismatched size");
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
            return sizeof(DistGradHeader)+(sizeof(double) * (nEvalNode - 1));
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
