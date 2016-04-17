#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

struct DistGradHeader
{
public:
    size_t numSamples;
    size_t numSamplesWithLabel; // this is the denominator for 'criterion'
    double criterion;

    // variable-size array
    int numEvalNode;
    pair<double,size_t> evalErrors[1];

    static DistGradHeader* Create(int numEvalNode)
    {
        DistGradHeader* header = (DistGradHeader*) new char[DistGradHeaderSize(numEvalNode)];
        header->numEvalNode = numEvalNode;
        return header;
    }

    static void Destroy(DistGradHeader* header)
    {
        delete[]((char*) header);
    }

    // aggregate header information
    void Aggregate(DistGradHeader* other, bool add = false)
    {
        if (other->numEvalNode != numEvalNode)
            RuntimeError("mismatched size");
        if (!add)
        {
            memcpy((void*) this, (void*) other, DistGradHeaderSize(numEvalNode));
        }
        else
        {
            numSamples += other->numSamples;
            numSamplesWithLabel += other->numSamplesWithLabel;
            criterion += other->criterion;
            for (int i = 0; i < numEvalNode; i++)
            {
                evalErrors[i].first  += other->evalErrors[i].first;  // numer
                evalErrors[i].second += other->evalErrors[i].second; // denom
            }
        }
    }

    size_t Size() const
    {
        return DistGradHeaderSize(numEvalNode);
    }

    void Clear()
    {
        numSamples = 0;
        numSamplesWithLabel = 0;
        criterion = 0;
        for (int i = 0; i < numEvalNode; i++)
        {
            evalErrors[i].first  = 0;
            evalErrors[i].second = 0;
        }
    }

    friend void swap(DistGradHeader& first, DistGradHeader& second)
    {
        if (first.numEvalNode != second.numEvalNode)
            LogicError("Cannot swap DistGradHeader objects with different number of evalNodes!");

        std::swap(first.numSamples, second.numSamples);
        std::swap(first.numSamplesWithLabel, second.numSamplesWithLabel);
        std::swap(first.criterion, second.criterion);
        for (int i = 0; i < first.numEvalNode; i++)
        {
            std::swap(first.evalErrors[i], second.evalErrors[i]);
        }
    }

private:
    static size_t DistGradHeaderSize(size_t nEvalNodes)
    {
        // BUGBUG: Should be sizeof(evalErrors[0]), but the compiler won't let me. This is only correct because evalErrors has 1 element.
        return sizeof(DistGradHeader) + (sizeof(decltype(evalErrors)) * (nEvalNodes - 1));
    }

    // Disallow construction and destruction since this type contains a variable sized array member
    // and hence must be constructed through the create and destroy functions
    DistGradHeader()  = delete;
    ~DistGradHeader() = delete;

    // Disallow copy and move construction/assignment
    DISABLE_COPY_AND_MOVE(DistGradHeader);
};

}}}
