#pragma once

#include "DistGradHeader.h"
#include "MPIWrapper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class IDistGradAggregator
{
public:
    IDistGradAggregator(const MPIWrapperPtr& mpi)
        : m_mpi(mpi)
    {}

    virtual ~IDistGradAggregator()
    {}

    // Returns a boolean indicating if any samples were processed
    virtual bool AggregateGradients(const std::vector<Matrix<ElemType>*>& gradients, DistGradHeader* headerCPU, bool resetState) = 0;

    size_t NumProc()
    {
        return m_mpi->NumNodesInUse();
    }

    size_t MyRank()
    {
        return m_mpi->CurrentNodeRank();
    }

    void WaitAll()
    {
        m_mpi->WaitAll();
    }

protected:
    MPIWrapperPtr m_mpi;
};

#define UsingIDistGradAggregatorMembers           \
    \
protected:                                        \
    using IDistGradAggregator<ElemType>::m_mpi;   \
    using IDistGradAggregator<ElemType>::NumProc; \
    using IDistGradAggregator<ElemType>::MyRank
} } }
