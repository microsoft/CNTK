#pragma once

#include "DistGradHeader.h"
#include "MPIWrapper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class IDistGradAggregator
    {
    public:
        IDistGradAggregator(MPIWrapper* mpi)
            : m_mpi(mpi)
        {
        }
        
        virtual ~IDistGradAggregator()
        {
        }
        
        virtual void AggregateGradients(DistGradHeader<ElemType> *headerCPU, int epochNumber) = 0;

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
        MPIWrapper* m_mpi;
    };

#define UsingIDistGradAggregatorMembers \
    protected: \
        using IDistGradAggregator<ElemType>::m_mpi; using IDistGradAggregator<ElemType>::NumProc; using IDistGradAggregator<ElemType>::MyRank

}}}
