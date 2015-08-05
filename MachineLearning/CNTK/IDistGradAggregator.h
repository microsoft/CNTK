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
        
        virtual void AggregateGradients(DistGradHeader<ElemType> *headerCPU) = 0;

        size_t NumProc()
        {
            return m_mpi->nodes();
        }

        size_t MyRank()
        {
            return m_mpi->node();
        }

        void WaitAll()
        {
            m_mpi->waitall();
        }

    protected:
        MPIWrapper* m_mpi;
    };

#define UsingIDistGradAggregatorMembers \
    protected: \
        using IDistGradAggregator<ElemType>::m_mpi;

}}}
