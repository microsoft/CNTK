//
// <copyright file="PTaskComputationNetwork.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <string>
#include "ComputationNetwork.h"

#include "PTask.h"
#include "PTaskGraphBuilder.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class PTaskComputationNetwork : public ComputationNetwork<ElemType>
    {
    public:
        PTaskComputationNetwork(short deviceId=AUTOPLACEMATRIX) 
            : ComputationNetwork<ElemType>(deviceId)
        {
            m_PTaskGraphBuilder = new PTaskGraphBuilder<ElemType>();
        }

        virtual ~PTaskComputationNetwork() { }

        virtual void LoadFromFile(const std::wstring& fileName, FileOptions fileFormat = FileOptions::fileOptionsBinary) override
        {
            // Let the base class implementation deserialize all the state
            // and construct its regular CN ...
            this->ComputationNetwork<ElemType>::LoadFromFile(fileName, fileFormat);

            // ... then use that state to create the corresponding PTask graph.
            m_PTaskGraphBuilder->BuildFromComputationNetwork(this);
        }

        virtual void ComputeGradient(ComputationNodePtr rootNode) override
        {
            //printf("PTaskComputationNetwork::ComputeGradient called.\n");
            this->ComputationNetwork<ElemType>::ComputeGradient(rootNode);
        }

    private:
        // Copy constructor, should never be called.
        PTaskComputationNetwork(const PTaskComputationNetwork<ElemType>& deepCopyFrom) {};

        // Assignment operator, should never be called.
        PTaskComputationNetwork<ElemType>& operator=(const PTaskComputationNetwork<ElemType>& deepCopyFrom) 
        {
            assert(false);
            return const_cast<PTaskComputationNetwork<ElemType>&>(deepCopyFrom); // return a value to avoid compile errors
        };

        PTaskGraphBuilder<ElemType>*    m_PTaskGraphBuilder;
    };

}}}
