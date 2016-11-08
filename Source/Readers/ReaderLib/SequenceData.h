//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains helper classes for exposing sequence data in deserializers.
//

#pragma once

#include "DataDeserializer.h"
#include "ConcStack.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Class represents a sparse sequence for category data.
    // m_data is a non-owning pointer to some staticlly allocated category.
    // TOOD: Possibly introduce typed data here.
    struct CategorySequenceData : SparseSequenceData
    {
        const void* GetDataBuffer() override
        {
            return m_data;
        }

        // Non-owning pointer to the static data describing the label.
        void *m_data;
    };

    typedef std::shared_ptr<CategorySequenceData> CategorySequenceDataPtr;

    // The class represents a sequence that returns the internal data buffer
    // back to the stack when destroyed.
    template<class TElemType>
    struct DenseSequenceWithBuffer : DenseSequenceData
    {
        DenseSequenceWithBuffer(conc_stack<std::vector<TElemType>>& memBuffers, size_t numberOfElements) : m_memBuffers(memBuffers)
        {
            m_buffer = m_memBuffers.pop_or_create([numberOfElements]() { return vector<TElemType>(numberOfElements); });
            m_buffer.resize(numberOfElements);
        }

        const void* GetDataBuffer() override
        {
            return m_buffer.data();
        }

        TElemType* GetBuffer()
        {
            return m_buffer.data();
        }

        ~DenseSequenceWithBuffer()
        {
            // Giving the memory back.
            m_memBuffers.push(std::move(m_buffer));
        }

    private:
        std::vector<TElemType> m_buffer;
        conc_stack<std::vector<TElemType>>& m_memBuffers;
        DISABLE_COPY_AND_MOVE(DenseSequenceWithBuffer);
    };

} } }