//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains helper classes for exposing sequence data in deserializers.
//

#pragma once

#include "DataDeserializer.h"
#include "ConcStack.h"

namespace CNTK {

    // Class represents a sparse sequence for category data.
    // m_data is a non-owning pointer to some staticlly allocated category.
    // TOOD: Possibly introduce typed data here.
    struct CategorySequenceData : SparseSequenceData
    {
        CategorySequenceData(const NDShape& sampleShape) : m_sampleShape(sampleShape)
        {}

        const void* GetDataBuffer() override
        {
            return m_data;
        }

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        // Non-owning pointer to the static data describing the label.
        void *m_data;

        // Non-owning reference on the sample shape.
        const NDShape& m_sampleShape;
    };

    typedef std::shared_ptr<CategorySequenceData> CategorySequenceDataPtr;

    // The class represents a sequence that returns the internal data buffer
    // back to the stack when destroyed.
    template<class TElemType>
    struct DenseSequenceWithBuffer : DenseSequenceData
    {
        DenseSequenceWithBuffer(Microsoft::MSR::CNTK::conc_stack<std::vector<TElemType>>& memBuffers, size_t numberOfElements, const NDShape& sampleShape)
            : m_memBuffers(memBuffers), m_sampleShape(sampleShape)
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

        const NDShape& GetSampleShape() override
        {
            return m_sampleShape;
        }

        ~DenseSequenceWithBuffer()
        {
            // Giving the memory back.
            m_memBuffers.push(std::move(m_buffer));
        }

    private:
        NDShape m_sampleShape;
        std::vector<TElemType> m_buffer;
        Microsoft::MSR::CNTK::conc_stack<std::vector<TElemType>>& m_memBuffers;
        DISABLE_COPY_AND_MOVE(DenseSequenceWithBuffer);
    };

    class InvalidSequenceData : public SequenceDataBase
    {
    public:
        static SequenceDataPtr Instance()
        {
            static SequenceDataPtr invalid = std::make_shared<InvalidSequenceData>();
            return invalid;
        }

        InvalidSequenceData() : SequenceDataBase(0, false) {}
        virtual ~InvalidSequenceData() {}

        virtual const NDShape& GetSampleShape() { NOT_IMPLEMENTED; }
        virtual const void* GetDataBuffer() { NOT_IMPLEMENTED; }
    };

}