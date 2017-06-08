//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Swig specific utility classes, the file should be used only from cntk_py.i
//

#pragma once

#include <memory>

namespace CNTK
{
    typedef std::shared_ptr<PyObject> SharedPyObjectPtr;

    class SwigChunk final : public CNTK::Chunk
    {
        std::vector<CNTK::StreamInformation> m_streamInfos;

        struct SwigDensePtrData final : public CNTK::DenseSequenceData
        {
            SwigDensePtrData(void* ptr, const SharedPyObjectPtr& chunk) : m_data(ptr), m_chunk(chunk)
            {
                m_data = ptr;
            }
            virtual const void* GetDataBuffer() { return m_data; }
            virtual const CNTK::NDShape& GetSampleShape() { RuntimeError("Sample shape should be specified on the stream.");}

        private:
            void* m_data;
            SharedPyObjectPtr m_chunk;

            SwigDensePtrData(const SwigDensePtrData&) = delete; SwigDensePtrData& operator=(const SwigDensePtrData&) = delete;
            SwigDensePtrData& operator=(SwigDensePtrData&&) = delete; SwigDensePtrData(SwigDensePtrData&& other) = delete;
        };

        struct SwigSparsePtrData final : public CNTK::SparseSequenceData
        {
            void* m_data;
            SharedPyObjectPtr m_chunk;

            SwigSparsePtrData(void* data, SparseIndexType* indices, SparseIndexType v, const SharedPyObjectPtr& chunk)
                : m_data(data), m_chunk(chunk)
            {
                m_indices = indices;
                m_nnzCounts.resize(1, v);
                m_totalNnzCount = v;
            }

            virtual const void* GetDataBuffer() { return m_data; }
            virtual const CNTK::NDShape& GetSampleShape() { RuntimeError("Sample shape should be specified on the stream."); }

        };

        struct SwigDenseData final : public CNTK::DenseSequenceData
        {
            SwigDenseData(PyArrayObject* object) : m_object(object)
            {
                Py_INCREF(m_object);
            }

            ~SwigDenseData()
            {
                GilStateGuard guard;
                Py_DECREF(m_object);
            };

            virtual const void* GetDataBuffer()
            {
                GilStateGuard guard;
                return PyArray_DATA(m_object);
            }

            virtual const CNTK::NDShape& GetSampleShape()
            {
                RuntimeError("Sample shape should be specified on the stream.");
            }

        private:
            PyArrayObject* m_object;

            SwigDenseData(const SwigDenseData&) = delete; SwigDenseData& operator=(const SwigDenseData&) = delete;
            SwigDenseData& operator=(SwigDenseData&&) = delete; SwigDenseData(SwigDenseData&& other) = delete;
        };

        struct SwigSparseData final : public CNTK::SparseSequenceData
        {
            SwigSparseData(PyObject* object, PyArrayObject* data, PyArrayObject* indices, PyArrayObject* indptr)
                : m_object(object), m_pyData(data), m_pyIndptr(indptr), m_pyIndices(indices)
            {
                Py_INCREF(m_object);
                Py_INCREF(m_pyData);
                Py_INCREF(m_pyIndptr);
                Py_INCREF(m_pyIndices);

                m_indices = (SparseIndexType*)PyArray_DATA(m_pyIndices);
                m_totalNnzCount = static_cast<SparseIndexType>(PyArray_SIZE(m_pyData));
                auto nnzCountsSize = PyArray_SIZE(m_pyIndptr);
                m_nnzCounts.resize(nnzCountsSize);
                auto type = PyArray_TYPE(m_pyIndptr);
                auto indPtr = PyArray_DATA(m_pyIndptr);

                size_t elementSize = 0;
                switch (type)
                {
                case NPY_LONG:
                    elementSize = NPY_SIZEOF_LONG;
                    break;
                case NPY_INT:
                    elementSize = NPY_SIZEOF_INT;
                    break;
                default:
                    RuntimeError("Unsupported index type '%d'", type);
                }

                if (elementSize != sizeof(SparseIndexType))
                    RuntimeError("Number of bits for index is unsupported for type '%d'", type);

                memcpy(&m_nnzCounts[0], indPtr, nnzCountsSize * elementSize);
                for (size_t i = 0; i < m_nnzCounts.size() - 1; ++i)
                    m_nnzCounts[i] = m_nnzCounts[i + 1] - m_nnzCounts[i];
                m_nnzCounts.resize(m_nnzCounts.size() - 1);
            }

            virtual ~SwigSparseData()
            {
                GilStateGuard guard;
                Py_DECREF(m_object);
                Py_DECREF(m_pyData);
                Py_DECREF(m_pyIndptr);
                Py_DECREF(m_pyIndices);
            };

            virtual const void* GetDataBuffer()
            {
                GilStateGuard guard;
                return PyArray_DATA(m_pyData);
            }

            virtual const CNTK::NDShape& GetSampleShape()
            {
                RuntimeError("Sample shape should be specified on the stream.");
            }

        private:
            PyObject* m_object;
            PyArrayObject* m_pyData;
            PyArrayObject* m_pyIndptr;
            PyArrayObject* m_pyIndices;
        };

        SequenceDataPtr FromNumPy(PyObject* object, const StreamInformation& info)
        {
            PyArrayObject* array = (PyArrayObject*)object;
            int rank = PyArray_NDIM(array);
            npy_intp* np_shape = PyArray_SHAPE(array);

            uint32_t numSamples = info.m_sampleLayout.Rank() == rank ? 1 : static_cast<uint32_t>(np_shape[0]);
            auto type = PyArray_TYPE(array);
            if (type != NPY_FLOAT32)
                RuntimeError("Only float numbers are currently supported.");

            SequenceDataPtr result = std::make_shared<SwigDenseData>(array);
            result->m_numberOfSamples = numSamples;
            return result;
        }

        SequenceDataPtr FromCSR(PyObject* object, const StreamInformation& info)
        {
            auto data = (PyArrayObject*)PyObject_GetAttrString(object, "data");
            auto type = PyArray_TYPE(data);
            if (type != NPY_FLOAT32)
                RuntimeError("Only float numbers are currently supported.");

            auto indptr = (PyArrayObject*)PyObject_GetAttrString(object, "indptr");
            auto indices = (PyArrayObject*)PyObject_GetAttrString(object, "indices");

            auto shape = PyObject_GetAttrString(object, "shape");
            auto numElements = PyTuple_GET_ITEM(shape, 0);

            SequenceDataPtr result = std::make_shared<SwigSparseData>(object, data, indices, indptr);
            result->m_numberOfSamples = static_cast<uint32_t>(PyLong_AsSize_t(numElements));
            return result;
        }

        std::shared_ptr<PyObject> m_pyChunk;
        std::vector<size_t> m_indices;
        std::vector<PyObject*> m_pyData;

        std::vector<SequenceDataPtr> m_data;

        size_t m_chunkId;
    public:
        SwigChunk(size_t chunkId, const std::vector<CNTK::StreamInformation>& streamInfos, PyObject* chunk)
            : m_streamInfos(streamInfos), m_chunkId(chunkId)
        {
            Py_XINCREF(chunk);
            m_pyChunk = std::shared_ptr<PyObject>(chunk, [](PyObject* p)
            {
                // Only the last guy will aquire the lock and release the chunk data.
                GilStateGuard guard;
                Py_XDECREF(p);
            });

            // Chunks are dictionaries "name" -> numpy or list of sequences for this stream.
            if (PyDict_Check(m_pyChunk.get()))
            {
                PyObject *py_key, *py_value;
                Py_ssize_t pos = 0;
                m_pyData.resize(m_streamInfos.size(), nullptr);
                while (PyDict_Next(m_pyChunk.get(), &pos, &py_key, &py_value))
                {
                    if (!PyUnicode_Check(py_key))
                        RuntimeError("Dictionary should contain stream names");

                    size_t len = PyUnicode_GET_SIZE(py_key);
                    std::wstring name(PyUnicode_AS_UNICODE(py_key), len);

                    auto it = std::find_if(m_streamInfos.begin(), m_streamInfos.end(),
                        [name](const StreamInformation& s) { return s.m_name == name; });
                    if (it == m_streamInfos.end())
                        RuntimeError("Stream with name '%ls' does not exist", name.c_str());

                    m_pyData[it - m_streamInfos.begin()] = py_value;
                }

                size_t size = GetSize(m_pyData.front());
                for (size_t i = 1; i < m_streamInfos.size(); i++)
                {
                    auto currentSize = GetSize(m_pyData[i]);
                    if (size != currentSize)
                        RuntimeError("Provide an equal number of sequences across all streams in chunk, currently "
                            "'%ls'- %d sequences, '%ls'- %d sequences", m_streamInfos.front().m_name.c_str(), (int)size,
                            m_streamInfos[i].m_name.c_str(), (int)currentSize);
                }

                m_data.resize(size * m_streamInfos.size());
                for (size_t i = 0; i < m_streamInfos.size(); i++)
                    FillDataFrom(i, m_pyData[i], size);
            }
            else
                RuntimeError("The chunk should either be a dictionary stream name -> list of sequences. ");
        }

        size_t GetSize(PyObject* o)
        {
            if (PyList_Check(o))
                return PyList_Size(o);
            else if (PyArray_Check(o))
            {
                PyArrayObject* array = (PyArrayObject*)o;
                npy_intp* np_shape = PyArray_SHAPE(array);
                return np_shape[0];
            }
            else if (o->ob_type->tp_name == std::string("csr_matrix"))
            {
                auto shape = PyObject_GetAttrString(o, "shape");
                return static_cast<uint32_t>(PyLong_AsSize_t(PyTuple_GET_ITEM(shape, 0)));
            }
            else
                RuntimeError("Unexpected type");
            return 0;
        }

        void FillDataFrom(size_t streamIndex, PyObject* o, size_t dataSize)
        {
            const auto& info = m_streamInfos[streamIndex];
            auto storageFormat = info.m_storageFormat;
            SequenceDataPtr sequence;
            if (PyList_Check(o))
            {
                m_sampleMode = false;
                for (size_t i = 0; i < dataSize; ++i)
                {
                    PyObject* item = PyList_GetItem(o, i);
                    if (storageFormat == StorageFormat::Dense)
                    {
                        if(!PyArray_Check(item))
                            RuntimeError("Expect dense data to be represented as numpy array.");
                        sequence = FromNumPy(item, info);
                    }
                    else // Sparse
                    {
                        if (item->ob_type->tp_name != std::string("csr_matrix"))
                            RuntimeError("Expect sparsa data to be represented as csr_matrix.");
                        sequence = FromCSR(item, info);
                    }
                    m_data[i * m_streamInfos.size() + streamIndex] = sequence;
                }
            }
            else if (storageFormat == StorageFormat::Dense && PyArray_Check(o))
            {
                PyArrayObject* array = (PyArrayObject*)o;
                int rank = PyArray_NDIM(array);
                npy_intp* np_shape = PyArray_SHAPE(array);
                auto type = PyArray_TYPE(array);
                if (type != NPY_FLOAT32)
                    RuntimeError("Only float numbers are currently supported.");

                if(info.m_sampleLayout.Rank() == rank + 1)
                    RuntimeError("Dense data supported only as single sample per row.");

                size_t sampleSize = info.m_sampleLayout.TotalSize();
                for (size_t i = 0; i < dataSize; ++i)
                {
                    auto d = (float*)PyArray_GETPTR1(array, i);
                    sequence = std::make_shared<SwigDensePtrData>(d, m_pyChunk);
                    sequence->m_numberOfSamples = 1;
                    m_data[i * m_streamInfos.size() + streamIndex] = sequence;
                }
            }
            else if (storageFormat == StorageFormat::SparseCSC && o->ob_type->tp_name == std::string("csr_matrix"))
            {
                auto pyData = (PyArrayObject*)PyObject_GetAttrString(o, "data");
                auto type = PyArray_TYPE(pyData);
                if (type != NPY_FLOAT32)
                    RuntimeError("Only float numbers are currently supported.");

                auto data = (float*)PyArray_DATA(pyData);
                auto indices = (SparseIndexType*)PyArray_DATA((PyArrayObject*)PyObject_GetAttrString(o, "indices"));
                auto indptr = (SparseIndexType*)PyArray_DATA((PyArrayObject*)PyObject_GetAttrString(o, "indptr"));

                for (size_t i = 0; i < dataSize; ++i)
                {
                    sequence = std::make_shared<SwigSparsePtrData>(data + indptr[i], indices + indptr[i], indptr[i + 1] - indptr[i], m_pyChunk);
                    sequence->m_numberOfSamples = 1;
                    m_data[i * m_streamInfos.size() + streamIndex] = sequence;
                }
            }
            else
                RuntimeError("Unexpected data type '%s'. Please use numpy arrays, csr_matrix or list of those.", o->ob_type->tp_name);
        }

        void GetSequencesInfo(std::vector<CNTK::SequenceDescription>& descriptions)
        {
            size_t numElements = m_data.size() / m_streamInfos.size();
            descriptions.reserve(numElements);
            if (m_sampleMode)
            {
                for (size_t i = 0; i < numElements; ++i)
                    descriptions.push_back(SequenceDescription{i, 1, (ChunkIdType)m_chunkId});
            }
            else
            {
                unsigned int sampleCount = 1;
                for (size_t i = 0, j = 0; i < m_data.size(); ++i)
                {
                    sampleCount = std::max(sampleCount, m_data[i]->m_numberOfSamples);
                    if (i % m_streamInfos.size() == 0)
                    {
                        descriptions.push_back(SequenceDescription{ j++, sampleCount, (ChunkIdType)m_chunkId });
                        sampleCount = 1;
                    }
                }
            }
        }

        void GetSequence(size_t sequenceIndex, std::vector<CNTK::SequenceDataPtr>& result) override
        {
            auto offset = m_data.data() + sequenceIndex * m_streamInfos.size();
            result.insert(result.end(), offset, offset + m_streamInfos.size());
        }

        virtual void _GetSequence(size_t index, PyObject*) { NOT_IMPLEMENTED; }

        bool m_sampleMode = true;
    };

    typedef std::shared_ptr<SwigChunk> SwigChunkPtr;

    // Swig deserializer is used to expose user defined deserializers
    // to Python.
    class SwigDataDeserializer final : public CNTK::DataDeserializer
    {
        mutable std::vector<StreamInformation> m_streamInfos;
        mutable std::once_flag m_streamInfosInitFlag;

        mutable ChunkDescriptions m_chunkInfos;
        mutable std::once_flag m_chunkInfosInitFlag;

    public:
        SwigDataDeserializer() { }

        // Interface implemented in Python.
        virtual void _GetStreamInfos(std::vector<CNTK::StreamInformation>&) { NOT_IMPLEMENTED; }
        virtual void _GetChunkInfos(std::vector<CNTK::ChunkDescription>&) { NOT_IMPLEMENTED; }
        virtual PyObject* _GetChunk(ChunkIdType chunkId) { NOT_IMPLEMENTED; }

        // Simple 2Py redirectors.
        std::vector<StreamInformation> GetStreamDescriptions() override
        {
            std::call_once(m_streamInfosInitFlag, [this]() {
                GilStateGuard guard;
                _GetStreamInfos(m_streamInfos);
            });
            return m_streamInfos;
        }

        ChunkDescriptions GetChunkDescriptions() override
        {
            std::call_once(m_chunkInfosInitFlag, [this]() {
                GilStateGuard guard;
                _GetChunkInfos(m_chunkInfos);
            });
            return m_chunkInfos;
        }

        void GetSequencesForChunk(ChunkIdType chunkId, std::vector<CNTK::SequenceDescription>& descriptions) override
        {
            if (chunkId != m_lastChunkId)
                LogicError("Unexpected chunk %d", chunkId);

            m_lastChunk->GetSequencesInfo(descriptions);
            m_lastChunk = nullptr;
            m_lastChunkId = (size_t)0;
        }

        ChunkPtr GetChunk(ChunkIdType chunkId)
        {
            m_lastChunkId = chunkId;
            GilStateGuard guard;
            m_lastChunk = std::make_shared<SwigChunk>(chunkId, m_streamInfos, _GetChunk(chunkId));
            return m_lastChunk;
        }

        bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription& description) override
        {
            NOT_IMPLEMENTED;
        }

        SwigChunkPtr m_lastChunk;
        size_t m_lastChunkId;
    };
}