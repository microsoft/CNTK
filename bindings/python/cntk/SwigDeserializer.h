//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Swig wrapper for deserializers, the file should be used only from cntk_py.i.
//

#pragma once

#include <memory>

namespace CNTK
{
    // A scope guard class that makes sure that the current thread
    // has properly acquired GIL.
    class GilStateGuard final
    {
        PyGILState_STATE m_state;

    public:
        GilStateGuard() : m_state(PyGILState_Ensure())
        {}

        ~GilStateGuard()
        {
            PyGILState_Release(m_state);
        }

    private:
        GilStateGuard(const GilStateGuard&) = delete; GilStateGuard& operator=(const GilStateGuard&) = delete;
        GilStateGuard& operator=(GilStateGuard&&) = delete; GilStateGuard(GilStateGuard&& other) = delete;
    };

    typedef std::shared_ptr<PyObject> PyObjectPtr;

    // Wraps a python object pointer into a shared pointer
    // with thread safe destructor.
    inline PyObjectPtr MakeShared(PyObject* object, bool increaseRefCount = true)
    {
        if (increaseRefCount)
            Py_XINCREF(object);

        return  PyObjectPtr(object, [](PyObject* p)
        {
            // The destructor can potentially be called on another thread (prefetch, i.e.
            // for sequence and chunk destructor).
            // We should make sure that the state of the thread is properly initialized
            // and GIL is aquired before using any Python API.
            GilStateGuard guard;

            Py_XDECREF(p);
        });
    }

    // Dense sequence that references some memory from a Python object.
    // Makes sure the Python object is not released while the sequence exists.
    struct DenseDataFromPy final : public DenseSequenceData
    {
        DenseDataFromPy(void* ptr, unsigned int numberOfSamples, const PyObjectPtr& object)
            : DenseSequenceData(numberOfSamples), m_data(ptr), m_object(object)
        {
        }

        virtual const void* GetDataBuffer() { return m_data; }
        virtual const NDShape& GetSampleShape() { LogicError("All sequences have the same shape, please use stream.shape instead."); }

    private:
        void* m_data;
        PyObjectPtr m_object;

        DenseDataFromPy(const DenseDataFromPy&) = delete; DenseDataFromPy& operator=(const DenseDataFromPy&) = delete;
        DenseDataFromPy& operator=(DenseDataFromPy&&) = delete; DenseDataFromPy(DenseDataFromPy&& other) = delete;
    };

    // Sparse sequence that references some memory from a Python object.
    // Makes sure the Python object is not released while the sequence exists.
    struct SparseDataFromPy final : public SparseSequenceData
    {
        SparseDataFromPy(void* data, SparseIndexType* indices, SparseIndexType nonZeroCount, unsigned int numberOfSamples, const PyObjectPtr& object)
            : m_data(data), m_object(object), SparseSequenceData(numberOfSamples)
        {
            m_indices = indices;
            m_totalNnzCount = nonZeroCount;
        }

        virtual const void* GetDataBuffer() { return m_data; }
        virtual const NDShape& GetSampleShape() { LogicError("All sequences have the same shape, please use stream.shape instead."); }

    private:
        void* m_data;
        PyObjectPtr m_object;

        SparseDataFromPy(const SparseDataFromPy&) = delete; SparseDataFromPy& operator=(const SparseDataFromPy&) = delete;
        SparseDataFromPy& operator=(SparseDataFromPy&&) = delete; SparseDataFromPy(SparseDataFromPy&& other) = delete;
    };

    // A wrapper around a Python chunk
    // A Python chunk is a dictionary of the following form:
    // { <stream name> -> [numpy array | csr_matrix | list of numpy arrays | list of csr_matrices] }
    class SwigChunk final : public Chunk
    {
        std::vector<StreamInformation> m_streamInfos;

        // A helper function that creates a dense sequence data from a Python array.
        SequenceDataPtr FromNumPy(PyArrayObject* array, const StreamInformation& info)
        {
            int rank = PyArray_NDIM(array);
            npy_intp* np_shape = PyArray_SHAPE(array);

            // In case rank is the same as in sample layout
            // the sequence length is 1, othewise we take it from the first dimension.
            uint32_t numSamples = info.m_sampleLayout.Rank() == rank ? 1 : static_cast<uint32_t>(np_shape[0]);
            auto type = PyArray_TYPE(array);
            if (type != NPY_FLOAT32)
                RuntimeError("Only array of type float is currently supported.");

            return std::make_shared<DenseDataFromPy>(PyArray_DATA(array), numSamples, MakeShared((PyObject*)array));
        }

        SequenceDataPtr FromCSR(PyObject* object, const StreamInformation& info)
        {
            PyObjectPtr data = GetProperty(object, "data");
            PyArrayObject* dataRaw = (PyArrayObject*)data.get();
            auto type = PyArray_TYPE(dataRaw);
            if (type != NPY_FLOAT32)
                RuntimeError("Only csr_matrix of float is currently supported.");

            PyObjectPtr indptr = GetProperty(object, "indptr");
            PyArrayObject* indptrRaw = (PyArrayObject*)indptr.get();
            PyObjectPtr indices = GetProperty(object, "indices");
            PyArrayObject* indicesRaw = (PyArrayObject*)indices.get();

            PyObjectPtr shape = GetProperty(object, "shape");
            auto numElements = PyTuple_GET_ITEM(shape.get(), 0);

            auto result = std::make_shared<SparseDataFromPy>(
                PyArray_DATA(dataRaw),
                (SparseIndexType*)PyArray_DATA(indicesRaw),
                static_cast<SparseIndexType>(PyArray_SIZE(dataRaw)),
                static_cast<uint32_t>(PyLong_AsLong(numElements)),
                MakeShared(object));

            // Checking the type
            type = PyArray_TYPE(indptrRaw);
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

            // Filling in nnzCount
            auto nnzCountsSize = PyArray_SIZE(indptrRaw);
            result->m_nnzCounts.resize(nnzCountsSize);
            memcpy(&result->m_nnzCounts[0], PyArray_DATA(indptrRaw), nnzCountsSize * elementSize);
            for (size_t i = 0; i < result->m_nnzCounts.size() - 1; ++i)
                result->m_nnzCounts[i] = result->m_nnzCounts[i + 1] - result->m_nnzCounts[i];
            result->m_nnzCounts.resize(result->m_nnzCounts.size() - 1);
            return result;
        }

        static std::wstring ToWString(PyObject* object)
        {
            if (object == nullptr)
                InvalidArgument("Null cannot be converted to string.");

            if (PyUnicode_Check(object))
                return std::wstring((wchar_t*)PyUnicode_AS_UNICODE(object), PyUnicode_GET_SIZE(object));

            if (PyString_Check(object)) // Non unicode string.
            {
                std::string tmp(PyString_AsString(object));
                return std::wstring(tmp.begin(), tmp.end());
            }

            RuntimeError("Expected a string, '%s' was provided.", object->ob_type->tp_name);
            return std::wstring(); // make compiler happy.
        }

    public:
        SwigChunk(size_t chunkId, const std::vector<StreamInformation>& streamInfos, PyObject* chunk)
            : m_streamInfos(streamInfos), m_chunkId(chunkId)
        {
            m_pyChunk = MakeShared(chunk);

            // Chunks are dictionaries "stream name" -> <numpy|csr_matrix|list of sequences>.
            if (!PyDict_Check(m_pyChunk.get()))
                RuntimeError("The chunk should be a dictionary of a stream name into a list of sequences");

            PyObject* key;
            PyObject* value;
            Py_ssize_t pos = 0;

            std::vector<PyObject*> pyData;

            // Remembering the py chunk for each stream.
            pyData.resize(m_streamInfos.size(), nullptr);
            
            while (PyDict_Next(m_pyChunk.get(), &pos, &key, &value))
            {
                auto name = ToWString(key);
                auto it = std::find_if(m_streamInfos.begin(), m_streamInfos.end(),
                    [name](const StreamInformation& s) { return s.m_name == name; });

                if (it == m_streamInfos.end())
                    RuntimeError("Stream with name '%ls' does not exist", name.c_str());

                pyData[it - m_streamInfos.begin()] = value;
            }

            // Let's check that the size of the chunk across all streams
            // is the same.
            size_t size = GetFirstDimension(pyData.front());
            for (size_t i = 1; i < m_streamInfos.size(); i++)
            {
                auto currentSize = GetFirstDimension(pyData[i]);
                if (size != currentSize)
                    RuntimeError("Please provide an equal number of sequences across all streams in the chunk"
                        ", currently stream '%ls' has %d sequences, whereas '%ls'- %d",
                        m_streamInfos.front().m_name.c_str(), (int)size,
                        m_streamInfos[i].m_name.c_str(), (int)currentSize);
            }

            // Now fill in the data for all streams.
            // The data for sequence I is at position I * <number of streams>.
            m_data.resize(size * m_streamInfos.size());
            for (size_t i = 0; i < m_streamInfos.size(); i++)
                FillChunkData(i, pyData[i], size);
        }

        // Helper function that returns the first dimension of the object (list, numpy or csr_matrix).
        size_t GetFirstDimension(PyObject* o)
        {
            if (PyList_Check(o))
                return PyList_Size(o);
            else if (PyArray_Check(o))
            {
                PyArrayObject* array = (PyArrayObject*)o;
                npy_intp* np_shape = PyArray_SHAPE(array);
                return np_shape[0];
            }
            // TODO: profile, probably need to have some form of
            // vtable in here, same goes for other places where we use string comparisons.
            else if (o->ob_type->tp_name == std::string("csr_matrix"))
            {
                auto shape = GetProperty(o, "shape");
                return static_cast<uint32_t>(PyLong_AsLong(PyTuple_GET_ITEM(shape.get(), 0)));
            }
            else
                RuntimeError("Unexpected type %s, only list, numpy or csr_matrix are expected.", o->ob_type->tp_name);

            return 0;
        }

        // For a given stream and from the Python data, the functions fills in m_data with
        // sequences.
        void FillChunkData(size_t streamIndex, PyObject* o, size_t dataSize)
        {
            auto storageFormat = m_streamInfos[streamIndex].m_storageFormat;
            if (PyList_Check(o)) // Data is a list of sequences.
                FillDataWithSequences(streamIndex, o, dataSize);

            // Data is a numpy array of dense samples.
            else if (storageFormat == StorageFormat::Dense && PyArray_Check(o))
                FillDataWithDenseSamples(streamIndex, o, dataSize);

            // Data is a csr matrix of sparse samples.
            else if (storageFormat == StorageFormat::SparseCSC &&
                o->ob_type->tp_name == std::string("csr_matrix"))
                FillDataWithSparseSamples(streamIndex, o, dataSize);
            else
                RuntimeError("Unexpected data type '%s'. Please use numpy arrays, csr_matrix or list of those.", o->ob_type->tp_name);
        }

        // Fills chunk data with dense samples.
        void FillDataWithDenseSamples(size_t streamIndex, PyObject* o, size_t dataSize)
        {
            const auto& info = m_streamInfos[streamIndex];
            PyArrayObject* array = (PyArrayObject*)o;
            int rank = PyArray_NDIM(array);
            auto type = PyArray_TYPE(array);
            if (type != NPY_FLOAT32)
                RuntimeError("Only float numbers are currently supported.");

            if (info.m_sampleLayout.Rank() + 1 != rank)
                RuntimeError("Dense data supported only as single sample per row.");

            for (size_t i = 0; i < dataSize; ++i)
            {
                auto d = (float*)PyArray_GETPTR1(array, i);
                auto sequence = std::make_shared<DenseDataFromPy>(d, 1, m_pyChunk);
                m_data[i * m_streamInfos.size() + streamIndex] = sequence;
            }
        }

        // Fills chunk data with sparse samples.
        void FillDataWithSparseSamples(size_t streamIndex, PyObject* o, size_t dataSize)
        {
            PyObjectPtr pyData = GetProperty(o, "data");
            PyArrayObject* pyDataRaw = (PyArrayObject*)pyData.get();

            auto type = PyArray_TYPE(pyDataRaw);
            if (type != NPY_FLOAT32)
                RuntimeError("Only float numbers are currently supported.");

            auto data = (float*)PyArray_DATA(pyDataRaw);

            PyObjectPtr indices = GetProperty(o, "indices");
            SparseIndexType* indicesRaw = (SparseIndexType*)PyArray_DATA((PyArrayObject*)indices.get());
            PyObjectPtr indptr = GetProperty(o, "indptr");
            SparseIndexType* indptrRaw = (SparseIndexType*)PyArray_DATA((PyArrayObject*)indptr.get());

            for (size_t i = 0; i < dataSize; ++i)
            {
                auto sequence = std::make_shared<SparseDataFromPy>(
                    data + indptrRaw[i],
                    indicesRaw + indptrRaw[i],
                    indptrRaw[i + 1] - indptrRaw[i],
                    1,
                    m_pyChunk);

                sequence->m_nnzCounts.resize(1, indptrRaw[i + 1] - indptrRaw[i]);
                m_data[i * m_streamInfos.size() + streamIndex] = sequence;
            }
        }

        // Fills chunk data with sequences.
        void FillDataWithSequences(size_t streamIndex, PyObject* o, size_t dataSize)
        {
            const auto& info = m_streamInfos[streamIndex];
            auto storageFormat = info.m_storageFormat;
            m_sampleMode = false;
            for (size_t i = 0; i < dataSize; ++i)
            {
                PyObject* item = PyList_GetItem(o, i);
                SequenceDataPtr sequence = nullptr;
                if (storageFormat == StorageFormat::Dense)
                {
                    if (!PyArray_Check(item))
                        RuntimeError("Expecting dense data to be represented as a numpy array.");
                    sequence = FromNumPy((PyArrayObject*)item, info);
                }
                else // Sparse
                {
                    if (item->ob_type->tp_name != std::string("csr_matrix"))
                        RuntimeError("Expecting sparse data to be represented as a csr_matrix.");
                    sequence = FromCSR(item, info);
                }

                m_data[i * m_streamInfos.size() + streamIndex] = sequence;
            }
        }

        // Gets sequence infos for the chunk.
        void SequenceInfos(std::vector<SequenceInfo>& descriptions)
        {
            size_t numSequences = m_data.size() / m_streamInfos.size();
            descriptions.reserve(numSequences);

            if (m_sampleMode)
            {
                for (size_t i = 0; i < numSequences; ++i)
                    descriptions.push_back(SequenceInfo{ i, 1, (ChunkIdType)m_chunkId });
            }
            else
            {
                // Implement logic to specify mbsize based on a stream.
                const StreamInformation* pDefMbInfo = nullptr;
                for (const StreamInformation& info : m_streamInfos)
                {
                    if (info.m_definesMbSize)
                    {
                        if (pDefMbInfo == nullptr)
                            pDefMbInfo = &info;
                        else
                            RuntimeError("Only a single stream is allowed to define minibatch size, but at least two are found.");
                    }
                }
                // Scan over the data to set sampleCount for each sequence
                unsigned int sampleCount = 1;
                for (size_t i = 0, j = 0; i < m_data.size(); ++i)
                {
                    //Note that the stream streamIndex of sequence j is at m_data[j * m_streamInfos.size() + streamIndex]
                    size_t streamIndex = i % m_streamInfos.size();
                    if (pDefMbInfo == nullptr)
                        //No stream is specified to define the minibatch size, the number of samples in the sequence
                        //is defined by the stream with maximum number of samples
                        sampleCount = std::max(sampleCount, m_data[i]->m_numberOfSamples);
                    else if (pDefMbInfo == &m_streamInfos[streamIndex])
                        //A stream is specified to define the minibatch size, the number of samples in the sequence
                        //is defined by this stream
                        sampleCount = m_data[i]->m_numberOfSamples;

                    // Last stream of the sequence, remember the max sample count as the sequence sample count.
                    if (streamIndex == m_streamInfos.size() - 1)
                    {
                        descriptions.push_back(SequenceInfo{ j++, sampleCount, (ChunkIdType)m_chunkId });
                        sampleCount = 1;
                    }
                }
            }
        }

        // Get sequence data for a given sequence index across all streams.
        void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
        {
            auto offset = m_data.data() + sequenceIndex * m_streamInfos.size();
            result.insert(result.end(), offset, offset + m_streamInfos.size());
        }

        // Get property of python object by name.
        PyObjectPtr GetProperty(PyObject* object, const std::string& propertyName)
        {
            // TODO: profile, probably need to have some form of
            // vtable in here, same goes for other places where we use string comparisons.
            auto result = PyObject_GetAttrString(object, propertyName.c_str());
            
            if (!result)
                RuntimeError("PyObject does not have property '%s'.", propertyName.c_str());

            // PyObject_GetAttrString() increases the refcount internally, so when we wrap the pointer in shared_ptr, 
            // we pass false to not increase refcount again. During delete, the refcount is decreased normally in shared_ptr's deleter.
            return MakeShared(result, false);
        }

    private:
        size_t m_chunkId;                       // Chunk id
        bool m_sampleMode = true;               // True, if the data is in sample form.
        std::shared_ptr<PyObject> m_pyChunk;    // Python chunk data.
        std::vector<SequenceDataPtr> m_data;    // Sequence data for each sequence in chunk.
    };

    // Swig deserializer is used to expose user defined deserializers
    // to Python.
    class SwigDataDeserializer : public DataDeserializer
    {
        std::vector<StreamInformation> m_streamInfos;
        std::once_flag m_streamInfosInitFlag;

        std::vector<ChunkInfo> m_chunkInfos;
        std::once_flag m_chunkInfosInitFlag;

    public:
        SwigDataDeserializer() { }

        // Interface implemented in Python.
        virtual void _GetStreamInfos(std::vector<StreamInformation>&) { NOT_IMPLEMENTED; }
        virtual void _GetChunkInfos(std::vector<ChunkInfo>&) { NOT_IMPLEMENTED; }
        virtual PyObject* _GetChunk(ChunkIdType chunkId) { NOT_IMPLEMENTED; return nullptr;  }

        // Simple python redirectors.
        std::vector<StreamInformation> StreamInfos() override
        {
            std::call_once(m_streamInfosInitFlag, [this]() {
                _GetStreamInfos(m_streamInfos);
            });

            return m_streamInfos;
        }

        std::vector<ChunkInfo> ChunkInfos() override
        {
            std::call_once(m_chunkInfosInitFlag, [this]() {
                _GetChunkInfos(m_chunkInfos);
            });

            return m_chunkInfos;
        }

        ChunkPtr GetChunk(ChunkIdType chunkId)
        {
            auto chunk = _GetChunk(chunkId);

            GilStateGuard guard;
            return std::make_shared<SwigChunk>(chunkId, m_streamInfos, chunk);
        }

        bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& description) override
        {
            NOT_IMPLEMENTED;
            return false;
        }

        void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& descriptions) override
        {
            NOT_IMPLEMENTED;
        }
    };
}