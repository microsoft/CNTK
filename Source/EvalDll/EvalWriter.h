//
// <copyright file="EvalWriter.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#define DATAWRITER_LOCAL
#include "DataWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Evaluation Writer class
// interface to pass to evaluation DLL
template<class ElemType>
class EvalWriter : public IDataWriter<ElemType>
{
    typedef typename IDataWriter<ElemType>::LabelType LabelType;
    typedef typename IDataWriter<ElemType>::LabelIdType LabelIdType;
private:
    std::map<std::wstring, std::vector<ElemType>*>* m_outputs; // our output data
    std::map<std::wstring, size_t>* m_dimensions; // the number of rows for the output data
    size_t m_recordCount; // count of records in this data
    size_t m_currentRecord; // next record number to read
public:
    // Method to setup the data for the reader
    void SetData(std::map<std::wstring, std::vector<ElemType>*>* outputs, std::map<std::wstring, size_t>* dimensions)
    {
        m_outputs = outputs;
        m_dimensions = dimensions;
        m_currentRecord = 0;
        m_recordCount = 0;
        for (auto iter = outputs->begin(); iter != outputs->end(); ++iter)
        {
            // figure out the dimension of the data
            const std::wstring& val = iter->first;
            size_t count = (*outputs)[val]->size();

            if (dimensions->find(val) == dimensions->end())
            {
                RuntimeError("Output %ls not found in CNTK model.", val.c_str());
            }

            size_t rows = (*dimensions)[val];
            size_t recordCount = count/rows;


            if (m_recordCount != 0)
            {
                // record count must be the same for all the data
                if (recordCount != m_recordCount)
                    RuntimeError("Record Count of %ls (%lux%lu) does not match the record count of previous entries (%lu).", val.c_str(), rows, recordCount, m_recordCount);
            }
            else
            {
                m_recordCount = recordCount;
            }
        }
    }

    virtual void Init(const ConfigParameters & /*config*/) override { }
    virtual void Init(const ScriptableObjects::IConfigRecord & /*config*/) override { }

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy()
    {
        delete this;
    }

    // EvalWriter Constructor
    // config - [in] configuration parameters for the datareader 
    template<class ConfigRecordType>
    EvalWriter(const ConfigRecordType& config)
    {
        m_recordCount = m_currentRecord = 0;
        Init(config);
    }

    // Destructor - free up the matrix values we allocated
    virtual ~EvalWriter()
    {
    }

    virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/) 
    {
        assert(false);
        NOT_IMPLEMENTED;
    }
    virtual bool SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
    {
        // loop through all the output vectors to copy the data over
        for (auto iter = m_outputs->begin(); iter != m_outputs->end(); ++iter)
        {
            // figure out the dimension of the data
            std::wstring val = iter->first;
            size_t rows = (*m_dimensions)[val];
            //size_t count = rows*numRecords;

            // find the output matrix we want to fill
            const std::map<std::wstring, void*, nocase_compare>::const_iterator iterIn = matrices.find(val);

            // allocate the matrix if we don't have one yet
            if (iterIn == matrices.end())
            {
                RuntimeError("No matrix data found for key '%ls', cannot continue", val.c_str());
            }

            Matrix<ElemType>* matrix = (Matrix<ElemType>*)iterIn->second;

            // copy over the data
            std::vector<ElemType>* data = iter->second;
            size_t index = m_currentRecord*rows;
            size_t numberToCopy = rows*numRecords;
            data->resize(index+numberToCopy);
            void* dataPtr = (void*)((ElemType*)data->data() + index);
            size_t dataSize = numberToCopy*sizeof(ElemType);
            void* mat = &(*matrix)(0,0);
            size_t matSize = matrix->GetNumElements()*sizeof(ElemType);
            memcpy_s(dataPtr, dataSize, mat, matSize);
        }

        // increment our record pointer
        m_currentRecord += numRecords;

        // return the "done with all records" value
        return (m_currentRecord >= m_recordCount);

    }
    virtual void SaveMapping(std::wstring saveId, const std::map<typename EvalWriter<ElemType>::LabelIdType, typename EvalWriter<ElemType>::LabelType>& /*labelMapping*/) {};

};

}}}
