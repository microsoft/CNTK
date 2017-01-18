//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CSparseDensePCReader.h - Include file for the Compressed Sparse Parallel Corpus reader.
//
#pragma once
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "RandomOrdering.h"
#include <string>
#include <map>
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>

// Windows or Posix? Originally the reader was done only for Windows. Keep it this way for now when running on Windows.
//#ifdef __WINDOWS__
//#define SPARSE_PCREADER_USE_WINDOWS_API
//#endif

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template <typename T>
            class DenseSparseBlockingQueue
            {
            private:
                std::mutex              d_mutex;
                std::condition_variable d_condition;
                std::deque<T>           d_queue;
            public:
                void push(T const& value) {
                    {
                        std::unique_lock<std::mutex> lock(this->d_mutex);
                        d_queue.push_front(value);
                    }
                    this->d_condition.notify_one();
                }
                T pop() {
                    std::unique_lock<std::mutex> lock(this->d_mutex);
                    this->d_condition.wait(lock, [=] { return !this->d_queue.empty(); });
                    T rc(std::move(this->d_queue.back()));
                    this->d_queue.pop_back();
                    return rc;
                }
            };

            template<class ElemType>
            class SparseDenseMemory
            {
            public:
                bool m_bIsSparse;

                int32_t m_sampleNumber;
                int32_t m_Dim;
                size_t m_nnZ;
                int32_t* m_rowIndices;
                int32_t* m_colIndices;
                ElemType* m_values;

                size_t m_denseIndex;

                size_t m_sparseValueIndex;
                size_t m_sparseRowIndex;
                size_t m_sparseColIndex;

                void Clear();
                void AddDenseData(void* pSource, size_t dim);
                void AddSparseData(void* nnzValue, void* nnzIndex, int32_t length);

                void FillDenseMatrix(Matrix<ElemType>* pMatrix);
                void FillSparseMatrix(Matrix<ElemType>* pMatrix);

                SparseDenseMemory()
                {
                    m_bIsSparse = true;
                    m_Dim = 0;
                    m_nnZ = 0;
                    m_rowIndices = nullptr;
                    m_colIndices = nullptr;
                    m_values = nullptr;
                    m_denseIndex = 0;
                    m_sparseValueIndex = 0;
                    m_sparseRowIndex = 0;
                    m_sparseColIndex = 0;


                }


            };



            class SparseDenseFeatureInfo
            {
            public:
                bool m_bIsSparse;
                int32_t m_Dim;


                //int32_t m_memoryStartIndex;
                //int32_t m_memoryDenseLength;

                ////Below are for Sparse Feature

                //int32_t m_nnzValue;
                //int32_t m_nnzRowIndexStartIndex;
                //int32_t m_nnzRowIndexLength;
                //int32_t m_nnzValueIndexStartIndex;
                //int32_t m_nnzValueIndexLength;

                //int32_t m_columnIndexStartIndex;
                //int32_t m_columnIndexLength;

                int32_t m_SparseFactor;

            };


            template<class ElemType>
            class CSparseDensePCReader : public DataReaderBase
            {
            private:
                ConfigParameters m_readerConfig;
                std::wstring m_file;
                //size_t m_featureCount;
                //std::vector<std::wstring> m_featureNames;
                //std::vector<size_t> m_dims;
                //std::wstring m_labelName;
                size_t m_miniBatchSize;
                size_t m_microBatchSize;
                int64_t m_maxReadData; // For early exit during debugging
                bool m_doGradientCheck;
                bool m_returnDense;
                size_t m_sparsenessFactor;
                int32_t m_verificationCode;
                size_t m_reshapeInputToRowSize;
                //   std::vector<ElemType*> m_values;
                //   std::vector<int32_t*> m_rowIndices;
                //std::vector<int32_t*> m_colIndices;
                //   ElemType* m_labelsBuffer;
                MBLayoutPtr m_pMBLayout;

                //HANDLE m_hndl;
                //HANDLE m_filemap;
                //void* m_dataBuffer;
                int64_t m_filePositionMax;
                int64_t m_currOffset;
                int m_traceLevel;

                std::map<LabelIdType, LabelType> m_mapIdToLabel;
                std::map<LabelType, LabelIdType> m_mapLabelToId;

                //Below 

                DenseSparseBlockingQueue<std::map<std::wstring, SparseDenseMemory<ElemType>*>> m_dataToProduce;
                DenseSparseBlockingQueue<std::map<std::wstring, SparseDenseMemory<ElemType>*>> m_dataToConsume;

                //This buffer is used to 
                std::map<std::wstring, void*> m_mappedBuffer;


                std::map <std::wstring, SparseDenseMemory<ElemType>*> m_mapWorkingMemory;

                std::map<std::wstring, SparseDenseFeatureInfo> m_mapSparseDenseInfo;
                bool m_bSparseDenseInfoInitialized;

                std::map<int32_t, std::wstring> m_mapReaderOrder2FeatureName;


                ifstream m_inFile;


            public:
                CSparseDensePCReader() : m_pMBLayout(make_shared<MBLayout>()) {};
                virtual ~CSparseDensePCReader();
                virtual void Destroy();
                template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
                virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
                virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }
                virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);
                virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices);

                size_t GetNumParallelSequences() { return m_pMBLayout->GetNumParallelSequences(); }
                void SetNumParallelSequences(const size_t) { };
                void CopyMBLayoutTo(MBLayoutPtr pMBLayout) { pMBLayout->CopyFrom(m_pMBLayout); }
                virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);
                virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);
                virtual bool GetData(const std::wstring& /*sectionName*/, size_t /*numRecords*/, void* /*data*/, size_t& /*dataBufferSize*/, size_t /*recordStart*/)
                {
                    RuntimeError("GetData not supported in CSparseDensePCReader");
                };
                virtual bool DataEnd();
                void ReadData();

                size_t GetNumParallelSequencesForFixingBPTTMode()
                {
                    return m_pMBLayout->GetNumParallelSequences();
                }
            };
        }
    }
}