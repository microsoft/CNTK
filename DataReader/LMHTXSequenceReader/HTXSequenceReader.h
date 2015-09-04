//
// <copyright file="SequenceReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// SequenceReader.h - Include file for the MTK and MLF format of features and samples 
#pragma once
//#define LEAKDETECT

#include "commandArgUtil.h"
#include "DataReader.h"
#include "DataWriter.h"
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include "minibatchsourcehelpers.h"
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

#define CACHE_BLOG_SIZE 50000

#define STRIDX2CLS L"idx2cls"
#define CLASSINFO  L"classinfo"

#define STRIDX2PROB L"idx2prob"
#define MAX_STRING  2048

#define DEBUG_HTX if(debughtx) //Do you want to print the debug info?

enum LMSLabelType
{
    compressed = 0,  //dim-4, for classBasedCESoftmaxNode
    onehot = 1, //dim-vocab, for crossEntropy
};

template<class ElemType>
class BatchSequenceReader : public IDataReader<ElemType>
{
public:
	int class_size;
	int nwords; //vocabSize
    map<string, int> word4idx;
    map<int, string> idx4word;
    map<int, int> idx4class;
    map<int, size_t> idx4cnt;
    string mUnk;
	int mBlgSize; //sequence Number
    bool randomize;

    size_t m_mbSize;

    int senCount; //Count of current sentence readed, used to determine whether feed data or noise
    vector<list<int>*> sequence_cache;
    string fileName; 
    ifstream fin;
    int debughtx; //used in the DEBUG_HTX macro, control the debug output
    int oneSentenceInMB; //Only allow at most one sentence in a MB, should be turned on for bi-directional training, so we will have sentence end in MB
    LMSLabelType outputLabelType; //"compressed", "onehot"
    int labelDim;

    int sentenceEndId;
    Matrix<ElemType> minibatchFlag; //A place to set minibatchFlag in the getMinibatch function

    static void ReadClassInfo(const wstring & vocfile, int& class_size,
        map<string, int>& word4idx,
        map<int, string>& idx4word,
        map<int, int>& idx4class,
        map<int, size_t> & idx4cnt,
        int nwords,
        string mUnk,
        bool flatten);
    bool refreshCacheSeq(int seq_id); //Refresh sequence_cache
    void PrintMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    void Destroy() {
    }

    ~BatchSequenceReader() {
    };
   
    void Init(const ConfigParameters& readerConfig);
    void Reset();

    /// return length of sentences size
    bool   DataEnd(EndDataType endDataType);

    void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize);
    bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    size_t NumberSlicesInEachRecurrentIter();

    void SetSentenceSegBatch(Matrix<ElemType>& sentenceBegin, vector<MinibatchPackingFlag>& minibatchPackingFlag);

};

}}}
