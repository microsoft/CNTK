//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SequenceReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "SequenceReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#include "DataWriter.h"
#include "fileutil.h" // for fexists()
#include <iostream>
#include <vector>
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

// add this to all 
static void FailBecauseDeprecated(const char * fnName)
{
    // this is an old version that is neither used nor maintained, should be deleted
    bool isInUse = false;
    if (!isInUse)
        LogicError("%s: This function is deprecated and is not expected to be called.", fnName);
}
    
// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template <class ElemType>
bool SequenceReader<ElemType>::ReadRecord(size_t /*readSample*/)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    return false; // not used
}

// RecordsToRead - Determine number of records to read to populate record buffers
// mbStartSample - the starting sample from which to read
// tail - we are checking for possible remainer records to read (default false)
// returns - true if we have more to read, false if we hit the end of the dataset   --BUGBUG: Not returning a bool
template <class ElemType>
size_t SequenceReader<ElemType>::RecordsToRead(size_t mbStartSample, bool tail)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample % m_epochSize;

    // determine number left to read for this epoch
    size_t numberToEpoch = m_epochSize - epochSample;
    // we will take either a minibatch or the number left in the epoch
    size_t numberToRead = min(numberToEpoch, m_mbSize);
    if (numberToRead == 0 && !tail)
        numberToRead = m_mbSize;

    return numberToRead;
}

// GetIdFromLabel - get an Id from a Label
// mbStartSample - the starting sample we are ensureing are good
// endOfDataCheck - check if we are at the end of the dataset (no wraparound)
// returns - true if we have more to read, false if we hit the end of the dataset
template <class ElemType>
IDataReader::LabelIdType SequenceReader<ElemType>::GetIdFromLabel(const std::string& labelValue, LabelInfo& labelInfo)
{
    auto found = labelInfo.mapLabelToId.find(labelValue);
    // not yet found, add to the map
    if (found == labelInfo.mapLabelToId.end())
    {
        found = labelInfo.mapLabelToId.find(mUnk);
        if (found == labelInfo.mapLabelToId.end())
            RuntimeError("%s not in vocabulary", labelValue.c_str());
    }
    return found->second;
}

template <class ElemType>
bool SequenceReader<ElemType>::CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned/*TODO: LabelIdType?*/& labelId)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    auto found = labelInfo.mapLabelToId.find(labelValue);

    // not yet found, add to the map
    if (found == labelInfo.mapLabelToId.end())
        return false;
    labelId = found->second;
    return true;
}

// EnsureDataAvailable - Read enough lines so we can request a minibatch starting as requested
// mbStartSample - the starting sample we are starting with
// endOfDataCheck - check if we are at the end of the dataset (no wraparound)
// returns - true if we have more to read, false if we hit the end of the dataset
template <class ElemType>
bool SequenceReader<ElemType>::EnsureDataAvailable(size_t mbStartSample, bool /*endOfDataCheck*/)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    bool moreToRead = true;

    size_t numberToRead = RecordsToRead(mbStartSample);

    // check to see if we have the proper records read already
    if (m_readNextSample >= mbStartSample + numberToRead && mbStartSample >= m_epochStartSample)
        return true;

    // if we have another sequence already read and waiting, just return now
    if (m_seqIndex < m_sequence.size())
        return true;

    m_seqIndex = 0;
    m_mbStartSample = 0;
    m_sequence.clear();
    m_featureData.clear();
    m_labelIdData.clear();

    m_readNextSample = 0;
    epochSample = 0;

    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];

    bool nextWord = false;
    if (m_labelInfo[labelInfoOut].type == labelNextWord)
    {
        nextWord = true;
    }
    LabelInfo& labelInfo = m_labelInfo[nextWord ? labelInfoIn : labelInfoOut];

    // if (m_labelIdData.size() > epochSample)
    // {
    //    m_labelIdData.resize(epochSample);
    //    m_labelData.resize(epochSample*labelInfo.dim);
    // }

    // see how many we already read
    int sequencesRead = 0;
    std::vector<ElemType> featureTemp;
    std::vector<LabelType> labelTemp;
    std::vector<SequencePosition> seqPos;
    do
    {
        int numRead = m_parser.Parse(m_cacheBlockSize, &labelTemp, &featureTemp, &seqPos);
        moreToRead = (numRead != 0);

        // translate from the sparse parsed data format to the training format data
        int label = 0;
        bool bSentenceStart = false;
        SequencePosition sposLast = SequencePosition(0, 0, seqFlagNull);
        for (int seq = 0; seq < numRead; seq++)
        {
            // check
            SequencePosition spos = seqPos[seq];
            if (spos.labelPos == sposLast.labelPos && spos.numberPos == sposLast.numberPos)
                continue;
            sposLast = spos;

            bSentenceStart = true;

            // loop through the labels for this entry
            while (label < spos.labelPos) // need to minus one since
            {
                // labelIn should be a category label
                LabelType labelValue = labelTemp[label++];

                if (trim(labelValue).size() == 0)
                    continue; // empty input

                // check for end of sequence marker
                if (!bSentenceStart && (EqualCI(labelValue, m_labelInfo[labelInfoIn].endSequence) || ((label - 1) % m_mbSize == 0)))
                {
                    // ignore those cases where $</s> is put in the begining, because those are used for initialization purpose
                    spos.flags |= seqFlagStopLabel;
                    sequencesRead++;

                    // create the seqence table
                    m_sequence.push_back(epochSample);
                    if ((m_sequence.size() == 1 ? epochSample : epochSample - m_sequence[m_sequence.size() - 2]) > m_mbSize)
                    {
                        // fprintf(stderr, "read sentence length is longer than the minibatch size. should be smaller. increase the minibatch size to at least %d", epochSample);

                        std::wcerr << "read sentence length is longer than the minibatch size. should be smaller. increase the minibatch size to at least " << epochSample << endl;
                        RuntimeError("read sentence length is longer than the minibatch size. should be smaller. increase the minibatch size to at least %d", (int) epochSample);
                    }

                    if (EqualCI(labelValue, m_labelInfo[labelInfoIn].endSequence))
                        continue; // ignore sentence ending
                }

                // TODO: should ignore <s>, check the sentence ending is </s>
                // need to remove <s> from the training set
                // allocate and initialize the next chunck of featureData
                if (labelIn.type == labelCategory)
                {
                    LabelIdType index = GetIdFromLabel(labelValue, labelIn);

                    // use the found value, and set the appropriate location to a 1.0
                    assert(labelIn.dim > index); // if this goes off labelOut dimension is too small
                    m_featureData.push_back((float) index);
                }
                else
                {
                    RuntimeError("Input label expected to be a category label");  // TODO: ensure this at config time (maybe keep an assert() here as a reminder to the reader)
                }

                // if we have potential features
                if (m_featureDim > 0)
                {
                    RuntimeError("to-do. Assume sparse input feature. need to change the code from dense matrix");
                    // move the position up to the start of the additional features section
                    /*                    pos += labelIn.dim;
                    assert(pos + m_featureDim == m_featureData.size());
                    // this has to be an even number, a pair of index and value
                    if  ((spos.numberPos&1) != 0)
                        RuntimeError("Features must be specified in pairs (index:value). Invalid features for label '%s'\n", labelValue);

                    while (feature < spos.numberPos)
                    {
                        int index = (int)featureTemp[feature++];
                        if (index < 0 || index >= m_featureDim)
                            RuntimeError("Invalid feature index: %d for label '%s', feature max dimension = %lld\n", index, labelValue, m_featureDim);

                        ElemType value = featureTemp[feature++];
                        m_featureData[pos+index] = value;
                    }
                    */
                }

                // now get the output label
                if (m_labelInfo[labelInfoOut].type == labelCategory)
                {
                    labelValue = labelTemp[label++];
                }
                else if (nextWord)
                {
                    // this is the next word (label was incremented above)
                    // TODO: Why is this produced by the reader, and not just realized through the use of delay nodes in the network?
                    labelValue = labelTemp[label];
                    if (EqualCI(labelValue, m_labelInfo[labelInfoIn].endSequence))
                    {
                        labelValue = labelInfo.endSequence;
                    }
                }
                else
                {
                    RuntimeError("Invalid output label type, expected Category, or Next Word");
                }

                // get the id from the label
                LabelIdType id = GetIdFromLabel(labelValue, labelInfo);
                m_labelIdData.push_back(id);

                m_readNextSample++;
                epochSample++;
                if (!m_endReached)
                    m_totalSamples++; // add to the total number of records in the dataset

                bSentenceStart = false;
            }

            {
                // check if the reading is right
                int jEnd = (int) m_labelIdData.size() - 1;
                LabelIdType index;
                if (CheckIdFromLabel(labelInfo.endSequence, labelInfo, index) == false)
                    RuntimeError("cannot find sentence begining label");

                if (m_labelIdData[jEnd] != index)
                    // for language model, the first word/letter has to be <s>
                    RuntimeError("LMSequenceReader: The last letter/word of a batch has to be the sentence ending symbol.");
            }
        }

        m_readNextSampleLine += numRead;
    } while (sequencesRead < 1 && moreToRead); // we need to read at least one sequence or have no more data

    // if we read to the end, update appropriate variables
    if (!moreToRead)
    {
        UpdateDataVariables();
    }

    // if there more to read
    return moreToRead;
}

// UpdateDataVariables - Update variables that depend on the dataset being completely read
template <class ElemType>
void SequenceReader<ElemType>::UpdateDataVariables()
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // if we haven't been all the way through the file yet
    if (!m_endReached)
    {
        // get the size of the dataset
        assert(m_totalSamples * m_featureCount >= m_featureData.size());

        // if they want us to determine epoch size based on dataset size, do that
        if (m_epochSize == requestDataSize)
        {
            m_epochSize = m_totalSamples;
        }

        // TODO: comment what this does, in a function called UpdateDataVariables
        WriteLabelFile();

        // we got to the end of the dataset
        m_endReached = true;
    }

    // update the label dimension if it is not big enough, need it here because m_labelIdMax get's updated in the processing loop (after a read)
    for (int index = 0; index < labelInfoNum; ++index)
    {
        if (m_labelInfo[index].type == labelCategory && m_labelInfo[index].numIds > m_labelInfo[index].dim)
            m_labelInfo[index].dim = m_labelInfo[index].numIds; // update the label dimensions if different
    }
}

// TODO: move this to class File as well
template <class ElemType>
void SequenceReader<ElemType>::WriteLabelFile()
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // update the label dimension if it is not big enough, need it here because m_labelIdMax get's updated in the processing loop (after a read)
    for (int index = 0; index < labelInfoNum; ++index)
    {
        LabelInfo& labelInfo = m_labelInfo[index];

        // write out the label file if they don't have one
        if (!labelInfo.fileToWrite.empty())
        {
            if (labelInfo.mapIdToLabel.size() > 0)
            {
                File labelFile(labelInfo.fileToWrite, fileOptionsWrite | fileOptionsText);
                for (int i = 0; i < labelInfo.mapIdToLabel.size(); ++i)
                {
                    labelFile << labelInfo.mapIdToLabel[i] << '\n';
                }
                labelInfo.fileToWrite.clear();
            }
            else if (!m_cachingWriter)
            {
                // fprintf(stderr, "WARNING: file %ls NOT written to disk, label files only written when starting at epoch zero!", labelInfo.fileToWrite.c_str());
                std::wcerr << "WARNING: file " << labelInfo.fileToWrite.c_str() << " NOT written to disk, label files only written when starting at epoch zero!" << endl;
            }
        }
    }
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void SequenceReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
// Sample format below:
//# Parameter values for the reader
//reader=[
//  # reader to use
//  readerType=SequenceReader
//  randomize=None
// # additional features dimension
//  featureDim=784
//  file=c:\data\sequence\sequence.txt
//  labelIn=[
//    dim=26
//      labelMappingFile=c:\data\sequence\alphabet.txt
//      labelType=Category
//    beginSequence="<s>"
//    endSequence="</s>"
//  ]
//  labelOut=[
//    dim=129
//      labelMappingFile=c:\data\sequence\phonemes.txt
//      labelType=Category
//    beginSequence="O"
//    endSequence="O"
//  ]
//]
// Note: This is to a great deal a duplicate of BatchSequenceReader::InitFromConfig(), which has gotten some clean-up love, which we should apply here as well.
// TODO: This function seems to be never called. Remove it if that is the case.
template <class ElemType>
template <class ConfigRecordType>
void SequenceReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    // NOTE: probably want to re-enable at some point

    // initialize the cache
    // InitCache(readerConfig);
    // m_readerConfig = readerConfig;

    // // if we have a cache, no need to parse the test files...
    // if (m_cachingReader)
    //    return;

    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;
    GetFileConfigNames(readerConfig, features, labels);
    if (features.size() > 0)
    {
        m_featuresName = features[0];
    }

    if (labels.size() != labelInfoNum)
        RuntimeError("LMSequenceReader: Two label definitions (in and out) are required.");

    for (int index = 0; index < labelInfoNum; ++index)
        m_labelsName[index] = labels[index];

    const ConfigRecordType& featureConfig = readerConfig(m_featuresName.c_str(), ConfigRecordType::Record());

    m_classSize = 0;
    m_featureDim = featureConfig(L"dim");
    for (int index = 0; index < labelInfoNum; ++index)
    {
        const ConfigRecordType& labelConfig = readerConfig(m_labelsName[index].c_str(), ConfigRecordType::Record());

        // TODO: use a reference for m_labelInfo[index]
        m_labelInfo[index].numIds = 0;
        m_labelInfo[index].beginSequence = msra::strfun::utf8(labelConfig(L"beginSequence", L""));
        m_labelInfo[index].endSequence   = msra::strfun::utf8(labelConfig(L"endSequence",   L""));

        // determine label type desired
        wstring labelType(labelConfig(L"labelType", L"Category"));
        if (labelType == L"Category")
        {
            m_labelInfo[index].type = labelCategory;
        }
        else if (labelType == L"NextWord")
        {
            // in this case, it's all identical to the Input labels, except the data type
            m_labelInfo[index].type = labelNextWord;
            m_labelInfo[index].dim = m_labelInfo[labelInfoIn].dim;
        }
        else if (labelType == L"None")
        {
            m_labelInfo[index].type = labelNone;
            m_labelInfo[index].dim = 0; // override for no labels
        }

        // if we have labels, we need a label Mapping file, it will be a file with one label per line
        if (m_labelInfo[index].type != labelNone)
        {
            std::wstring wClassFile = readerConfig(L"wordclass", L"");
            nwords = labelConfig(L"labelDim");
            if (wClassFile != L"")
            {
                ReadClassInfo(wClassFile, m_classSize,
                              word4idx,
                              idx4word,
                              idx4class,
                              idx4cnt,
                              nwords,
                              mUnk, m_noiseSampler,
                              false);
            }

            std::vector<string> arrayLabels;
            std::wstring labelPath = labelConfig(L"labelMappingFile");
            if (fexists(labelPath))
            {
                File::LoadLabelFile(labelPath, arrayLabels);
                for (int i = 0; i < arrayLabels.size(); ++i)
                {
                    LabelType label = arrayLabels[i];
                    m_labelInfo[index].mapIdToLabel[i] = label;
                    m_labelInfo[index].mapLabelToId[label] = i;
                }
                m_labelInfo[index].numIds = (LabelIdType) arrayLabels.size();
                m_labelInfo[index].mapName = labelPath;
            }
            else
            {
                if (wClassFile != L"")
                {
                    ReadClassInfo(wClassFile, m_classSize,
                                  word4idx,
                                  idx4word,
                                  idx4class,
                                  idx4cnt,
                                  nwords, mUnk, m_noiseSampler,
                                  false);
                    int iMax = -1, i;
                    for (auto ptr = word4idx.begin(); ptr != word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first;
                        i = ptr->second;
                        iMax = max(i, iMax);
                        m_labelInfo[index].mapIdToLabel[i] = label;
                        m_labelInfo[index].mapLabelToId[label] = i;
                    }
                    m_labelInfo[index].numIds = (LabelIdType)(iMax + 1);
                }
                m_labelInfo[index].mapName = labelPath;

                m_labelInfo[index].fileToWrite = labelPath;
            }
        }

        m_labelInfo[index].dim = (LabelIdType)(size_t) labelConfig(L"labelDim");

        // update dimension if the file says it's bigger
        if (m_labelInfo[index].dim < m_labelInfo[index].numIds)
        {
            m_labelInfo[index].dim = m_labelInfo[index].numIds;
        }
    }

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = m_seqIndex = 0;
    m_endReached = false;
    m_readNextSampleLine = 0;
    m_readNextSample = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);
    m_parser.SetTraceLevel(m_traceLevel);

    // The input data is a combination of the label Data and extra feature dims together
    //    m_featureCount = m_featureDim + m_labelInfo[labelInfoIn].dim;
    m_featureCount = 1;

    wstring pathName = readerConfig(L"file");
    if (m_traceLevel > 0)
        fprintf(stderr, "LMSequenceReader: reading input file %ls...\n", pathName.c_str());

    // TODO: how many of these do we have? labelInfoIn, Min, Out, Max, and there must be exactly 2?
    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseInit(pathName.c_str(), m_featureDim, labelIn.dim, labelOut.dim, labelIn.beginSequence, labelIn.endSequence, labelOut.beginSequence, labelOut.endSequence);

    // unk symbol
    mUnk = readerConfig(L"unk", "<unk>");
}

#if 0
template <class ElemType>
void SequenceReader<ElemType>::ReadWord(char* word, FILE* fin)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    int a = 0, ch;

    while (!feof(fin))
    {
        ch = fgetc(fin);

        if (ch == 13)
            continue;

        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0)
            {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }

            if (ch == '\n')
            {
                strcpy_s(word, strlen("</s>"), (char*) "</s>");
                return;
            }
            else
                continue;
        }

        word[a] = (char) ch;
        a++;

        if (a >= MAX_STRING)
        {
            // printf("Too long word found!\n");   // truncate too long words
            a--;
        }
    }
    word[a] = 0;
}
#endif

template <class ElemType>
void SequenceReader<ElemType>::ReadClassInfo(const wstring& vocfile, int& classSize,
                                             map<string, int>& word4idx,
                                             map<int, string>& idx4word,
                                             map<int, int>& idx4class,
                                             map<int, size_t>& idx4cnt,
                                             int nwords, // only used for a consistency check
                                             string mUnk,
                                             noiseSampler<long>& m_noiseSampler,
                                             bool /*flatten*/)
{
    string tmp_vocfile(vocfile.begin(), vocfile.end()); // convert from wstring to string
    string strtmp;
    size_t cnt;
    int clsidx, b;
    classSize = 0;

    string line;
    vector<string> tokens;
    ifstream fin;
    fin.open(tmp_vocfile.c_str());
    if (!fin)
    {
        RuntimeError("cannot open word class file");
    }

    while (getline(fin, line))
    {
        line = trim(line);
        tokens = msra::strfun::split(line, "\t ");
        assert(tokens.size() == 4);

        b = stoi(tokens[0]);
        cnt = (size_t) stof(tokens[1]);
        strtmp = tokens[2];
        clsidx = stoi(tokens[3]);

        idx4cnt[b] = cnt;
        word4idx[strtmp] = b;
        idx4word[b] = strtmp;

        idx4class[b] = clsidx;
        classSize = max(classSize, clsidx);
    }
    fin.close();
    classSize++;

    // Note: If users specify labelDim = 0 (->nwords) this will not fail. Later we will interpret this as "infer".
    if (idx4class.size() < nwords)
        RuntimeError("ReadClassInfo: The actual number of words %d is smaller than the specified vocabulary size %d. Check if labelDim is too large. ", (int) idx4class.size(), (int) nwords);

    std::vector<double> counts(idx4cnt.size());
    for (const auto& p : idx4cnt)
        counts[p.first] = (double) p.second;
    m_noiseSampler = noiseSampler<long>(counts);

    // check if unk is the same used in vocabulary file
    if (word4idx.find(mUnk.c_str()) == word4idx.end())
        RuntimeError("ReadClassInfo unknown symbol '%s' is not in vocabulary file.", mUnk.c_str());
}

// InitCache - Initialize the caching reader if cache files exist, otherwise the writer
// readerConfig - reader configuration
template <class ElemType>
void SequenceReader<ElemType>::InitCache(const ConfigParameters& readerConfig)
{
    // check for a writer tag first (lets us know we are caching)
    if (!readerConfig.Exists(L"writerType"))
        return;

    // first try to open the binary cache
    bool found = false;
    try
    {
        // TODO: need to go down to all levels, maybe search for sectionType
        ConfigArray filesList(',');
        vector<std::wstring> names;
        if (readerConfig.Exists(L"wfile"))
        {
            filesList.push_back(readerConfig(L"wfile"));
            if (fexists(readerConfig(L"wfile")))
                found = true;
        }
        FindConfigNames(readerConfig, "wfile", names);
        for (const auto& name : names)
        {
            ConfigParameters config = readerConfig(name);
            filesList.push_back(config("wfile"));
            if (fexists(config("wfile")))
                found = true;
        }

        // if we have a file already, we are going to read the cached files
        if (found)
        {
            ConfigParameters config;
            readerConfig.CopyTo(config);
            // mmodify the config so the reader types look correct
            config["readerType"] = config("writerType");
            config["file"] = filesList;
            m_cachingReader = new DataReader(config);
        }
        else
        {
            m_cachingWriter = new DataWriter(readerConfig);

            // now get the section names for map and category types
            std::map<std::wstring, SectionType, nocase_compare> sections;
            m_cachingWriter->GetSections(sections);
            for (const auto& pair : sections)
            {
                // TODO: we would need to add a sequenceMap type here as well
                // or maybe change to heirarchal name (i.e. root.labelIn.map)
                if (pair.second == sectionTypeCategoryLabel)
                {
                    m_labelsCategoryName[labelInfoOut] = pair.first;
                }
                else if (pair.second == sectionTypeLabelMapping)
                {
                    m_labelsMapName[labelInfoOut] = pair.first;
                }
            }
        }
    }
    catch (runtime_error err)
    {
        fprintf(stderr, "Error attemping to create Binary%s\n%s\n", found ? "Reader" : "Writer", err.what());
        delete m_cachingReader;
        m_cachingReader = NULL;
        delete m_cachingWriter;
        m_cachingWriter = NULL;
    }
    catch (...)
    {
        // if there is any error, just get rid of the object
        fprintf(stderr, "Error attemping to create Binary%s\n", found ? "Reader" : "Writer");
        delete m_cachingReader;
        m_cachingReader = NULL;
        delete m_cachingWriter;
        m_cachingWriter = NULL;
    }
}

// destructor - virtual so it gets called properly
template <class ElemType>
SequenceReader<ElemType>::~SequenceReader()
{
    ReleaseMemory();
    delete m_cachingReader;
    delete m_cachingWriter;
}

// ReleaseMemory - release the memory footprint of SequenceReader
// used when the caching reader is taking over
template <class ElemType>
void SequenceReader<ElemType>::ReleaseMemory()
{
    if (m_featuresBuffer != NULL)
        delete[] m_featuresBuffer;
    m_featuresBuffer = NULL;
    if (m_labelsBuffer != NULL)
        delete[] m_labelsBuffer;
    m_labelsBuffer = NULL;
    if (m_labelsIdBuffer != NULL)
        delete[] m_labelsIdBuffer;
    m_labelsIdBuffer = NULL;
    m_featureData.clear();
    m_labelIdData.clear();
    //m_labelData.clear();
    m_sequence.clear();
}

// SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template <class ElemType>
void SequenceReader<ElemType>::SetupEpoch()
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // if we are starting fresh (epoch zero and no data read), init everything
    // however if we are using cachingWriter, we need to know record count, so do that first
    if (m_epoch == 0 && m_totalSamples == 0 && m_cachingWriter == NULL)
    {
        m_readNextSampleLine = m_readNextSample = m_epochStartSample = m_mbStartSample = m_seqIndex = 0;
        m_parser.SetFilePosition(0);    // TODO: can this ever be set to not 0?
    }
    else // otherwise, position the read to start at the right location
    {
        m_seqIndex = 0;
        // don't know the total number of samples yet, so count them
        if (m_totalSamples == 0)
        {
            if (m_traceLevel > 0)
                fprintf(stderr, "starting at epoch %zd parsing all data to determine record count\n", m_epoch);
            // choose a large number to read
            m_parser.SetFilePosition(0);
            m_mbStartSample = 0;
            while (EnsureDataAvailable(m_mbStartSample))
            {
                m_mbStartSample = m_totalSamples;
                m_seqIndex = m_sequence.size();
            }
            if (m_traceLevel > 0)
                fprintf(stderr, "\n %zd records found\n", m_totalSamples);
        }
        m_seqIndex = 0;

        // we have a slight dilemma here, if we haven't determined the end of the file yet
        // and the user told us to find how many records are in the file, we can't distinguish "almost done"
        // with a file (a character away) and the middle of the file. So read ahead a record to see if it's there.
        bool endReached = m_endReached;
        if (!endReached)
        {
            if (!m_parser.HasMoreData())
            {
                endReached = true;
                UpdateDataVariables();
                assert(m_endReached);
            }
        }

        // always start from the first sample
        m_epochStartSample = m_mbStartSample = 0;
    }
}

template <class ElemType>
void SequenceReader<ElemType>::LMSetupEpoch()
{
    m_readNextSampleLine = m_readNextSample = m_epochStartSample = m_mbStartSample = m_seqIndex = 0;
}

// utility function to round an integer up to a multiple of size
inline size_t RoundUp(size_t value, size_t size)
{
    return ((value + size - 1) / size) * size;
}

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
//     NOTE: for sequence data, this will be the MAX size of a sequence, as every sequence could be a different length
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template <class ElemType>
void SequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory(); // free the memory used by the SequenceReader
    }

    // if we are reading from the cache, do so now and return
    if (m_cachingReader)
    {
        m_cachingReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
        return;
    }

    if (m_featuresBuffer == NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];
        m_featuresBuffer = new ElemType[mbSize * labelInfo.dim];
        memset(m_featuresBuffer, 0, sizeof(ElemType) * mbSize * labelInfo.dim);
    }

    if (m_labelsBuffer == NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];
        if (labelInfo.type == labelCategory)
        {
            m_labelsBuffer = new ElemType[mbSize * labelInfo.dim];
            memset(m_labelsBuffer, 0, sizeof(ElemType) * mbSize * labelInfo.dim);
            m_labelsIdBuffer = new LabelIdType[mbSize];
            memset(m_labelsIdBuffer, 0, sizeof(LabelIdType) * mbSize);
        }
        else if (labelInfo.type != labelNone)
        {
            m_labelsBuffer = new ElemType[mbSize];
            memset(m_labelsBuffer, 0, sizeof(ElemType) * mbSize);
            m_labelsIdBuffer = NULL;
        }
    }

    m_mbSize = mbSize;
    if (requestedEpochSamples == requestDataSize)
    {
        if (!m_endReached)
        {
            m_epochSize = requestDataSize;
        }
    }
    else
    {
        m_epochSize = requestedEpochSamples;
    }

    // we use epochSize, which might not be set yet, so use a default value for allocations if not yet set
    size_t epochSize = m_epochSize == requestDataSize ? 1000 : m_epochSize;
    m_epoch = epoch;
    m_mbStartSample = epoch * m_epochSize;

    // allocate room for the data
    m_featureData.reserve(m_featureCount * epochSize);
    if (m_labelInfo[labelInfoOut].type == labelCategory)
        m_labelIdData.reserve(epochSize);
    //else if (m_labelInfo[labelInfoOut].type != labelNone)
    //    m_labelData.reserve(epochSize);
    m_sequence.reserve(m_seqIndex); // clear out the sequence array
    // this is too complicated for LM
    // SetupEpoch();
    // use the LMSetupEpoch() instead
    LMSetupEpoch();

    m_clsinfoRead = false;
    m_idx2clsRead = false;

    m_parser.ParseReset();
}

template <class ElemType>
bool SequenceReader<ElemType>::DataEnd()
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    return SentenceEnd();
}

template <class ElemType>
bool SequenceReader<ElemType>::SentenceEnd()
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // this is after getMinibatch size, which has increased m_seqIndex by 1
    // so the real index is m_seqIndex - 1;
    int seqIndex = (int) m_seqIndex - 1;

    // now get the labels
    const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];

    size_t actualmbsize = 0;

    // figure out the size of the next sequence
    if (seqIndex > 0)
    {
        actualmbsize = m_sequence[seqIndex] - m_sequence[seqIndex - 1];
    }
    else
    {
        actualmbsize = m_sequence[0];
    }

    if (actualmbsize < m_mbSize)
        return true;

    size_t jEnd = m_sequence[seqIndex] - 1;

    if (labelInfo.type == labelCategory)
    {
        LabelIdType index;
        if (CheckIdFromLabel(labelInfo.endSequence, labelInfo, index) == false)
            RuntimeError("cannot find sentence begining label");

        if (m_labelIdData[jEnd] == index)
            return true;
        else
            return false;
    }
    return false;
}

/// the output label is a [4 x T] matrix, where T is the number of words observed
/// the first row is the word index
/// the second row is the class id of this word
/// the third row is begining index of the class for this word
/// the fourth row is the ending index + 1 of the class for this word
template <class ElemType>
void SequenceReader<ElemType>::GetLabelOutput(StreamMinibatchInputs& matrices,
                                              size_t m_mbStartSample, size_t actualmbsize)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    size_t j = 0;
    if (!matrices.HasInput(m_labelsName[labelInfoOut]))
        return;

    Matrix<ElemType>& labels = matrices.GetInputMatrix<ElemType>(m_labelsName[labelInfoOut]);

    if (readerMode == ReaderMode::NCE)
        labels.Resize(2 * (m_noiseSampleSize + 1), actualmbsize);
    else if (readerMode == ReaderMode::Class)
        labels.Resize(4, actualmbsize);
    else if (readerMode == ReaderMode::Softmax)
        labels.Resize(1, actualmbsize);

    for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
    {
        // pick the right sample with randomization if desired
        size_t jRand = jSample;
        int wrd = m_labelIdData[jRand];
        labels.SetValue(0, j, (ElemType) wrd);

        if (readerMode == ReaderMode::NCE)
        {
            labels.SetValue(1, j, (ElemType) m_noiseSampler.logprob(wrd));
            for (size_t noiseid = 0; noiseid < m_noiseSampleSize; noiseid++)
            {
                int wid = m_noiseSampler.sample();
                labels.SetValue(2 * (noiseid + 1), j, (ElemType) wid);
                labels.SetValue(2 * (noiseid + 1) + 1, j, -(ElemType) m_noiseSampler.logprob(wid));
            }
        }
        else if (readerMode == ReaderMode::Class)
        {
            int clsidx = idx4class[wrd];
            if (m_classSize > 0)
            {
                labels.SetValue(1, j, (ElemType) clsidx);
                // save the [begining ending_indx) of the class
                labels.SetValue(2, j, (*m_classInfoLocal)(0, clsidx)); // begining index of the class
                labels.SetValue(3, j, (*m_classInfoLocal)(1, clsidx)); // end index of the class
            }
        }
    }
}
template <class ElemType>
void SequenceReader<ElemType>::GetInputProb(StreamMinibatchInputs& matrices)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    if (!matrices.HasInput(STRIDX2PROB))
        return;

    if (m_idx2probRead)
        return;

    Matrix<ElemType>& idx2prob = matrices.GetInputMatrix<ElemType>(STRIDX2PROB);

    // populate local CPU matrix
    m_id2Prob->SwitchToMatrixType(MatrixType::DENSE, matrixFormatDense, false);
    m_id2Prob->Resize(nwords, 1, false);

    // move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = m_id2Prob->GetDeviceId();
    m_id2Prob->TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
    for (size_t j = 0; j < nwords; j++)
        (*m_id2Prob)((int) j, 0) = (float) m_noiseSampler.prob((int) j);
    m_id2Prob->TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    int oldDeviceId = idx2prob.GetDeviceId();
    // caution, SetValue changes idx2cls from GPU to CPU, may change this behavior later
    idx2prob.SetValue(*m_id2Prob);
    idx2prob.TransferFromDeviceToDevice(idx2prob.GetDeviceId(), oldDeviceId, true);

    m_idx2probRead = true;
}

// TODO: Document what this is. It seems we can fill specific hard-coded inputs with something interesting.
template <class ElemType>
void SequenceReader<ElemType>::GetInputToClass(StreamMinibatchInputs& matrices)
{
    if (!matrices.HasInput(STRIDX2CLS))
        return;

    if (m_idx2clsRead)
        return;

    Matrix<ElemType>& idx2cls = matrices.GetInputMatrix<ElemType>(STRIDX2CLS);

    // populate local CPU matrix
    m_id2classLocal->SwitchToMatrixType(MatrixType::DENSE, matrixFormatDense, false);
    m_id2classLocal->Resize(nwords, 1, false);

    // move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = m_id2classLocal->GetDeviceId();
    m_id2classLocal->TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
    for (size_t j = 0; j < nwords; j++)
    {
        int clsidx = idx4class[(int) j];
        (*m_id2classLocal)(j, 0) = (float) clsidx;
    }
    m_id2classLocal->TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    int oldDeviceId = idx2cls.GetDeviceId();
    // caution, SetValue changes idx2cls from GPU to CPU, may change this behavior later
    idx2cls.SetValue(*m_id2classLocal);
    idx2cls.TransferFromDeviceToDevice(idx2cls.GetDeviceId(), oldDeviceId, true);

    m_idx2clsRead = true;
}

template <class ElemType>
void SequenceReader<ElemType>::GetClassInfo()
{
    if (m_clsinfoRead)
        return;

    // populate local CPU matrix
    m_classInfoLocal->SwitchToMatrixType(MatrixType::DENSE, matrixFormatDense, false);
    m_classInfoLocal->Resize(2, m_classSize);

    // move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = m_classInfoLocal->GetDeviceId();
    m_classInfoLocal->TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);

    int clsidx;
    int prvcls = -1;
    for (size_t j = 0; j < nwords; j++)
    {
        clsidx = idx4class[(int) j];
        if (prvcls != clsidx && clsidx > prvcls)
        {
            if (prvcls >= 0)
                (*m_classInfoLocal)(1, prvcls) = (float) j;
            prvcls = clsidx;
            (*m_classInfoLocal)(0, prvcls) = (float) j;
        }
        else if (prvcls > clsidx)
        {
            // nwords is larger than the actual number of words
            LogicError("LMSequenceReader::GetClassInfo probably the number of words specified is larger than the actual number of words. Check network builder and data reader. ");
        }
    }
    (*m_classInfoLocal)(1, prvcls) = (float) nwords;

    //    (*m_classInfoLocal).Print();

    m_classInfoLocal->TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    m_clsinfoRead = true;
}

template <class ElemType>
bool SequenceReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    // get out if they didn't call StartMinibatchLoop() first
    if (m_mbSize == 0)
        return false;

    // check to see if we have changed epochs, if so we are done with this one.
    if (m_sequence.size() > 0 && m_mbStartSample > m_sequence[m_sequence.size() - 1])
        return false;

    bool moreData = EnsureDataAvailable(m_mbStartSample);
    if (moreData == false)
        return false;

    // figure which sweep of the randomization we are on
    size_t recordStart = m_totalSamples ? (m_mbStartSample % m_totalSamples) : m_mbStartSample;

    // actual size is the size of the next seqence
    size_t actualmbsize = 0;

    // figure out the size of the next sequence
    // All sequences are concatenated into one long vector; length is computed as the difference of two consecutive start indices.
    if (m_seqIndex > 0 && m_seqIndex < m_sequence.size() && m_sequence.size() > 1)
    {
        actualmbsize = m_sequence[m_seqIndex] - m_sequence[m_seqIndex - 1];
    }
    else
    {
        actualmbsize = m_sequence[0];
    }

    if (actualmbsize > m_mbSize)
    {
        RuntimeError("Specified minibatch size %d is smaller than the actual minibatch size %d.", (int) m_mbSize, (int) actualmbsize);
    }

    // hit the end of the dataset,
    if (!moreData)
    {
        // make sure we take into account hitting the end of the dataset (not wrapping around)
        actualmbsize = min(m_totalSamples - recordStart, actualmbsize);
    }

    // now get the labels
    const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];

    if (labelInfo.type == labelCategory)
    {
        memset(m_labelsBuffer, 0, sizeof(ElemType) * labelInfo.dim * actualmbsize);
        memset(m_labelsIdBuffer, 0, sizeof(LabelIdType) * actualmbsize);
    }
    else if (labelInfo.type != labelNone)
    {
        memset(m_labelsBuffer, 0, sizeof(ElemType) * 1 * actualmbsize);
    }

    if (actualmbsize > 0)
    {
        memset(m_featuresBuffer, 0, sizeof(ElemType) * actualmbsize * labelInfo.dim);

        // loop through all the samples
        int j = 0;
        Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(m_featuresName);
        if (matrices.find(m_featuresName) != matrices.end())
        {
            if (features.GetMatrixType() == MatrixType::DENSE)
            {
                features.Resize(labelInfo.dim, actualmbsize, false);
                features.SetValue(0);
            }
            else
            {
                features.Resize(labelInfo.dim, actualmbsize);
                features.Reset();
            }
        }

        for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
        {
            // pick the right sample with randomization if desired
            const size_t jRand = jSample; // TODO: This seems unfinished.

            // vector of feature data goes into matrix column
            size_t idx = (size_t) m_featureData[jRand];
            m_featuresBuffer[j * labelInfo.dim + idx] = (ElemType) 1;

            if (matrices.find(m_featuresName) != matrices.end())
                features.SetValue(idx, j, (ElemType) 1);
        }

        GetLabelOutput(matrices, m_mbStartSample, actualmbsize);
        GetInputToClass(matrices);
        GetClassInfo();

        // make sure that the sequence index matches our end index
        assert(m_sequence[m_seqIndex] == m_mbStartSample + actualmbsize);
        // go to the next sequence
        m_seqIndex++;
    }

    // advance to the next minibatch
    m_mbStartSample += actualmbsize;

    // if they don't want partial minibatches, skip data transfer and return
    if (actualmbsize == 0) // no records found (end of minibatch)
    {
        return false;
    }

    // now transfer to the GPU as needed
    try
    {
        // get the features array
        if (matrices.find(m_featuresName) == matrices.end())
        {
            Matrix<ElemType>& nbs = matrices.GetInputMatrix<ElemType>(L"numberobs");
            int curDevId = nbs.GetDeviceId();
            nbs.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
            nbs(0, 0) = (float) actualmbsize;
            nbs.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);
            for (size_t i = 0; i < actualmbsize; i++)
            {
                std::wstring ws = msra::strfun::wstrprintf(L"feature%d", i);
                Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(ws);
                features.SetValue(labelInfo.dim, 1, features.GetDeviceId(), &m_featuresBuffer[i * labelInfo.dim], matrixFlagNormal);
            }
        }
    }
    catch (...)
    {
        RuntimeError("Features size not sufficiently large. The asked minibatch size is %d. Check minibatchSize in the feature definition.", (int) actualmbsize);
    }

    try
    {
        if (labelInfo.type == labelCategory)
        {
            if (matrices.find(m_labelsName[labelInfoOut]) == matrices.end())
            {
                // TODO: What is this? Debug code?
                for (size_t i = 0; i < actualmbsize; i++)
                {
                    std::wstring ws = msra::strfun::wstrprintf(L"label%d", i);
                    // This writes into nodes named "labelN", or crashes if they don't exist. Seems this is dead code.
                    Matrix<ElemType>& labels = matrices.GetInputMatrix<ElemType>(ws);
                    labels.SetValue(labelInfo.dim, 1, labels.GetDeviceId(), &m_labelsBuffer[i * labelInfo.dim], matrixFlagNormal);
                }
            }
            // BUGBUG? If category labels then this will not output anything if such node is given.
        }
        else if (labelInfo.type != labelNone)
        {
            Matrix<ElemType>& labels = matrices.GetInputMatrix<ElemType>(m_labelsName[labelInfoOut]);
            labels.SetValue(1, actualmbsize, labels.GetDeviceId(), m_labelsBuffer, matrixFlagNormal);
        }
    }
    catch (...)
    {
        RuntimeError("cannot find matrices for %ls", m_labelsName[labelInfoOut].c_str());
    }

    // we read some records, so process them
    return true;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& SequenceReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    if (m_cachingReader)
    {
        return m_cachingReader->GetLabelMapping(sectionName);
    }
    const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];

    return labelInfo.mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void SequenceReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<IDataReader::LabelIdType, LabelType>& labelMapping)
{
    FailBecauseDeprecated(__FUNCTION__);    // DEPRECATED CLASS, SHOULD NOT BE USED ANYMORE

    if (m_cachingReader)
    {
        RuntimeError("Cannot set mapping table when the caching reader is being used");
    }
    LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];

    labelInfo.mapIdToLabel = labelMapping;
    labelInfo.mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        labelInfo.mapLabelToId[var.second] = var.first;
    }
}

// GetData - Gets metadata from the specified section (into CPU memory)
// sectionName - section name to retrieve data from
// numRecords - number of records to read
// data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
// dataBufferSize - [in] size of the databuffer in bytes
//                  [out] size of buffer filled with data
// recordStart - record to start reading from, defaults to zero (start of data)
// returns: true if data remains to be read, false if the end of data was reached
template <class ElemType>
bool SequenceReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    if (!m_cachingReader)
        RuntimeError("GetData not supported in SequenceReader");
    return m_cachingReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
}

// instantiate all the combinations we expect to be used
template class SequenceReader<double>;
template class SequenceReader<float>;

template <class ElemType>
template <class ConfigRecordType>
void BatchSequenceReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    m_cacheBlockSize = readerConfig(L"cacheBlockSize", m_cacheBlockSize); // in number of tokens

    // initialize the cache
    // InitCache(readerConfig);
    // m_readerConfig = readerConfig;

    // // if we have a cache, no need to parse the test files...
    // if (m_cachingReader)
    //    return;

    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;
    GetFileConfigNames(readerConfig, features, labels);

#if 0 // this actually fails for the RNN example, but still it is not clear what that 'sequence' section does
    // ############ BREAKING ############
    // Added this additional constraint, since I cannot see where we ever use more than one entry. But I may have missed something.
    if (features.size() > 1) // TODO: If this ever fails, please remove this check. One sample had 2 sections, but I could not see where they were used; this check is to verify that.
        InvalidArgument("BatchSequenceReader: Only one features section is allowed.");
    // ############ BREAKING ############
#endif
    if (features.size() > 0)
        m_featuresName = features[0];
    // TODO: Is it at all meaningful to allow no features section?

    if (labels.size() != labelInfoNum)
        RuntimeError("BatchSequenceReader: Two label definitions (in and out) are required, even if one is of labelType 'none'.");

    // BUGBUG: The names come from the dictionary in alphabetical order, not in definition order.
    //         E.g. "labelsIn=[]; labels=[]" will parse labelsIn as the output labels.
    for (int index = 0; index < labelInfoNum; ++index)
        m_labelsName[index] = labels[index];
    // We expect two label sections:
    //  - the alphabetically first one goes into features
    //  - the alphabetically second one goes into labels
    // These labels are interleaved in the input, except for output labelType = "nextWord",
    // in which case the word succeeding the input will be copied as the output (for LM training).
    // It is possible to specify labelType = "none" for either. --TODO: I only tested doing so for the first.

    const ConfigRecordType& featureConfig = m_featuresName.empty() ? readerConfig : readerConfig(m_featuresName.c_str(), ConfigRecordType::Record());
    wstring mode = featureConfig(L"mode", L"class"); // class, softmax, nce

    if (EqualCI(mode, L"nce"))
    {
        readerMode = ReaderMode::NCE;
        m_noiseSampleSize = featureConfig(L"noise_number", 0);
    }
    else if (EqualCI(mode, L"softmax"))
        readerMode = ReaderMode::Softmax;
    else if (EqualCI(mode, L"class"))
        readerMode = ReaderMode::Class;
    else
        LogicError("unsupported format %ls", mode.c_str());

    // unk sybol
    mUnk = msra::strfun::utf8(readerConfig(L"unk", L"<unk>"));

    m_classSize = 0;
    m_featureDim = m_featuresName.empty() ? 0 : featureConfig(L"dim");
    for (int index = 0; index < labelInfoNum; ++index)
    {
        // the code below builds the labelInfo record for this label input
        auto& labelInfo = m_labelInfo[index];

        const ConfigRecordType& labelConfig = readerConfig(m_labelsName[index].c_str(), ConfigRecordType::Record());

        labelInfo.numIds = 0; // if no mapping or word-class file is read, then we build the mapping on the fly
        labelInfo.beginSequence = msra::strfun::utf8(labelConfig(L"beginSequence", L""));
        labelInfo.endSequence   = msra::strfun::utf8(labelConfig(L"endSequence",   L""));

        // determine label type desired
        std::string labelType(labelConfig(L"labelType", "category"));
        if (EqualCI(labelType, "category"))
        {
            labelInfo.type = labelCategory;
            labelInfo.dim = (LabelIdType)(size_t)labelConfig(L"labelDim", 0); // 0 means infer from inputs
        }
        else if (EqualCI(labelType, "nextWord"))
        {
            if (index == 0)
                InvalidArgument("BatchSequenceReader: Input labels must not be of kind 'nextWord'.");
            // in this case, it's all identical to the Input labels, except the data type
            labelInfo.type = labelNextWord;
            labelInfo.dim = m_labelInfo[labelInfoIn].dim;
            // ########### BREAKING ##########
            // old version overwrote this by looking up a 'labelDim' parameter
            // ########### BREAKING ##########
        }
        else if (EqualCI(labelType, "none"))
        {
            labelInfo.type = labelNone;
            labelInfo.dim = 0; // override for no labels
        }
        else
            InvalidArgument("BatchSequenceReader: Invalid labelType '%s'", labelType.c_str());

        // if we have labels, we need a label Mapping file, it will be a file with one label per line
        if (labelInfo.type != labelNone)    // BUGBUG: This seems wrong for "nextWord" case. E.g. it reads the class info twice.
        {
            nwords = labelInfo.dim; // BUGBUG: 'nwords' is a class member, but it gets set twice in nextWord case

            // read the word-class definition file if specified
            std::wstring wClassFile = readerConfig(L"wordclass", L"");
            if (wClassFile != L"")
            {
                ReadClassInfo(wClassFile, m_classSize,
                              word4idx,
                              idx4word,
                              idx4class,
                              idx4cnt,
                              nwords, // only used for a consistency check
                              mUnk, m_noiseSampler,
                              false);
            }

            // read a word-mapping file if present
            std::vector<string> arrayLabels;
            // Note: labelMappingFile is I/O. It is created if it does not exist yet.
            std::wstring labelPath = labelConfig(L"labelMappingFile");
            if (File::Exists(labelPath))
            {
                File::LoadLabelFile(labelPath, arrayLabels);
                // build the two-way mapping tables
                for (int i = 0; i < arrayLabels.size(); ++i)
                {
                    LabelType label = arrayLabels[i];
                    labelInfo.mapIdToLabel[i] = label;
                    labelInfo.mapLabelToId[label] = i;
                }
                labelInfo.numIds = (LabelIdType) arrayLabels.size();
                labelInfo.mapName = labelPath;
                labelInfo.fileToWrite.clear();  // (not an output, so nothing to write at end)
            }
            else
            {
                fprintf(stderr, "LMSequenceReader: Label mapping will be created internally on the fly because the labelMappingFile was not found: %ls\n", labelPath.c_str());
                if (wClassFile != L"")
                {
#if 0
                    // ######### BREAKING #########
                    // removed this seemingly redundant read--but is it?
                    // ######### BREAKING #########
                    // BUGBUG: The same thing was just done above, so isn't redundant?
                    ReadClassInfo(wClassFile, m_classSize,
                                  word4idx,
                                  idx4word,
                                  idx4class,
                                  idx4cnt,
                                  nwords,
                                  mUnk, m_noiseSampler,
                                  false);
#endif
                    if (word4idx.size() != nwords) // TODO: Why not infer it at this point in time? If labelInfo.dim == 0 then set if to word4idx.size()
                        LogicError("BatchSequenceReader::Init : vocabulary size %d from setup file and %d from that in word class file %ls is not consistent", (int) nwords, (int) word4idx.size(), wClassFile.c_str());
                    int iMax = -1;
                    int i;
                    for (auto ptr = word4idx.begin(); ptr != word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first;
                        i = ptr->second;
                        iMax = max(i, iMax);
                        labelInfo.mapIdToLabel[i] = label;
                        labelInfo.mapLabelToId[label] = i;
                    }
                    labelInfo.numIds = (LabelIdType)(iMax + 1);
                }
                labelInfo.mapName = labelPath;
                labelInfo.fileToWrite = labelPath; // mapping path denotes an output: write the mapping here at the end
                // BUGBUG: This facility is not functional. No file is being created.
            }
        }

        // update dimension if the class or mapping file says it's bigger, or if it is specified as 0 (meaning 'infer from inputs')
        // TODO: Clean this up--do this only if numIds is 0 (no class or mapping read), otherwise require them to be identical or 0.
        if (labelInfo.dim < labelInfo.numIds)
            labelInfo.dim = labelInfo.numIds;
    }

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = m_seqIndex = 0;
    m_endReached = false;
    m_readNextSampleLine = 0;
    m_readNextSample = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);
    m_parser.SetTraceLevel(m_traceLevel);

    // BUGBUG: "randomize" option is ignored

    // The input data is a combination of the label Data and extra feature dims together
    //    m_featureCount = m_featureDim + m_labelInfo[labelInfoIn].dim;
    m_featureCount = 1;

    wstring pathName = readerConfig(L"file", L"");
    if (m_traceLevel > 0)
        fwprintf(stderr, L"LMSequenceReader: Input file is '%s'.\n", pathName.c_str());

    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseInit(pathName.c_str(), m_featureDim, labelIn.dim, labelOut.dim, labelIn.beginSequence, labelIn.endSequence, labelOut.beginSequence, labelOut.endSequence);

    mRequestedNumParallelSequences = readerConfig(L"nbruttsineachrecurrentiter", (size_t) 1); // 0 indicates auto-fill mbSize
    // TODO: ^^ This should depend on the sequences themselves.
}

template <class ElemType>
void BatchSequenceReader<ElemType>::Reset()
{
    mProcessed.clear();
    mToProcess.clear();
    mLastProcessedSentenceId = 0;
    mPosInSentence = 0;
    mLastPosInSentence = 0;
    mNumRead = 0;

    if (m_labelTemp.size() > 0)
        m_labelTemp.clear();
    if (m_featureTemp.size() > 0)
        m_featureTemp.clear();
    m_parser.mSentenceIndex2SentenceInfo.clear();
}

template <class ElemType>
void BatchSequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory(); // free the memory used by the SequenceReader
    }

    // if we are reading from the cache, do so now and return
    if (m_cachingReader)
    {
        m_cachingReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
        return;
    }

    // allocate buffers for building features and labels
    // We allocate 'mbSize' entries, i.e. that's the total token cound across multiple parallel sequences.
    const LabelInfo& labelInfo = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut];

    if (m_featuresBuffer == NULL)       // features
        m_featuresBuffer = new ElemType[mbSize * labelInfo.dim]();

    if (m_labelsBuffer == NULL)         // labels
    {
        if (labelInfo.type == labelCategory)
        {
            m_labelsBuffer = new ElemType[mbSize * labelInfo.dim]();
            m_labelsIdBuffer = new /*typename IDataReader<ElemType>::*/LabelIdType[mbSize]();     // TODO: no "new" please! Use a vector
        }
        else if (labelInfo.type != labelNone)
        {
            m_labelsBuffer = new ElemType[mbSize]();
            m_labelsIdBuffer = NULL;
        }
    }

    //m_featuresBufferRow = new size_t[mbSize];
    //m_featuresBufferRowIdx = new size_t[mbSize];

    m_id2classLocal = new Matrix<ElemType>(CPUDEVICE);
    m_classInfoLocal = new Matrix<ElemType>(CPUDEVICE);

    m_mbSize = mbSize;
    if (requestedEpochSamples == requestDataSize)
    {
        if (!m_endReached)
        {
            m_epochSize = requestDataSize;
        }
    }
    else
    {
        m_epochSize = requestedEpochSamples;
    }

    // we use epochSize, which might not be set yet, so use a default value for allocations if not yet set
    size_t epochSize = m_epochSize == requestDataSize ? 1000 : m_epochSize;
    m_epoch = epoch;
    m_randomSeed = (unsigned int)m_epoch;
    m_mbStartSample = epoch * m_epochSize;
    m_epochSamplesReturned = 0;     // counter to know when we returned one epoch

    // allocate room for the data
    m_featureData.reserve(m_featureCount * epochSize);
    if (m_labelInfo[labelInfoOut].type == labelCategory)
        m_labelIdData.reserve(epochSize);
    //else if (m_labelInfo[labelInfoOut].type != labelNone)
    //    m_labelData.reserve(epochSize);
    m_sequence.reserve(m_seqIndex); // clear out the sequence array
    // this is too complicated for LM
    // SetupEpoch();
    // use the LMSetupEpoch() instead
    LMSetupEpoch();

    m_clsinfoRead = false;
    m_idx2clsRead = false;

    m_parser.ParseReset();

    Reset();
}

// fill mToProcess[] with the next set of sequences of the same length
// This function updates mToProcess[] (only, except it also lazily initializes mProcessed[]).
// If mToProcess[] is not empty, then it will check whether those sequences are done, and if not, just return with mToProcess[] unchanged.
// It returns 0 if there are no more sequences to process in this epoch.
template <class ElemType>
size_t BatchSequenceReader<ElemType>::DetermineSequencesToProcess()
{
    if (mNumRead == 0) // no data loaded (either this is the first call, or the input file empty)
        return 0;

    // lazily create the mProcessed[] array
    if (mProcessed.size() == 0)
        mProcessed.resize(mNumRead, false);

    // lazily check if we are done
    // We are done if mToProcess[] contains a sequence for which mProcessed[] is set (...which likely applies to all?).
    if (mToProcess.size() > 0)
    {
        bool allDone = false;
        // allDone gets set when at least one entry in mToProcess[] is complete
        // I guess since they all have the same length, they are then all complete
        for (int s = 0; s < mToProcess.size(); s++)
        {
            int mp = (int) mToProcess[s];
            if (mProcessed[mp])
            {
                mLastProcessedSentenceId = mp;
                mLastPosInSentence = 0;
                allDone = true;
                break;
            }
        }
        if (allDone) // if we are done
            mToProcess.clear();
    }

    // if we still have unfinished sequences then just return their length
    if (mToProcess.size() > 0)
    {
        // They are all the same length, so we can just get the value from the first entry.
        return m_parser.mSentenceIndex2SentenceInfo[mToProcess[0]].sLen;
    }

    // mToProcess[] is empty: fill it up with at most mRequestedNumParallelSequences entries of the same length
    size_t sln = 0;
    size_t maxToProcess = mRequestedNumParallelSequences > 0 ? mRequestedNumParallelSequences : SIZE_MAX; // if mRequestedNumParallelSequences is 0 then we go by MB size
    size_t maxTokens    = mRequestedNumParallelSequences > 0 ?                       SIZE_MAX : m_mbSize;
    size_t numTokens = 0;  // token counter
    for (size_t seq = mLastProcessedSentenceId;
         seq < mNumRead &&                 // hit end of buffer
         mToProcess.size() < maxToProcess; // hit parallel-sequence limit
         seq++)
    {
        // skip entries that are done
        if (mProcessed[seq])
            continue;

        // first unprocessed sequence determines the length if this minibatch
        if (sln == 0)
            sln = m_parser.mSentenceIndex2SentenceInfo[seq].sLen;
        else if (sln != m_parser.mSentenceIndex2SentenceInfo[seq].sLen)
            continue;

        // check max tokens
        if (numTokens > 0 && numTokens + sln >= maxTokens)
            break;             // hit total token limit

        // include all of the same length as the first
        mToProcess.push_back(seq);

        // and count tokens
        numTokens += m_parser.mSentenceIndex2SentenceInfo[seq].sLen;
    }
    // if all were already done, we will get here with sln=0 and return that

    //fprintf(stderr, "DetermineSequencesToProcess: %d sequences of len %d, %d tokens\n", (int) mToProcess.size(), (int) sln, (int) numTokens);

    return sln;
}

// fill the buffer with the next one or more sequences to process
// Advances mLastPosInSentence, which is the cursor into the sentence(s).
// Returns the previous value of mLastPosInSentence, which is needed in creating the MB layout.
// This is a subroutine for GetMinibatch(), it is not called from elsewhere.
template <class ElemType>
bool BatchSequenceReader<ElemType>::GetMinibatchData(size_t& /*out*/ firstPosInSentence)
{
    // we returned enough samples to make up one epoch
    // This is a stopgap, with these flaws:
    //  - the next epoch will reload and re-randomize (with epoch# as random seed), so we don't guarantee to not return the same samples
    //  - the parser will start over, to cacheBlockSize must be >= corpus size, and all gets (re-)loaded into RAM each epoch
    // The alternative, set cacheBlockSize == epoch size, is not good since it will have very few utterances of the same length to batch.
    // We will not fix much more than this since the soon-to-come new reader API will solve these issues.
    if (m_epochSamplesReturned > m_epochSize)
        return false;

    m_featureData.clear();
    m_labelIdData.clear();

    // --- STEP 1: determine the next set of sequences to process (-> mToProcess[])

    size_t sLn = DetermineSequencesToProcess();
    if (sLn == 0) // none left: we hit the end of an epoch boundary; shuffle in the next epoch
    {
        Reset();

        std::vector<SequencePosition> seqPos;
        fprintf(stderr, "LMSequenceReader: Reading epoch data..."), fflush(stderr);
        mNumRead = m_parser.Parse(m_cacheBlockSize, &m_labelTemp, &m_featureTemp, &seqPos);
        fprintf(stderr, " %d sequences read.\n", (int) mNumRead);
        firstPosInSentence = mLastPosInSentence;
        if (mNumRead == 0)
            return false; // end

#ifdef _MSC_VER // make some old configurations reproducable (m_cacheBlockSize used to be a constant)  --TODO: remove in a few months
        if (m_cacheBlockSize == 50000)
        {
            srand(++m_randomSeed); // TODO: older code did not have that; so no idea what random seed was used
            std::random_shuffle(m_parser.mSentenceIndex2SentenceInfo.begin(), m_parser.mSentenceIndex2SentenceInfo.end());
            // Note: random_shuffle is deprecated since C++14.
        }
        else // new configs use a wider randomization
#endif
        {
            std::mt19937 g(++m_randomSeed); // random seed is initialized to epoch, but gets incremented for intermediate reshuffles
            std::shuffle(m_parser.mSentenceIndex2SentenceInfo.begin(), m_parser.mSentenceIndex2SentenceInfo.end(), g);
        }

        m_readNextSampleLine += mNumRead;
        sLn = DetermineSequencesToProcess();
    }

    // --- STEP 2: copy sentences in mToProcess[] into m_featureData[] and m_labelIdData[]

    const bool nextWord = (m_labelInfo[labelInfoOut].type == labelNextWord);
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelInfo& labelOut = m_labelInfo[labelInfoOut];

    firstPosInSentence = mLastPosInSentence;

    size_t & i = mLastPosInSentence;
    size_t iend = sLn - (labelOut.type != labelNone);       // exclude the last token since it is the last label to be predicted
    // ############### BREAKING CHANGE ################
    // We use sLn, not sLn -1, if labelOut.type is labelNode, assuming there is no output label, and all labels are inputs.
    // ############### BREAKING CHANGE ################
    const size_t jend = mRequestedNumParallelSequences > 0 ? m_mbSize : SIZE_MAX;                           // mbSize here is truncation length
    for (size_t j = 0; j < jend && i < iend; i++, j++)
    {
        for (int k = 0; k < mToProcess.size(); k++)
        {
            size_t seq = mToProcess[k];
            size_t pos = m_parser.mSentenceIndex2SentenceInfo[seq].sBegin + i;

            // labelIn should be a category label
            const auto& labelValue = m_labelTemp[pos];
            pos++; // consume it

            // generate the feature token
            if (labelIn.type == labelCategory)
            {
                LabelIdType labelId = GetIdFromLabel(labelValue, labelIn);

                // use the found value, and set the appropriate location to a 1.0
                assert(labelIn.dim > labelId); // if this goes off labelOut dimension is too small
                m_featureData.push_back((float) labelId);
            }
            else
                RuntimeError("Input labels are expected to be category labels.");

            // generate the output label token
            if (labelOut.type != labelNone)
            {
                const auto& labelValue = m_labelTemp[pos];
                LabelIdType labelId;
                if (labelOut.type == labelCategory)
                {
                    pos++; // consume it   --TODO: value is not used after this
                    labelId = GetIdFromLabel(labelValue, labelOut);
                }
                else if (nextWord)
                {
                    // this is the next word (pos was already incremented above when reading out labelValue)
                    if (EqualCI(labelValue, labelIn.endSequence)) // end symbol may differ between input and output
                        labelId = GetIdFromLabel(labelIn.endSequence, labelIn);
                    else
                        labelId = GetIdFromLabel(labelValue, labelIn);
                }
                else
                    LogicError("Unexpected output label type."); // should never get here

                m_labelIdData.push_back(labelId);
            }

            m_totalSamples++;
            m_epochSamplesReturned++;
        }
    }
    //mLastPosInSentence = i; // (we could also iterate over mLastPosInSentence directly)

    // remember if we are at the end
    // Note: This flag will propagate into setting mProcessed[] in DataEnd(), which seems a bit fragile. Can't we do it here?
    mSentenceEnd = (i == iend);

    return true;
}

// How this returns data:
//  - input sequences are chunked into "epochs" as defined in XXX?
//  - each epoch's sequences are randomly sorted
//  - up to N sequences of the same length are returned in each MB
//     - minibatches consist of sequences of the same length only (no gaps)
template <class ElemType>
bool BatchSequenceReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    // get out if they didn't call StartMinibatchLoop() first
    // TODO: Why not fail here?
    if (m_mbSize == 0)
        return false;

    // TODO: validate whether the passed matrices set matches reader section definitions

    // --- STEP 1: get the data for the this minibatch

    size_t firstPosInSentence;
    bool moreData = GetMinibatchData(firstPosInSentence);
    if (!moreData)
    {
        m_pMBLayout->Init(mToProcess.size(), 0);
        return false;
    }
    // now:
    //  - there is at least one sequence to return
    //  - mToProcess[] lists all sequences
    //  - mProcessed[s] says whether a sequence has completed
    //  - m_featureData[j] contains the input label indices
    //  - m_labelIdData[j] contains the output label indices

    // --- STEP 2: transfer the data to the matrices

    size_t actualmbsize = m_featureData.size(); // total number of tokens across all sequences in this MB
    assert(actualmbsize > 0);
    assert(m_labelIdData.empty() || m_labelIdData.size() == actualmbsize);

    // create MBLayout
    // This handles variable-length sequences.
    size_t nT = actualmbsize / mToProcess.size();
    assert(nT * mToProcess.size() == actualmbsize);
    m_pMBLayout->Init(mToProcess.size(), nT);
    for (size_t s = 0; s < mToProcess.size(); s++)
    {
        size_t seq = mToProcess[s];
        const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
        size_t len = m_parser.mSentenceIndex2SentenceInfo[seq].sLen - (labelOut.type != labelNone); // -1 because last one is label
        // ############### BREAKING CHANGE ################
        // We use sLen, not sLen -1, if labelOut.type is labelNode, assuming there is no output label, and all labels are inputs.
        // ############### BREAKING CHANGE ################
        ptrdiff_t begin = -(ptrdiff_t)firstPosInSentence;
        ptrdiff_t end = (ptrdiff_t) len - (ptrdiff_t) firstPosInSentence;
        if (begin >= (ptrdiff_t) nT)
            LogicError("BatchSequenceReader: Sentence begin outside minibatch?");
        if (end < 0)
            LogicError("BatchSequenceReader: Sentence end outside minibatch?");
        m_pMBLayout->AddSequence(NEW_SEQUENCE_ID, s, begin, (size_t) end);
        if (begin > 0)
            m_pMBLayout->AddGap(s, 0, (size_t) begin);
        if (end < (ptrdiff_t) nT)
            m_pMBLayout->AddGap(s, end, nT);
    }

    // copy m_featureData to matrix
    // m_featureData is a sparse, already with interleaved parallel sequences. We copy it into a dense matrix.
    // we always copy it to cpu first and then convert to gpu if gpu is desired.
    size_t featureDim = m_labelInfo[labelInfoIn].dim;
    auto iter = matrices.find(m_featuresName);
    if (iter != matrices.end()) // (if not found then feature matrix is not requested this time)
    {
        Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(iter->first);

        bool moveMatrix = features.GetMatrixType() != MatrixType::SPARSE;
        // BUGBUG: Matrix does not support modifying a dense GPU matrix temporarily ^^ on the CPU. We should instead have a local CPU matrix and assign it over.
        DEVICEID_TYPE featureDeviceId = features.GetDeviceId();
        features.TransferFromDeviceToDevice(featureDeviceId, CPUDEVICE, moveMatrix, true, false);

        if (features.GetMatrixType() == MatrixType::SPARSE)
        {
            features.Resize(featureDim, actualmbsize, actualmbsize);
            features.Reset();
        }
        else
        {
            features.Resize(featureDim, actualmbsize);
            features.SetValue(0);
        }

        for (size_t j = 0; j < actualmbsize; ++j) // note: this is a loop over matrix columns, not time steps
        {
            // vector of feature data goes into matrix column
            size_t idx = (size_t) m_featureData[j]; // one-hot index of the word, indexed by column (i.e. already interleaved, t=j/mToProcess.size(), s=j%mToProcess.size())

            features.SetValue(idx, j, (ElemType) 1);
        }

        features.TransferFromDeviceToDevice(CPUDEVICE, featureDeviceId, moveMatrix, false, false);
    }

    // get class info
    if (readerMode == ReaderMode::Class)
    {
        GetInputToClass(matrices);
        GetClassInfo();
    }

    // get labels
    GetLabelOutput(matrices, 0, actualmbsize);

    // go to the next sequence
    m_seqIndex++;

#if 0 // what is this?
    // now transfer to the GPU as needed
    // get the features array
    if (matrices.find(m_featuresName) == matrices.end())
    {
        Matrix<ElemType>& nbs = *matrices[L"numberobs"]; // TODO: what is this? We fall back to a different node?
        int curDevId = nbs.GetDeviceId();
        nbs.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
        nbs(0, 0) = (float)actualmbsize;
        nbs.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);
        for (size_t i = 0; i < actualmbsize; i++)
        {
            std::wstring ws = msra::strfun::wstrprintf(L"feature%d", i);
            Matrix<ElemType>& features = *matrices[ws];
            features.SetValue(featureDim, 1, features.GetDeviceId(), &m_featuresBuffer[i * featureDim], matrixFlagNormal);
        }
    }
#endif

    // we read some records, so process them
    return true;
}

#if 0
/**
timePos: the time position. for example, 100 actual minibatch with 10 streams,
timePosition = [0,..,9] for each actual tiem
*/
// This function was only called from BatchSequenceReader::TryGetMinibatch(), but no longer.
template <class ElemType>
void BatchSequenceReader<ElemType>::SetSentenceBegin(int wrd, int uttPos, int timePos)
{
    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelIdType index = GetIdFromLabel(labelIn.beginSequence.c_str(), labelIn);

    if (timePos == 0)
    {
        if (wrd == (int) index)
        {
            mSentenceBegin = true;
            // BUGBUG: This is currently not functional. Nothing in here marks gaps, nor sets any end flags.
            uttPos;
            LogicError("BatchSequenceReader::SetSentenceBegin(): Disabled because this implementation seems out of date w.r.t. setting end and gap flags.");
            // m_pMBLayout->SetWithoutOr(uttPos, timePos, MinibatchPackingFlags::SequenceStart);   // TODO: can we use Set() (with OR)?
        }
    }
}
#endif

#if 0
// TODO: this should have been renamed to CopyMBLayoutTo(), but it had the wrong signature??
template <class ElemType>
void BatchSequenceReader<ElemType>::SetSentenceSegBatch(vector<size_t>& sentenceEnd)
{
    sentenceEnd.resize(mToProcess.size());
    if (mSentenceBegin)
    {
        sentenceEnd.assign(mToProcess.size(), 0);
    }
    else
    {
        sentenceEnd.assign(mToProcess.size(), m_mbSize + 2);
    }
}
#endif

// note: DataEnd() must be called for each minibatch in order to propagate mSentenceEnd to mProcessed[]
template <class ElemType>
bool BatchSequenceReader<ElemType>::DataEnd()
{
    if (mSentenceEnd)
        for (auto seq : mToProcess)
            mProcessed[seq] = true;
    return mSentenceEnd;
}

// fill the labels (from m_labelIdData)
// When using classes, labels are in [L x T] matrix.
// where L depends on reader mode:
//     4             under CLASS           [wid, class-id, beg-class, end-class]
//     2*(noise + 1) under NCE training    [wid, prob, (noise-id, noise-prob)+]
//     1             o.w.                  [wid]
// the following comments are obsolete now
// 1st row is the word id
// 2nd row is the class id of this word
// 3rd and 4th rows are the begining and ending indices of this class
// notice that indices are defined as follows [begining ending_indx) of the class
// i.e., the ending_index is 1 plus of the true ending index
// This is a subroutine to GetMinibatch() and is only called from there.
template <class ElemType>
void BatchSequenceReader<ElemType>::GetLabelOutput(StreamMinibatchInputs& matrices,
                                                   size_t mbStartSample, size_t actualmbsize)
{
    if (!matrices.HasInput(m_labelsName[labelInfoOut]))
        return;

    size_t j = 0;
    Matrix<ElemType>& labels = matrices.GetInputMatrix<ElemType>(m_labelsName[labelInfoOut]);
    if (readerMode == ReaderMode::NCE)
        labels.Resize(2 * (m_noiseSampleSize + 1), actualmbsize);
    else if (readerMode == ReaderMode::Class)
        labels.Resize(4, actualmbsize, false);
    else
        labels.Resize(1, actualmbsize, false);

    // move to CPU since element-wise operation is expensive on GPU
    int curDevId = labels.GetDeviceId();
    labels.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
    assert(labels.GetCurrentMatrixLocation() == CPU);

    ElemType epsilon = (ElemType) 1e-6; // avoid all zero, although this is almost impossible.

    for (size_t jSample = mbStartSample; j < actualmbsize; ++j, ++jSample)
    {
        // get the token
        LabelIdType wrd = m_labelIdData[jSample];

        // write sample value into output
        // This writes an index into a row vector. Which is wrong, we want a sparse one-hot vector.
        labels.SetValue(0, j, (ElemType) wrd);

        if (readerMode == ReaderMode::NCE)
        {
            labels.SetValue(1, j, (ElemType) m_noiseSampler.logprob(wrd));
            for (size_t noiseid = 0; noiseid < m_noiseSampleSize; noiseid++)
            {
                int wid = m_noiseSampler.sample();
                labels.SetValue(2 * (noiseid + 1), j, (ElemType) wid);
                labels.SetValue(2 * (noiseid + 1) + 1, j, -(ElemType) m_noiseSampler.logprob(wid));
            }
        }
        else if (readerMode == ReaderMode::Class)
        {
            int clsidx = idx4class[wrd];
            if (m_classSize > 0)
            {
                labels.SetValue(1, j, (ElemType) clsidx);

                // save the [begining ending_indx) of the class
                size_t lft = (size_t) (*m_classInfoLocal)(0, clsidx);
                size_t rgt = (size_t) (*m_classInfoLocal)(1, clsidx);
                if (wrd < lft || lft > rgt || wrd >= rgt)
                {
                    LogicError("LMSequenceReader::GetLabelOutput word %d should be at least equal to or larger than its class's left index %d; right index %d of its class should be larger or equal to left index %d of its class; word index %d should be smaller than its class's right index %d.\n",
                               (int) wrd, (int) lft, (int) rgt, (int) lft, (int) wrd, (int) rgt);
                }
                labels.SetValue(2, j, (*m_classInfoLocal)(0, clsidx)); // begining index of the class
                labels.SetValue(3, j, (*m_classInfoLocal)(1, clsidx)); // end index of the class
            }
        }
        else if (readerMode == ReaderMode::Softmax)
        {
            if (wrd == 0)
                labels.SetValue(0, j, epsilon + (ElemType) wrd);
        }
        else if (readerMode == ReaderMode::Unnormalize)
        {
            labels.SetValue(0, j, -(ElemType) wrd);
            if (wrd == 0)
                labels.SetValue(0, j, -epsilon - (ElemType) wrd);
        }

#if 0   // This is fragile--if a line does not end with </s>, we will never get out of here. Instead, this is now set in GetMinibatchData().
        // keep track of sentence ends. mSentenceEnd is read only in DataEnd(), which will then reset mProcessed[].
        if (j == actualmbsize - 1) // if last entry then set mSentenceEnd to whether the word is the sentence-end symbol
        {
            // find numeric index of sentence-end token
            LabelInfo& labelIn = m_labelInfo[(m_labelInfo[labelInfoOut].type == labelNextWord) ? labelInfoIn : labelInfoOut]; // m_labelInfo[labelInfoIn]; <-- This is what it was before, which triggered a bug and I think is wrong wrd comes from m_labelIdData, which is the output.
            LabelIdType endSequenceIndex = GetIdFromLabel(labelIn.endSequence, labelIn);

            mSentenceEnd = (wrd == (int)endSequenceIndex);
        }
#endif
    }
    // send it back to where it came from
    // Note: This may leave this object in BOTH locations, which is desirable for
    // class-based models which access the information on the CPU.
    labels.TransferFromDeviceToDevice(CPUDEVICE, curDevId, false, false, false);
}

#if 0
template <class ElemType>
int BatchSequenceReader<ElemType>::GetSentenceEndIdFromOutputLabel()
{

    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    auto found = word4idx.find(labelIn.endSequence);

    // not yet found, add to the map
    if (found != word4idx.end())
    {
        return (int) found->second;
    }
    else
        return -1;
}
#endif

template class BatchSequenceReader<double>;
template class BatchSequenceReader<float>;
} } }
