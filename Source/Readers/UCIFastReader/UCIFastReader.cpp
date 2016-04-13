//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// UCIFastReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
#include "UCIFastReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#include "fileutil.h" // for fexists()

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
size_t UCIFastReader<ElemType>::RandomizeSweep(size_t mbStartSample)
{
    // size_t randomRangePerEpoch = (m_epochSize+m_randomizeRange-1)/m_randomizeRange;
    // return m_epoch*randomRangePerEpoch + epochSample/m_randomizeRange;
    return mbStartSample / m_randomizeRange;
}

// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template <class ElemType>
bool UCIFastReader<ElemType>::ReadRecord(size_t /*readSample*/)
{
    return false; // not used
}

// RecordsToRead - Determine number of records to read to populate record buffers
// mbStartSample - the starting sample from which to read
// tail - we are checking for possible remainer records to read (default false)
// returns - true if we have more to read, false if we hit the end of the dataset
template <class ElemType>
size_t UCIFastReader<ElemType>::RecordsToRead(size_t mbStartSample, bool tail)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    bool randomize = Randomize();
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    epochSample %= m_epochSize;

    // determine number left to read for this epoch
    size_t numberToEpoch = m_epochSize - epochSample;
    // we will take either a minibatch or the number left in the epoch
    size_t numberToRead = min(numberToEpoch, m_mbSize);
    if (numberToRead == 0 && !tail)
        numberToRead = m_mbSize;

    if (randomize)
    {
        size_t randomizeSweep = RandomizeSweep(mbStartSample);
        // if first read or read takes us to another randomization range
        // we need to read at least randomization range records
        if (randomizeSweep != m_randomordering.CurrentSeed()) // the range has changed since last time
        {
            numberToRead = RoundUp(epochSample, m_randomizeRange) - epochSample;
            if (numberToRead == 0 && !tail)
                numberToRead = m_randomizeRange;
        }
    }
    return numberToRead;
}

// EnsureDataAvailable - Read enough lines so we can request a minibatch starting as requested
// mbStartSample - the starting sample we are ensureing are good
// endOfDataCheck - check if we are at the end of the dataset (no wraparound)
// returns - true if we have more to read, false if we hit the end of the dataset
template <class ElemType>
bool UCIFastReader<ElemType>::EnsureDataAvailable(size_t mbStartSample, bool endOfDataCheck)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    Randomize();
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    epochSample %= m_epochSize;
    bool moreToRead = true;

    size_t numberToRead = RecordsToRead(mbStartSample);

    // check to see if we have the proper records read already
    if (m_readNextSample >= mbStartSample + numberToRead && mbStartSample >= m_epochStartSample)
        return true;

    // truncate the present arrays to the location we are reading from, parser appends on these arrays
    if (m_featureData.size() > epochSample * m_featureCount) // should be this size, if not, truncate
        m_featureData.resize(epochSample * m_featureCount);
    if (m_labelType == labelCategory && m_labelData.size() > epochSample)
    {
        m_labelIdData.resize(epochSample);
        m_labelData.resize(epochSample);
    }
    else if (m_labelType != labelNone && m_labelData.size() > epochSample * m_labelDim)
    {
        m_labelData.resize(epochSample * m_labelDim);
    }

    int recordsRead = 0;
    do
    {
        int numRead = m_parser->Parse(numberToRead - recordsRead, &m_featureData, &m_labelData);

        recordsRead += numRead;
        if (!m_endReached)
            m_totalSamples += numRead; // total number of records in the dataset

        // we should only get less records than requested at when we hit the end of the dataset
        if (recordsRead < numberToRead)
        {
            // update dataset variables
            size_t additionalToRead = UpdateDataVariables(mbStartSample + recordsRead);

            m_parser->SetFilePosition(0); // make another pass of the dataset

            // if doing and end of data check, and we are at the end
            // or a partial minibatch was found exit now
            if ((endOfDataCheck && recordsRead == 0) ||
                (m_partialMinibatch && recordsRead > 0))
            {
                moreToRead = false;
                break;
            }

            // get the additional number to read
            numberToRead = recordsRead + additionalToRead;
        }
    } while (recordsRead < numberToRead);
    m_readNextSample += recordsRead;

    // for category labels, we need to build up a list of IDs and a mapping table
    if (m_labelType == labelCategory)
    {
        // loop through all the newly read records
        for (int numberRead = 0; numberRead < recordsRead; numberRead++)
        {
            LabelType& label = m_labelData[epochSample + numberRead];
            // check to see if we have seen this label before
            auto value = m_mapLabelToId.find(label);
            LabelIdType labelId;
            if (value == m_mapLabelToId.end())
            {
                if (m_labelFileToWrite.empty())
                    RuntimeError("label found in data not specified in label mapping file: %s", label.c_str());
                // new label so add it to the mapping tables
                m_mapLabelToId[label] = m_labelIdMax;
                m_mapIdToLabel[m_labelIdMax] = label;
                labelId = m_labelIdMax++;

                // if our label dimension is lower than the current labelId then increase it
                if (m_labelDim < m_labelIdMax)
                    m_labelDim = m_labelIdMax;
            }
            else
            {
                labelId = value->second;
            }

            // now add the label id to the label data array
            m_labelIdData.push_back(labelId);
        }
    }
    // if there more to read (always is, unless we want partial minibatches
    return moreToRead;
}

// UpdateDataVariables - Update variables that depend on the dataset being completely read
template <class ElemType>
size_t UCIFastReader<ElemType>::UpdateDataVariables(size_t mbStartSample)
{
    // if we haven't been all the way through the file yet
    if (!m_endReached)
    {
        // get the size of the dataset
        assert(m_totalSamples * m_featureCount >= m_featureData.size());

        // if they want us to determine epoch size based on dataset size, do that
        if (m_epochSize == requestDataSize)
        {
            // set the epoch size to be a multiple of mbSize or randomization range
            if (m_partialMinibatch)
                m_epochSize = m_totalSamples;
            else
            {
                size_t roundUpTo = m_mbSize;
                if (m_randomizeRange != randomizeAuto && m_randomizeRange != randomizeNone)
                    roundUpTo = m_randomizeRange;
                m_epochSize = RoundUp(m_totalSamples, roundUpTo);
            }
        }

        // make sure randomization range is within the sample bounds
        if (m_randomizeRange > m_epochSize)
        {
            m_randomizeRange = m_epochSize;
            m_randomordering.Resize(m_randomizeRange, m_randomizeRange);
        }

        // write the label file if we hit the end of the file
        WriteLabelFile();

        // we got to the end of the dataset
        m_endReached = true;
    }

    // update the label dimension if it is not big enough, need it here because m_labelIdMax get's updated in the processing loop (after a read)
    if (m_labelType == labelCategory && m_labelIdMax > m_labelDim)
        m_labelDim = m_labelIdMax; // update the label dimensions if different

    bool recordsToRead = mbStartSample < m_epochStartSample + m_epochSize; // still some to read after potential epochSize change?
    return recordsToRead ? RecordsToRead(mbStartSample) : 0;
}

template <class ElemType>
void UCIFastReader<ElemType>::WriteLabelFile()
{
    // write out the label file if they don't have one
    if (!m_labelFileToWrite.empty())
    {
        if (m_mapIdToLabel.size() > 0)
        {
            File labelFile(m_labelFileToWrite, fileOptionsWrite | fileOptionsText);
            for (int i = 0; i < m_mapIdToLabel.size(); ++i)
            {
                labelFile << m_mapIdToLabel[i] << '\n';
            }
            fprintf(stderr, "label file %ls written to disk\n", m_labelFileToWrite.c_str());
            m_labelFileToWrite.clear();
        }
        else if (!m_cachingWriter)
        {
            fprintf(stderr, "WARNING: file %ls NOT written to disk yet, will be written the first time the end of the entire dataset is found.\n", m_labelFileToWrite.c_str());
        }
    }
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void UCIFastReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
// Sample format below:
//# Parameter values for the reader
//reader=[
//  # reader to use
//  readerType=UCIFastReader
//  miniBatchMode=Partial
//  randomize=None
//  features=[
//    dim=784
//    start=1
//    file=c:\speech\mnist\mnist_test.txt
//  ]
//  labels=[
//    dim=1
//      start=0
//      file=c:\speech\mnist\mnist_test.txt
//      labelMappingFile=c:\speech\mnist\labels.txt
//      labelDim=10
//      labelType=Category
//  ]
//]
template <class ElemType>
template <class ConfigRecordType>
void UCIFastReader<ElemType>::InitFromConfig(const ConfigRecordType& readerConfig)
{
    //customize delimiter and decimal points.
    string customDelimiterStr = readerConfig(L"customDelimiter", "");
    char customDelimiter = customDelimiterStr == "" ? char(0) : customDelimiterStr[0];

    string customDecimalPointStr = readerConfig(L"customDecimalPoint", "");
    char customDecimalPoint = customDecimalPointStr == "" ? char(0) : customDecimalPointStr[0];

    m_parser = make_shared<UCIParser<ElemType, LabelType>>(customDelimiter, customDecimalPoint);

    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    // initialize the cache
    InitCache(readerConfig);

    // if we have a cache, no need to parse the text files...
    if (m_cachingReader)
        return;

    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;
    GetFileConfigNames(readerConfig, features, labels);
    if (features.size() > 0)
    {
        m_featuresName = features[0];
        if (!readerConfig.Exists(m_featuresName)) // BUGBUG: How can this ever fire? We wouldn't be able to get this name in the first place if it wasn't there.
            RuntimeError("features file not found, required in configuration: i.e. 'features=[file=c:\\myfile.txt;start=1;dim=123]'");
    }
    if (labels.size() > 0)
    {
        m_labelsName = labels[0];
    }
    bool hasLabels = readerConfig.Exists(m_labelsName);
    if (!hasLabels)
        fprintf(stderr, "Warning: labels are not specified.");

    const ConfigRecordType& configFeatures = readerConfig(m_featuresName.c_str(), ConfigRecordType::Record());
    const ConfigRecordType& configLabels = readerConfig(m_labelsName.c_str(), ConfigRecordType::Record());

    // determine label type desired
    std::wstring labelType;
    if (!hasLabels)
        labelType = L"none";
    else
        labelType = (wstring)configLabels(L"labelType", L"category");

    if (!EqualCI(labelType, L"none") && configFeatures(L"file", L"") != configLabels(L"file", L""))
        RuntimeError("features and label files must be the same file, use separate readers to define single use files");

    size_t vdim = configFeatures(L"dim");
    // string name = configFeatures.Name();            // TODO: Aaargh!!!
    size_t udim = configLabels(L"labelDim", (size_t) 0);

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = 0;
    m_labelIdMax = m_labelDim = 0;
    m_partialMinibatch = m_endReached = false;
    m_labelType = labelCategory;
    m_featureCount = vdim;
    m_readNextSample = 0;
    m_traceLevel = readerConfig(L"traceLevel", 0);
    m_parser->SetTraceLevel(m_traceLevel);

    m_prefetchEnabled = readerConfig(L"prefetch", false);
    // set the feature count to at least one (we better have one feature...)
    assert(m_featureCount != 0);

    if (readerConfig.Exists(L"randomize"))
    {
        string randomizeString = readerConfig(L"randomize");
        if (EqualCI(randomizeString, "none"))
        {
            m_randomizeRange = randomizeNone;
        }
        else if (EqualCI(randomizeString, "auto"))
        {
            m_randomizeRange = randomizeAuto;
        }
        else
        {
            m_randomizeRange = readerConfig(L"randomize");
        }
    }
    else
    {
        m_randomizeRange = randomizeAuto;
    }

    // determine if we partial minibatches are desired
    std::string minibatchMode(readerConfig(L"minibatchMode", "partial"));
    m_partialMinibatch = EqualCI(minibatchMode, "partial");

    // get start and dimensions for labels and features
    size_t startLabels = configLabels(L"start", (size_t) 0);
    size_t dimLabels = configLabels(L"dim", (size_t) 0);

    size_t startFeatures = configFeatures(L"start", (size_t) 0);
    size_t dimFeatures = configFeatures(L"dim", (size_t) 0);


    // convert to lower case for case insensitive comparison
    if (EqualCI(labelType, L"category"))
    {
        m_labelType = labelCategory;
    }
    else if (EqualCI(labelType, L"regression"))
    {
        m_labelType = labelRegression;
    }
    else if (EqualCI(labelType, L"none"))
    {
        m_labelType = labelNone;
        dimLabels = 0; // override for no labels
    }

    std::wstring file = configFeatures(L"file");
    if (m_traceLevel > 0)
        fprintf(stderr, "Reading UCI file %ls\n", file.c_str());

    // Simple heuristic to ensure buffer size and avoid breaking existing experiments.
    size_t bufSize = max(dimFeatures * 16, (size_t) 256 * 1024);
    m_parser->ParseInit(file.c_str(), startFeatures, dimFeatures, startLabels, dimLabels, bufSize);

    // if we have labels and labels are categorical values, we need a label Mapping file, it will be a file with one label per line
    if (m_labelType == labelCategory)
    {
        std::wstring labelPath = configLabels(L"labelMappingFile");
        if (fexists(labelPath))
        {
            // TODO: We use the old CNTK config reader for this. With BrainScript, we would have to parse the file locally here, which should be easy.
            ConfigArray arrayLabels;
            arrayLabels.LoadConfigFile(labelPath);
            for (int i = 0; i < arrayLabels.size(); ++i)
            {
                LabelType label = arrayLabels[i];
                m_mapIdToLabel[i] = label;
                m_mapLabelToId[label] = i;
            }
            m_labelIdMax = (LabelIdType) arrayLabels.size();
        }
        else
        {
            // only do label creation if we have the allow flag, should only be done as a separate command
            // to ensure that the label file will exist for verification step in training
            bool allowLabelCreation = readerConfig(L"allowMapCreation", false);
            if (allowLabelCreation)
                m_labelFileToWrite = labelPath;
            else
                RuntimeError("label mapping file %ls not found, can be created with a 'createLabelMap' command/action\n", labelPath.c_str());
        }

        // if the value they passed in as udim is not big enough, add something on
        if (udim < m_labelIdMax)
            udim = m_labelIdMax;
        m_labelDim = (LabelIdType)udim;
    }
    else
    {
        m_labelDim = (LabelIdType)dimLabels;
    }

    // if we know the size of the randomization now, resize, otherwise wait until we know the epochSize in StartMinibatchLoop()
    if (Randomize() && m_randomizeRange != randomizeAuto)
        m_randomordering.Resize(m_randomizeRange, m_randomizeRange);

    mOneLinePerFile = readerConfig(L"oneLinePerFile", false);
}

// InitCache - Initialize the caching reader if cache files exist, otherwise the writer
// readerConfig - reader configuration
template <class ElemType>
void UCIFastReader<ElemType>::InitCache(const ScriptableObjects::IConfigRecord& readerConfig)
{
    // check for a writer tag first (lets us know we are caching)
    if (readerConfig.Exists(L"writerType"))
        InvalidArgument("UCIFastReader: Caching ('writerType') is currently not supported for BrainScript.");
    // BrainScript cannot support this because of the manipulations of ConfigRecords, which the ScriptableObjects interface does not support (IConigRecords are immutable).
    // TODO: We could implement an overlay IConfigRecord implementation that fakes the two values that are being added to the interface.
    // BrainScript also cannot support this because a copy of the readerConfig is kept. IConfigRecords are not copyable.
    // TODO: We could copy the IConfigRecordPtr. That is allowed. Not trivial to do with template magic.
}
template <class ElemType>
void UCIFastReader<ElemType>::InitCache(const ConfigParameters& readerConfig)
{
    // first time we are called we cache our parameters
    // Note: This is a little ugly. It is an artifact of integrating with BrainScript. Since BrainScript does not allow to copy configs, I moved that copy code here into the ConfigParameters implementation.
    if (&readerConfig != &m_readerConfig)
        readerConfig.CopyTo(m_readerConfig);

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
                if (pair.second == sectionTypeCategoryLabel)
                {
                    m_labelsCategoryName = pair.first;
                }
                else if (pair.second == sectionTypeLabelMapping)
                {
                    m_labelsMapName = pair.first;
                }
            }
        }
    }
    catch (runtime_error err)
    {
        // In case caching reader/writer cannot be created, we gracefully fail over and disable caching.
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
UCIFastReader<ElemType>::~UCIFastReader()
{
    ReleaseMemory();
    delete m_cachingReader;
    delete m_cachingWriter;
}

// ReleaseMemory - release the memory footprint of UCIFastReader
// used when the caching reader is taking over
template <class ElemType>
void UCIFastReader<ElemType>::ReleaseMemory()
{
    m_featuresBuffer = NULL;
    m_labelsBuffer = NULL;
    m_labelsIdBuffer = NULL;
    m_featureData.clear();
    m_labelIdData.clear();
    m_labelData.clear();
}

//SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template <class ElemType>
void UCIFastReader<ElemType>::SetupEpoch()
{
    // if we are starting fresh (epoch zero and no data read), init everything
    // however if we are using cachingWriter, we need to know record count, so do that first
    if (m_epoch == 0 && m_totalSamples == 0 && m_cachingWriter != NULL)
    {
        m_readNextSample = m_epochStartSample = m_mbStartSample = 0;
        m_parser->SetFilePosition(0);
    }
    else // otherwise, position the read to start at the right location
    {
        // don't know the total number of samples yet, so count them
        if (m_totalSamples == 0)
        {
            if (m_traceLevel > 0)
                fprintf(stderr, "UCIFastReader: Starting at epoch %lu, counting lines to determine record count...\n", (unsigned long) m_epoch);
            m_parser->SetParseMode(ParseLineCount);
            m_totalSamples = m_parser->Parse(size_t(-1), NULL, NULL);
            m_parser->SetParseMode(ParseNormal);
            m_parser->SetFilePosition(0);
            m_mbStartSample = 0;
            UpdateDataVariables(0); // update all the variables since we read to the end...
            if (m_traceLevel > 0)
                fprintf(stderr, " %lu records found.\n", (unsigned long) m_totalSamples);
        }

        // make sure we are in the correct location for mid-dataset epochs
        size_t mbStartSample = m_epoch * m_epochSize;

        size_t fileRecord = m_totalSamples ? mbStartSample % m_totalSamples : 0;
        fprintf(stderr, "starting epoch %lu at record count %lu, and file position %lu\n", (unsigned long) m_epoch, (unsigned long) mbStartSample, (unsigned long) fileRecord);
        size_t currentFileRecord = m_mbStartSample % m_totalSamples;

        // reset the next read sample
        m_readNextSample = mbStartSample;
        if (currentFileRecord == fileRecord)
        {
            fprintf(stderr, "already there from last epoch\n");

            // we have a slight delima here, if we haven't determined the end of the file yet
            // and the user told us to find how many records are in the file, we can't distinguish "almost done"
            // with a file (a character away) and the middle of the file. So read ahead a record to see if it's there.
            bool endReached = m_endReached;
            if (!endReached)
            {
                if (!m_parser->HasMoreData())
                {
                    endReached = true;
                    UpdateDataVariables(mbStartSample);
                    assert(m_endReached);
                }
            }
            // move the read pointer to the end since we have everything already in memory.
            if (endReached && m_epochStartSample % m_totalSamples == fileRecord && m_featureData.size() >= m_epochSize * m_featureCount)
            {
                m_readNextSample = mbStartSample + m_epochSize;
                // write the label file here to make sure we do it somewhere. We know the entire dataset has been read at this point
                WriteLabelFile();
            }
        }
        // not the right position, need to get there
        else
        {
            // if we are already past the desired record, start at the beginning again
            if (currentFileRecord > fileRecord)
            {
                m_parser->SetFilePosition(0);
                currentFileRecord = 0;
            }
            fprintf(stderr, "reading from record %lu to %lu to be positioned properly for epoch\n", (unsigned long) currentFileRecord, (unsigned long) fileRecord);
            m_parser->SetParseMode(ParseLineCount);
            m_parser->Parse(fileRecord - currentFileRecord, NULL, NULL);
            m_parser->SetParseMode(ParseNormal);
            if (!m_labelFileToWrite.empty())
            {
                fprintf(stderr, "WARNING: file %ls NOT written to disk, label file will only be written when starting epochs at the beginning of the dataset\n", m_labelFileToWrite.c_str());
                m_labelFileToWrite.clear();
                RuntimeError("LabelMappingFile not provided in config, must be provided if not starting from epoch Zero (0)");
            }
        }
        m_epochStartSample = m_mbStartSample = mbStartSample;
    }
}

// utility function to round an integer up to a multiple of size
size_t RoundUp(size_t value, size_t size)
{
    return ((value + size - 1) / size) * size;
}

template <class ElemType>
void UCIFastReader<ElemType>::SetNumParallelSequences(const size_t sz)
{
    mRequestedNumParallelSequences = sz;
    if (mOneLinePerFile)
        m_mbSize = mRequestedNumParallelSequences;
};

//StartMinibatchLoop - Startup a minibatch loop
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
//   this value must be a multiple of mbSize, if it is not, it will be rounded up to one.
template <class ElemType>
void UCIFastReader<ElemType>::StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples /*= requestDataSize */)
{
    m_subsetNum = subsetNum;
    m_numSubsets = numSubsets;
    if (mOneLinePerFile)
        mbSize = mRequestedNumParallelSequences; // each file has only one observation, therefore the number of data to read is the number of files

    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory(); // free the memory used by the UCIFastReader
    }

    // if we are reading from the cache, do so now and return
    if (m_cachingReader)
    {
        m_cachingReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
        return;
    }

    // if we are reallocating bigger, release the original
    if (mbSize > m_mbSize)
    {
        m_featuresBuffer = NULL;
        m_labelsBuffer = NULL;
        m_labelsIdBuffer = NULL;
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
        if (!m_partialMinibatch)
            m_epochSize = RoundUp(requestedEpochSamples, mbSize);
        if (m_epochSize != requestedEpochSamples)
            fprintf(stderr, "epochSize rounded up to %d to fit an integral number of minibatches\n", (int) m_epochSize);
    }

    // set the randomization range for randomizationAuto
    // or if it's invalid less than the minibatch size, we need to make it at least minibatch size
    if (m_randomizeRange != randomizeNone)
    {
        if (m_epochSize != requestDataSize && m_randomizeRange == randomizeAuto)
        {
            m_randomizeRange = m_epochSize;
        }
        m_randomizeRange = max(m_randomizeRange, m_mbSize);
        if (m_randomizeRange != randomizeAuto)
        {
            if ((m_epochSize != requestDataSize && m_epochSize % m_randomizeRange != 0) || (m_randomizeRange % m_mbSize != 0))
                RuntimeError("randomizeRange must be an even multiple of mbSize and an integral factor of epochSize");
            m_randomordering.Resize(m_randomizeRange, m_randomizeRange);
        }
    }

    // we use epochSize, which might not be set yet, so use a default value for allocations if not yet set
    size_t epochSize = m_epochSize == requestDataSize ? 1000 : m_epochSize;
    m_epoch = epoch;
    m_mbStartSample = epoch * m_epochSize;

    // allocate room for the data
    m_featureData.reserve(m_featureCount * epochSize);
    if (m_labelType == labelCategory)
    { 
        m_labelIdData.reserve(epochSize);
        m_labelData.reserve(epochSize);
    }
    else if (m_labelType != labelNone)
        m_labelData.reserve(m_labelDim * epochSize);

    SetupEpoch();
}

// function to store the LabelType in an ElemType
// required for string labels, which can't be stored in ElemType arrays
template <class ElemType>
void UCIFastReader<ElemType>::StoreLabel(ElemType& labelStore, const LabelType& labelValue)
{
    labelStore = (ElemType) m_mapLabelToId[labelValue];
}

// GetMinibatch - Get the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool UCIFastReader<ElemType>::TryGetMinibatch(StreamMinibatchInputs& matrices)
{
    bool minibatchesRemaining = true;
    if (m_pendingAsyncGetMinibatch.valid())
    {
        // An async GetMinibatch is in flight. Wait for it to finish and swap
        // the contents of the m_prefetchedMatrices and parameter matrices
        minibatchesRemaining = m_pendingAsyncGetMinibatch.get();

        // Now swap the m_prefetchedMatrices and parameter matrices
        for (auto& iter : matrices)
        {
            if (m_prefetchMatrices.find(iter.first) == m_prefetchMatrices.end())
                LogicError("GetMinibatch: No matching prefetch matrix found for matrix named %ls.", iter.first.c_str());

            std::swap(matrices.GetInputMatrix<ElemType>(iter.first), m_prefetchMatrices.GetInputMatrix<ElemType>(iter.first)); // BUGBUG?: This swaps the matrix structures directly, messing with ownership. And are we sure it does not do deep copies for this?
            //Matrix<ElemType>* prefetchMatrix = m_prefetchMatrices[iter->first].get();
            //std::swap(*(iter->second), *prefetchMatrix);
        }
    }
    else
    {
        minibatchesRemaining = GetMinibatchImpl(matrices);

        // Allocate prefetch matrices if were firing an async minibatch prefetch
        if (minibatchesRemaining && m_prefetchEnabled)
        {
            // create a matrix for every output
            m_prefetchMatrices.clear();

            for (auto& iter : matrices)
                m_prefetchMatrices.AddInput(iter.first, make_shared<Matrix<ElemType>>(iter.second.matrix->GetDeviceId()), iter.second.pMBLayout, iter.second.sampleLayout);
        }
    }

    // Fire a new prefetch if there are any minibatches remaining
    if (minibatchesRemaining && m_prefetchEnabled)
    {
        Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(m_featuresName);
        int deviceId = features.GetDeviceId();
        m_pendingAsyncGetMinibatch = std::async(std::launch::async, [this, deviceId]()
        {
            // Set the device since this will execute on a new thread
            Matrix<ElemType>::SetDevice(deviceId);

            StreamMinibatchInputs prefetchMatrices;
            for (auto& iter : m_prefetchMatrices)
                prefetchMatrices.AddInput(iter.first, iter.second);
            // TODO: Why can we not just pass m_prefetchMatrices?

            return GetMinibatchImpl(prefetchMatrices);
        });
    }

    return minibatchesRemaining;
}

// GetMinibatchImpl - The actual implementation of getting the next minibatch (features and labels)
// matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
//             [out] each matrix resized if necessary containing data.
// returns - true if there are more minibatches, false if no more minibatchs remain
template <class ElemType>
bool UCIFastReader<ElemType>::GetMinibatchImpl(StreamMinibatchInputs& matrices)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetMinibatch(matrices);
    }
    // get the features array
    if (matrices.find(m_featuresName) == matrices.end())
        RuntimeError("Features matrix not found in config file, there should be a node '%ls' in the network.", m_featuresName.c_str());

    Matrix<ElemType>& features = matrices.GetInputMatrix<ElemType>(m_featuresName);

    // get out if they didn't call StartMinibatchLoop() first  --BUGBUG: We should throw in that case.
    if (m_mbSize == 0)
        return false;

    // check to see if we have changed epochs, if so we are done with this one.
    if (m_mbStartSample / m_epochSize != m_epoch)
        return false;

    bool randomize = Randomize();
    bool moreData = EnsureDataAvailable(m_mbStartSample);

    // figure which sweep of the randomization we are on
    size_t epochSample = m_mbStartSample % m_epochSize; // where the minibatch starts in this epoch
    // size_t samplesExtra = m_totalSamples % m_epochSize; // extra samples at the end of an epoch
    // size_t epochsDS = (m_totalSamples+m_epochSize-1)/m_epochSize; // how many epochs per dataset
    size_t randomizeSet = randomize ? RandomizeSweep(m_mbStartSample) : 0;
    const auto& tmap = m_randomordering(randomizeSet);
    size_t epochEnd = m_epochSize;
    size_t recordStart = m_totalSamples ? m_mbStartSample % m_totalSamples : m_mbStartSample;

    // actual size is either what requested, or total number of samples read so far
    size_t actualmbsize = min(m_totalSamples, m_mbSize); // it may still return less if at end of sweep

    // check for an odd sized last minibatch
    if (epochSample + actualmbsize > epochEnd)
    {
        actualmbsize = epochEnd - epochSample;
    }

    // hit the end of the dataset, we should only get here in "one=pass mode"
    if (!moreData)
    {
        // make sure we take into account hitting the end of the dataset (not wrapping around)
        actualmbsize = min(m_totalSamples - recordStart, actualmbsize);
    }

    if (m_featuresBuffer == NULL)
    {
        m_featuresBuffer = AllocateIntermediateBuffer(features.GetDeviceId(), m_mbSize * m_featureCount);
        memset(m_featuresBuffer.get(), 0, sizeof(ElemType) * m_mbSize * m_featureCount);
    }

    if (m_labelsBuffer == NULL)
    {
        if (m_labelType == labelCategory)
        {
            m_labelsBuffer = AllocateIntermediateBuffer(features.GetDeviceId(), m_labelDim * m_mbSize);
            m_labelsIdBuffer = std::shared_ptr<LabelIdType>(new LabelIdType[m_mbSize], [](LabelIdType* p)
                                                            {
                                                                delete[] p;
                                                            });
        }
        else if (m_labelType != labelNone)
        {
            m_labelsBuffer = AllocateIntermediateBuffer(features.GetDeviceId(), m_labelDim * m_mbSize);
            m_labelsIdBuffer = NULL;
        }
    }

    if (m_labelType == labelCategory)
    {
        memset(m_labelsBuffer.get(), 0, sizeof(ElemType) * m_labelDim * actualmbsize);
        memset(m_labelsIdBuffer.get(), 0, sizeof(LabelIdType) * actualmbsize);
    }
    else if (m_labelType != labelNone)
    {
        memset(m_labelsBuffer.get(), 0, sizeof(ElemType) * m_labelDim * actualmbsize);
    }

    if (actualmbsize > 0)
    {
        // loop through and copy data to matrix
        int j = 0; // vector of vectors of feature data
        // determine randomization base index
        size_t randBase = 0; // (keep compiler happy)
        if (randomize)
            randBase = epochSample - epochSample % m_randomizeRange;

        // loop through all the samples
        for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
        {
            // pick the right sample with randomization if desired
            size_t jRand = randomize ? (randBase + tmap[jSample % m_randomizeRange]) : jSample;
            jRand %= m_epochSize; // BUGBUG: this will make it randomize only inside m_epochSize which is not enough 

            // vector of feature data goes into matrix column
            memcpy(&m_featuresBuffer.get()[j * m_featureCount], &m_featureData[jRand * m_featureCount], sizeof(ElemType) * m_featureCount);

            if (m_labelType == labelCategory)
            {
                m_labelsBuffer.get()[j * m_labelDim + m_labelIdData[jRand]] = (ElemType) 1;
                m_labelsIdBuffer.get()[j] = m_labelIdData[jRand];
            }
            else if (m_labelType != labelNone)
            {
                for (size_t ii = 0; ii < m_labelDim; ii++)
                {
                    ElemType v = (ElemType)atof(m_labelData[jRand*m_labelDim + ii].c_str());
                    m_labelsBuffer.get()[j*m_labelDim + ii] = v;
                }
                }
            }
        }

    // There may be multiple parallel trainers reading at the same time in which case
    // we will slice the data to only return the share of the current trainer's subset
    size_t currSubsetStartCol = (actualmbsize * m_subsetNum) / m_numSubsets;
    size_t currSubsetEndCol = (actualmbsize * (m_subsetNum + 1)) / m_numSubsets;
    size_t currSubsetSize = currSubsetEndCol - currSubsetStartCol;
    // create the respective MBLayout
    // Every sample is returned as a sequence of 1 frame.
    m_pMBLayout->InitAsFrameMode(currSubsetSize);

    // if we are writing out to the caching writer, do it now
    if (m_cachingWriter && (m_subsetNum == 0))
    {
        map<std::wstring, void*, nocase_compare> writeBuffer;
        writeBuffer[m_featuresName] = m_featuresBuffer.get();
        if (m_labelType == labelCategory)
        {
            writeBuffer[m_labelsName] = m_labelsIdBuffer.get();
            if (!m_labelsCategoryName.empty())
                writeBuffer[m_labelsCategoryName] = m_labelsBuffer.get();
        }
        else if (m_labelType != labelNone)
        {
            writeBuffer[m_labelsName] = m_labelsBuffer.get();
        }

        // write out the data, on a second pass compute statistics as needed
        bool moreToWrite = m_cachingWriter->SaveData(m_mbStartSample, writeBuffer, actualmbsize, m_totalSamples, 0);

        // done writing
        if (!moreToWrite)
        {
            // write out the mapping table as necessary
            if (m_labelType == labelCategory && !m_labelsMapName.empty())
            {
                m_cachingWriter->SaveMapping(m_labelsMapName, m_mapIdToLabel);
            }

            WriteLabelFile();

            // now close the cache writer
            delete m_cachingWriter;
            m_cachingWriter = NULL;
        }
    }

    // advance to the next minibatch
    m_mbStartSample += actualmbsize;

    // if they don't want partial minibatches, skip data transfer and return
    if (actualmbsize < m_mbSize && !m_partialMinibatch || actualmbsize == 0 || currSubsetSize == 0) // no records found (end of minibatch)
    {
        return false;
    }

    // now transfer to the GPU as needed
    features.SetValue(m_featureCount, currSubsetSize, features.GetDeviceId(), m_featuresBuffer.get() + (m_featureCount * currSubsetStartCol), matrixFlagNormal);
    if (m_labelType != labelNone)
    {
        if (matrices.HasInput(m_labelsName))
        {
            auto& labels = matrices.GetInputMatrix<ElemType>(m_labelsName);
            labels.SetValue(m_labelDim, currSubsetSize, labels.GetDeviceId(), m_labelsBuffer.get() + (m_labelDim * currSubsetStartCol), matrixFlagNormal);
        }
    }

    // we read some records, so process them
    return true;
}

// GetLabelMapping - Gets the label mapping from integer index to label type
// returns - a map from numeric datatype to native label type
template <class ElemType>
const std::map<IDataReader::LabelIdType, IDataReader::LabelType>& UCIFastReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetLabelMapping(sectionName);
    }
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run
template <class ElemType>
void UCIFastReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<LabelIdType, LabelType>& labelMapping)
{
    if (m_cachingReader)
    {
        RuntimeError("Cannot set mapping table when the caching reader is being used");
    }
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for (std::pair<unsigned, LabelType> var : labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
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
bool UCIFastReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
    }
    RuntimeError("GetData not supported in UCIFastReader");
}

template <class ElemType>
bool UCIFastReader<ElemType>::DataEnd()
{
    if (m_cachingReader)
        return m_cachingReader->DataEnd();
    else
        return true;
}

template <class ElemType>
unique_ptr<CUDAPageLockedMemAllocator>& UCIFastReader<ElemType>::GetCUDAAllocator(int deviceID)
{
    if (m_cudaAllocator != nullptr)
    {
        if (m_cudaAllocator->GetDeviceId() != deviceID)
        {
            m_cudaAllocator.reset(nullptr);
        }
    }

    if (m_cudaAllocator == nullptr)
    {
        m_cudaAllocator.reset(new CUDAPageLockedMemAllocator(deviceID));
    }

    return m_cudaAllocator;
}

template <class ElemType>
std::shared_ptr<ElemType> UCIFastReader<ElemType>::AllocateIntermediateBuffer(int deviceID, size_t numElements)
{
    if (deviceID >= 0)
    {
        // Use pinned memory for GPU devices for better copy performance
        size_t totalSize = sizeof(ElemType) * numElements;
        return std::shared_ptr<ElemType>((ElemType*) GetCUDAAllocator(deviceID)->Malloc(totalSize), [this, deviceID](ElemType* p)
                                         {
                                             this->GetCUDAAllocator(deviceID)->Free((char*) p);
                                         });
    }
    else
    {
        return std::shared_ptr<ElemType>(new ElemType[numElements], [](ElemType* p)
                                         {
                                             delete[] p;
                                         });
    }
}

// instantiate all the combinations we expect to be used
template class UCIFastReader<double>;
template class UCIFastReader<float>;
} } }
