//
// <copyright file="SequenceReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// SequenceReader.cpp : Defines the exported functions for the DLL application.
//


#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "SequenceReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template<class ElemType>
bool SequenceReader<ElemType>::ReadRecord(size_t readSample)
{
    return false; // not used
}

// RecordsToRead - Determine number of records to read to populate record buffers
// mbStartSample - the starting sample from which to read
// tail - we are checking for possible remainer records to read (default false)
// returns - true if we have more to read, false if we hit the end of the dataset
template<class ElemType>
size_t SequenceReader<ElemType>::RecordsToRead(size_t mbStartSample, bool tail)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    epochSample %= m_epochSize;

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
template<class ElemType>
/*IDataReader<ElemType>::LabelIdType*/ unsigned SequenceReader<ElemType>::GetIdFromLabel(const std::string& labelValue, LabelInfo& labelInfo)
{
    auto found = labelInfo.mapLabelToId.find(labelValue);

    // not yet found, add to the map
    if (found == labelInfo.mapLabelToId.end())
    {
        labelInfo.mapLabelToId[labelValue] = labelInfo.idMax;
        labelInfo.mapIdToLabel[labelInfo.idMax] = labelValue;
        found = labelInfo.mapLabelToId.find(labelValue);
        labelInfo.idMax++;
    }
    return found->second;
}

template<class ElemType>
/*IDataReader<ElemType>::LabelIdType*/ bool SequenceReader<ElemType>::CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned & labelId)
{
    auto found = labelInfo.mapLabelToId.find(labelValue);

    // not yet found, add to the map
    if (found == labelInfo.mapLabelToId.end())
    {
        return false; 
    }
    labelId = found->second;
    return true; 
}

// EnsureDataAvailable - Read enough lines so we can request a minibatch starting as requested
// mbStartSample - the starting sample we are starting with
// endOfDataCheck - check if we are at the end of the dataset (no wraparound)
// returns - true if we have more to read, false if we hit the end of the dataset
template<class ElemType>
bool SequenceReader<ElemType>::EnsureDataAvailable(size_t mbStartSample, bool endOfDataCheck)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    bool moreToRead = true;

    size_t numberToRead = RecordsToRead(mbStartSample);

    // check to see if we have the proper records read already
    if (m_readNextSample >= mbStartSample+numberToRead && mbStartSample >= m_epochStartSample)
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
    LabelInfo& labelInfo = m_labelInfo[nextWord?labelInfoIn:labelInfoOut];

    //if (m_labelIdData.size() > epochSample)
    //{
    //    m_labelIdData.resize(epochSample);
    //    m_labelData.resize(epochSample*labelInfo.dim);
    //}

    // see how many we already read
    int sequencesRead = 0;
    std::vector<ElemType> featureTemp;
    std::vector<LabelType> labelTemp;
    std::vector<SequencePosition> seqPos;
    do
    {
        int numRead = m_parser.Parse(CACHE_BLOG_SIZE, &labelTemp, &featureTemp, &seqPos);
        moreToRead = (numRead != 0);

        // translate from the sparse parsed data format to the to the training format data
        int label = 0;
        int feature = 0;
        bool bSentenceStart = false;
        SequencePosition sposLast = SequencePosition(0,0,seqFlagNull);
        for (int seq = 0; seq < numRead; seq++)
        {
            // check 
            SequencePosition spos = seqPos[seq];
            if (spos.labelPos == sposLast.labelPos && spos.numberPos == sposLast.numberPos)
                continue;
            sposLast = spos;

            bSentenceStart = true; 

            // loop through the labels for this entry
            while (label < spos.labelPos)  /// need to minus one since 
            {

                // labelIn should be a category label 
                LabelType labelValue = labelTemp[label++];

                if (trim(labelValue).size() == 0)
                    continue; // empty input

                // check for end of sequence marker
                if (!bSentenceStart && (!_strcmpi(labelValue.c_str(), m_labelInfo[labelInfoIn].endSequence.c_str()) || ((label - 1 )% m_mbSize == 0) ))
                {
                    // ignore those cases where $</s> is put in the begining, because those are used for initialization purpose
                    spos.flags |= seqFlagStopLabel;
                    sequencesRead++;

                    // create the seqence table
                    m_sequence.push_back(epochSample);
                    if ((m_sequence.size() == 1 ? epochSample : epochSample - m_sequence[m_sequence.size()-2]) > m_mbSize)
                    {
                        fprintf(stderr, "read sentence length is longer than the minibatch size. should be smaller. increase the minibatch size to at least %d", epochSample);
                        Error("read sentence length is longer than the minibatch size. should be smaller. increase the minibatch size to at least %d", epochSample);
                    }

                    if (!_strcmpi(labelValue.c_str(), m_labelInfo[labelInfoIn].endSequence.c_str()))
                        continue; /// ignore sentence ending
                }

                // to-do, should ignore <s>, check the sentence ending is </s> 
                // need to remove <s> from the training set
                // allocate and initialize the next chunck of featureData
                if (labelIn.type == labelCategory)
                {
                    LabelIdType index = GetIdFromLabel(labelValue, labelIn);

                    // use the found value, and set the appropriate location to a 1.0
                    assert(labelIn.dim > index); // if this goes off labelOut dimension is too small
                    m_featureData.push_back(index);
                }
                else
                {
                    Error("Input label expected to be a category label");
                }

                // if we have potential features
                if (m_featureDim > 0)
                {
                    Error("to-do. Assume sparse input feature. need to change the code from dense matrix");
                    // move the position up to the start of the additional features section
/*                    pos += labelIn.dim;
                    assert(pos + m_featureDim == m_featureData.size());
                    // this has to be an even number, a pair of index and value
                    if  ((spos.numberPos&1) != 0)
                        Error("Features must be specified in pairs (index:value). Invalid features for label '%s'\n", labelValue);
                
                    while (feature < spos.numberPos)
                    {
                        int index = (int)featureTemp[feature++];
                        if (index < 0 || index >= m_featureDim)
                            Error("Invalid feature index: %d for label '%s', feature max dimension = %lld\n", index, labelValue, m_featureDim);

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
                    labelValue = labelTemp[label];
                    if (!_strcmpi(labelValue.c_str(), m_labelInfo[labelInfoIn].endSequence.c_str()))
                    {
                        labelValue = labelInfo.endSequence;
                    }
                }
                else
                {
                    Error("Invalid output label type, expected Category, or Next Word");
                }

                // get the ID from the label
                LabelIdType id = GetIdFromLabel(labelValue, labelInfo);
                m_labelIdData.push_back(id);

                m_readNextSample++;
                epochSample++;
                if (!m_endReached)
                    m_totalSamples++;   // add to the total number of records in the dataset

                bSentenceStart = false;
            }

            {
                // check if the reading is right
                int jEnd = m_labelIdData.size() - 1; 
                LabelIdType index ;
                if (CheckIdFromLabel(labelInfo.endSequence, labelInfo, index) == false)
                    Error("cannot find sentence begining label");

                if (m_labelIdData[jEnd] != index )
                     /// for language model, the first word/letter has to be <s>
                    Error("SequenceReader: the last letter/word of a batch has to be the sentence ending symbol");
            }

        }

        m_readNextSampleLine += numRead;
    } 
    while (sequencesRead < 1 && moreToRead); // we need to read at least one sequence or have no more data

    // if we read to the end, update appropriate variables
    if (!moreToRead)
    {
        UpdateDataVariables();
    }

    // if there more to read 
    return moreToRead;
}

// UpdateDataVariables - Update variables that depend on the dataset being completely read
template<class ElemType>
void SequenceReader<ElemType>::UpdateDataVariables()
{
    // if we haven't been all the way through the file yet
    if (!m_endReached)
    {
        // get the size of the dataset
        assert(m_totalSamples*m_featureCount >= m_featureData.size());

        // if they want us to determine epoch size based on dataset size, do that
        if (m_epochSize == requestDataSize)
        {
            m_epochSize = m_totalSamples;
        }

        WriteLabelFile();

        // we got to the end of the dataset
        m_endReached = true;
    }

    // update the label dimension if it is not big enough, need it here because m_labelIdMax get's updated in the processing loop (after a read)
    for (int index = labelInfoMin; index < labelInfoMax; ++index)
    {
        if (m_labelInfo[index].type == labelCategory && m_labelInfo[index].idMax > m_labelInfo[index].dim)
            m_labelInfo[index].dim = m_labelInfo[index].idMax;  // update the label dimensions if different
    }
}

template<class ElemType>
void SequenceReader<ElemType>::WriteLabelFile()
{
    // update the label dimension if it is not big enough, need it here because m_labelIdMax get's updated in the processing loop (after a read)
    for (int index = labelInfoMin; index < labelInfoMax; ++index)
    {
        LabelInfo& labelInfo = m_labelInfo[index];

        // write out the label file if they don't have one
        if (!labelInfo.fileToWrite.empty())
        {
            if (labelInfo.mapIdToLabel.size() > 0)
            {
                File labelFile(labelInfo.fileToWrite, fileOptionsWrite | fileOptionsText);
                for (int i=0; i < labelInfo.mapIdToLabel.size(); ++i)
                {
                    labelFile << labelInfo.mapIdToLabel[i] << '\n';
                }
                labelInfo.fileToWrite.clear();
            }
            else if (!m_cachingWriter)
            {
                fprintf(stderr, "WARNING: file %ws NOT written to disk, label files only written when starting at epoch zero!", labelInfo.fileToWrite.c_str());
            }
        }
    }
}

template<class ElemType>
void SequenceReader<ElemType>::LoadLabelFile(const std::wstring &filePath, std::vector<LabelType>& retLabels)
{
    File file(filePath, fileOptionsRead);

    // initialize with file name
    std::string path = msra::strfun::utf8(filePath);
    auto location = path.find_last_of("/\\");
    if (location != npos)
        path = path.substr(location+1);
    
    // read the entire file into a string
    size_t fileLength = file.Size();
    string str;
    retLabels.resize(0);
    while (!file.IsEOF())
    {
        file.GetLine(str);

        // check for a comment line
        string::size_type pos = str.find_first_not_of(" \t");
        if (pos != -1)
        {
            retLabels.push_back((LabelType)trim(str));
        }
    }
}


// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType>
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
template<class ElemType>
void SequenceReader<ElemType>::Init(const ConfigParameters& readerConfig)
{
    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    // NOTE: probably want to re-enable at some point

    // initialize the cache
    //InitCache(readerConfig);
    //m_readerConfig = readerConfig;

    //// if we have a cache, no need to parse the test files...
    //if (m_cachingReader)
    //    return;

    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;
    GetFileConfigNames(readerConfig, features, labels);
    if (features.size() > 0)
    {
        m_featuresName = features[0];
    }

    if (labels.size() == 2)
    {
        for (int index = labelInfoMin; index < labelInfoMax; ++index)
        {
            m_labelsName[index] = labels[index];
        }
    }
    else
        Error("two label definitions (in and out) required for Sequence Reader");

    ConfigParameters featureConfig = readerConfig(m_featuresName,"");
    ConfigParameters labelConfig[2] = {readerConfig(m_labelsName[0],""),readerConfig(m_labelsName[1],"")};

    class_size = 0;
    m_featureDim = featureConfig("dim");
    for (int index = labelInfoMin; index < labelInfoMax; ++index)
    {
        m_labelInfo[index].idMax = 0; 
        m_labelInfo[index].beginSequence = labelConfig[index]("beginSequence", "");
        m_labelInfo[index].endSequence = labelConfig[index]("endSequence", "");

        // determine label type desired
        std::string labelType(labelConfig[index]("labelType","Category"));
        if (labelType == "Category")
        {
            m_labelInfo[index].type = labelCategory;
        }
        else if (labelType == "NextWord")
        {
            // in this case, it's all identical to the Input labels, except the data type
            m_labelInfo[index].type = labelNextWord;
            m_labelInfo[index].dim = m_labelInfo[labelInfoIn].dim;
        }
        else if (labelType == "None")
        {
            m_labelInfo[index].type = labelNone;
            m_labelInfo[index].dim = 0;   // override for no labels
        }

        // if we have labels, we need a label Mapping file, it will be a file with one label per line
        if (m_labelInfo[index].type != labelNone)
        {
            std::wstring wClassFile = readerConfig("wordclass", "");
            nwords = labelConfig[index]("labelDim");
            if (wClassFile != L""){
                ReadClassInfo(wClassFile  , false);
            }

            std::vector<string> arrayLabels;
            std::wstring labelPath = labelConfig[index]("labelMappingFile");
            if (fexists(labelPath))
            {
                LoadLabelFile(labelPath, arrayLabels);
                for (int i=0; i < arrayLabels.size(); ++i)
                {
                    LabelType label = arrayLabels[i];
                    m_labelInfo[index].mapIdToLabel[i] = label;
                    m_labelInfo[index].mapLabelToId[label] = i;
                }
                m_labelInfo[index].idMax = (LabelIdType)arrayLabels.size();
                m_labelInfo[index].mapName = labelPath;
            }
            else
            {
                if (wClassFile != L""){
                    ReadClassInfo(wClassFile  , false);
                    int iMax = -1, i; 
                    for (auto ptr = word4idx.begin(); ptr != word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first; 
                        i = ptr->second; 
                        iMax = max(i, iMax);
                        m_labelInfo[index].mapIdToLabel[i] = label;
                        m_labelInfo[index].mapLabelToId[label] = i;
                    }
                    m_labelInfo[index].idMax = (LabelIdType)(iMax+1);

                    OrganizeClass();

                }
                m_labelInfo[index].mapName = labelPath;

                m_labelInfo[index].fileToWrite = labelPath;
            }
        }

        m_labelInfo[index].dim = labelConfig[index]("labelDim");

        // update dimension if the file says it's bigger
        if (m_labelInfo[index].dim < m_labelInfo[index].idMax)
        {
            m_labelInfo[index].dim = m_labelInfo[index].idMax;
        }
    }

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = m_seqIndex = 0;
    m_endReached = false;
    m_readNextSampleLine = 0;
    m_readNextSample = 0;
    m_traceLevel = readerConfig("traceLevel","0");
    m_parser.SetTraceLevel(m_traceLevel);

    if (readerConfig.Exists("randomize"))
    {
        string randomizeString = readerConfig("randomize");
        if (randomizeString == "None")
        {
            ;
        }
        else if (randomizeString == "Auto")
        {
            ;
        }
        else
        {
            ;//readerConfig("randomize");
        }
    }
    else
    {
        ; //randomizeAuto;
    }

    // The input data is a combination of the label Data and extra feature dims together
//    m_featureCount = m_featureDim + m_labelInfo[labelInfoIn].dim;
    m_featureCount = 1; 

    std::wstring m_file = readerConfig("file");
    if (m_traceLevel > 0)
        fprintf(stderr, "reading sequence file %ws\n", m_file.c_str());

    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseInit(m_file.c_str(), m_featureDim, labelIn.dim, labelOut.dim, labelIn.beginSequence, labelIn.endSequence, labelOut.beginSequence, labelOut.endSequence);
}

template<class ElemType>
void SequenceReader<ElemType>::ReadWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) {
        ch=fgetc(fin);

        if (ch==13) continue;

        if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
            if (a>0) {
                if (ch=='\n') ungetc(ch, fin);
                break;
            }

            if (ch=='\n') {
                strcpy_s(word, strlen("</s>"), (char *)"</s>");
                return;
            }
            else continue;
        }

        word[a]=(char)ch;
        a++;

        if (a>=MAX_STRING) {
            //printf("Too long word found!\n");   //truncate too long words
            a--;
        }
    }
    word[a]=0;
}

template<class ElemType>
void SequenceReader<ElemType>::ReadClassInfo(const wstring & vocfile, bool flatten) 
{
    char strFileName[MAX_STRING];
    char stmp[MAX_STRING];
    string strtmp; 
    size_t sz;
    int cnt, clsidx, widx = 0, b;
    class_size  = 0;

    wcstombs_s(&sz, strFileName, 2048, vocfile.c_str(), vocfile.length());

    FILE * vin;
    vin = fopen(strFileName, "rt") ;

    if (vin == nullptr)
    {
        Error("cannot open word class file");
    }
    for (int a = 0; a < nwords; a++)
    {
        fscanf_s(vin, "%6d\t%10d\t", &b, &cnt);
        ReadWord(stmp, vin);
        fscanf_s(vin, "%d\t\n", &clsidx);
        strtmp = stmp;
        idx4cnt[b] = cnt;
        word4idx[strtmp] = b;
        idx4word[b]= strtmp;
        
        idx4class[b] = clsidx;
        class_size = max(class_size, clsidx);
    }
    fclose(vin);

    class_size ++;
}

// InitCache - Initialize the caching reader if cache files exist, otherwise the writer
// readerConfig - reader configuration
template<class ElemType>
void SequenceReader<ElemType>::InitCache(const ConfigParameters& readerConfig)
{
    // check for a writer tag first (lets us know we are caching)
    if (!readerConfig.Exists("writerType"))
        return;

    // first try to open the binary cache
    bool found = false;
    try
    {
        // TODO: need to go down to all levels, maybe search for sectionType
        ConfigArray filesList(',');
        vector<std::wstring> names;
        if (readerConfig.Exists("wfile"))
        {
            filesList.push_back(readerConfig("wfile"));
            if (fexists(readerConfig("wfile")))
                found = true;
        }
        FindConfigNames(readerConfig, "wfile", names);
        for (auto name : names)
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
            m_cachingReader = new DataReader<ElemType>(config);
        }
        else
        {
            m_cachingWriter = new DataWriter<ElemType>(readerConfig);

            // now get the section names for map and category types
            std::map<std::wstring, SectionType, nocase_compare> sections;
            m_cachingWriter->GetSections(sections);
            for (auto pair : sections)
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
        fprintf(stderr,"Error attemping to create Binary%s\n%s\n",found?"Reader":"Writer",err.what());
        delete m_cachingReader;
        m_cachingReader = NULL;
        delete m_cachingWriter;
        m_cachingWriter = NULL;
    }
    catch (...)
    {
        // if there is any error, just get rid of the object
        fprintf(stderr,"Error attemping to create Binary%s\n",found?"Reader":"Writer");
        delete m_cachingReader;
        m_cachingReader = NULL;
        delete m_cachingWriter;
        m_cachingWriter = NULL;
    }
}

// destructor - virtual so it gets called properly 
template<class ElemType>
SequenceReader<ElemType>::~SequenceReader()
{
    ReleaseMemory();
    delete m_cachingReader;
    delete m_cachingWriter;
}

// ReleaseMemory - release the memory footprint of SequenceReader
// used when the caching reader is taking over
template<class ElemType>
void SequenceReader<ElemType>::ReleaseMemory()
{
    if (m_featuresBuffer!=NULL)
        delete[] m_featuresBuffer;
    m_featuresBuffer=NULL;
    if (m_labelsBuffer!=NULL)
        delete[] m_labelsBuffer;
    m_labelsBuffer=NULL;
    if (m_labelsIdBuffer!=NULL)
        delete[] m_labelsIdBuffer;
    m_labelsIdBuffer=NULL;
    m_featureData.clear();
    m_labelIdData.clear();
    m_labelData.clear();
    m_sequence.clear();
}

//SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template<class ElemType>
void SequenceReader<ElemType>::SetupEpoch()
{
    // if we are starting fresh (epoch zero and no data read), init everything
    // however if we are using cachingWriter, we need to know record count, so do that first
    if (m_epoch == 0 && m_totalSamples == 0 && m_cachingWriter == NULL)
    {
        m_readNextSampleLine = m_readNextSample = m_epochStartSample = m_mbStartSample = m_seqIndex = 0;
        m_parser.SetFilePosition(0);
    }
    else  // otherwise, position the read to start at the right location
    {
        m_seqIndex = 0;
        // don't know the total number of samples yet, so count them
        if (m_totalSamples == 0)
        {
            if (m_traceLevel > 0)
                fprintf(stderr, "starting at epoch %d parsing all data to determine record count\n", m_epoch);
            // choose a large number to read
            m_parser.SetFilePosition(0);
            m_mbStartSample = 0;
            while (EnsureDataAvailable(m_mbStartSample))
            {
                m_mbStartSample = m_totalSamples;
                m_seqIndex = m_sequence.size();
            }
            if (m_traceLevel > 0)
                fprintf(stderr, "\n %lld records found\n", m_totalSamples);
        }
        m_seqIndex = 0;

        // we have a slight delima here, if we haven't determined the end of the file yet
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

template<class ElemType>
void SequenceReader<ElemType>::LMSetupEpoch()
{
    m_readNextSampleLine = m_readNextSample = m_epochStartSample = m_mbStartSample = m_seqIndex = 0;
}

// utility function to round an integer up to a multiple of size
size_t RoundUp(size_t value, size_t size) 
{
    return ((value + size -1)/size)*size;
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of Samples, etc.)
//     NOTE: for sequence data, this will be the MAX size of a sequence, as every sequence could be a different length
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType>
void SequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory();    // free the memory used by the SequenceReader
    }

    // if we are reading from the cache, do so now and return
    if (m_cachingReader)
    {
        m_cachingReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
        return;
    } 

    if (m_featuresBuffer==NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];
        m_featuresBuffer = new ElemType[mbSize*labelInfo.dim];
        memset(m_featuresBuffer,0,sizeof(ElemType)*mbSize*labelInfo.dim);
    }

    if (m_labelsBuffer==NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];
        if (labelInfo.type == labelCategory)
        {
            m_labelsBuffer = new ElemType[labelInfo.dim*mbSize];
            memset(m_labelsBuffer,0,sizeof(ElemType)*labelInfo.dim*mbSize);
            m_labelsIdBuffer = new IDataReader<ElemType>::LabelIdType[mbSize];
            memset(m_labelsIdBuffer,0,sizeof(IDataReader<ElemType>::LabelIdType)*mbSize);
        }
        else if (labelInfo.type != labelNone)
        {
            m_labelsBuffer = new ElemType[mbSize];
            memset(m_labelsBuffer,0,sizeof(ElemType)*mbSize);
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
    size_t epochSize = m_epochSize == requestDataSize?1000:m_epochSize;
    m_epoch = epoch;
    m_mbStartSample = epoch*m_epochSize;

    // allocate room for the data
    m_featureData.reserve(m_featureCount*epochSize);
    if (m_labelInfo[labelInfoOut].type == labelCategory)
        m_labelIdData.reserve(epochSize);
    else if (m_labelInfo[labelInfoOut].type != labelNone)
        m_labelData.reserve(epochSize);
    m_sequence.reserve(m_seqIndex); // clear out the sequence array
    /// this is too complicated for LM 
    // SetupEpoch(); 
    /// use the LMSetupEpoch() instead
    LMSetupEpoch();

    m_clsinfoRead = false; 
    m_idx2clsRead = false; 

    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseReset(); 

}

template<class ElemType>
bool SequenceReader<ElemType>::DataEnd(EndDataType endDataType)
{
    bool ret = false;
    switch (endDataType)
    {
    case endDataNull:
        assert(false);
        break;
    case endDataEpoch:
        ret = m_sequence.size() > 0 && m_mbStartSample > m_sequence[m_sequence.size()-1];
        break;
    case endDataSet:
        ret = !EnsureDataAvailable(m_mbStartSample);
        break;
    case endDataSentence:  // for fast reader each minibatch is considered a "sentence", so always true
        ret = SentenceEnd();
        break;
    }
    return ret;
}


template<class ElemType>
bool SequenceReader<ElemType>::SentenceEnd()
{
    // this is after getMinibatch size, which has increased m_seqIndex by 1
    // so the real index is m_seqIndex - 1; 
    int seqIndex = m_seqIndex - 1; 

    // now get the labels
    const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];

    size_t actualmbsize = 0;

    // figure out the size of the next sequence
    if (seqIndex > 0)
    {
        actualmbsize = m_sequence[seqIndex] - m_sequence[seqIndex-1];   
    }
    else
    {
        actualmbsize = m_sequence[0];
    }

    if (actualmbsize < m_mbSize)
        return true;

    size_t jEnd = m_sequence[seqIndex]-1;
         
    if (labelInfo.type == labelCategory)
    {
        LabelIdType index ;
        if (CheckIdFromLabel(labelInfo.endSequence, labelInfo, index) == false)
            Error("cannot find sentence begining label");

        if (m_labelIdData[jEnd] == index )
            return true; 
        else 
            return false; 
    }
    return false; 
}

template<class ElemType>
void SequenceReader<ElemType>::GetLabelOutput(std::map<std::wstring, Matrix<ElemType>*>& matrices, 
                                              size_t m_mbStartSample, size_t actualmbsize)
{
    size_t j = 0;
    Matrix<ElemType>* labels = matrices[m_labelsName[labelInfoOut]];
    if (labels == nullptr) return;
    
    labels->Resize(nwords + class_size, actualmbsize, false);
        
    for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
    {
        // pick the right sample with randomization if desired
        size_t jRand = jSample;
         
        int    wrd = m_labelIdData[jRand];
        int    clsidx = idx4class[wrd]; 
        
        labels->SetValue(wrd, j, 1); 

        if (class_size > 0)
            labels->SetValue(nwords + clsidx, j, 1); 
    }

}

template<class ElemType>
void SequenceReader<ElemType>::GetInputToClass(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    size_t j = 0;
    Matrix<ElemType>* idx2cls= matrices[STRIDX2CLS];
    if (idx2cls== nullptr) return;

    if (m_idx2clsRead) return;

    // populate local CPU matrix
    m_id2classLocal->SwitchToMatrixType(MatrixType::DENSE);
    m_id2classLocal->Resize(nwords , 1, false);        

    //move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = m_id2classLocal->GetDeviceId();
    m_id2classLocal->TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
    for (size_t j = 0; j < nwords ; j++) 
    {
        int clsidx = idx4class[j]; 
        (*m_id2classLocal)(j,0) = clsidx; 
    }
    m_id2classLocal->TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    int oldDeviceId = idx2cls->GetDeviceId();
    // caution, SetValue changes idx2cls from GPU to CPU, may change this behavior later
    idx2cls->SetValue(*m_id2classLocal); 
    idx2cls->TransferFromDeviceToDevice(idx2cls->GetDeviceId(), oldDeviceId, true);
    
    m_idx2clsRead = true;
}

template<class ElemType>
void SequenceReader<ElemType>::GetClassInfo(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    size_t j = 0;
    Matrix<ElemType>* clsinfo = matrices[CLASSINFO];
    if (clsinfo == nullptr) return;

    if (m_clsinfoRead) return;

    // populate local CPU matrix
    m_classInfoLocal->SwitchToMatrixType(MatrixType::DENSE);
    m_classInfoLocal->Resize(2, class_size);        

    //move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = m_classInfoLocal->GetDeviceId();
    m_classInfoLocal->TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);

    int clsidx; 
    int prvcls = -1;
    for (size_t j = 0; j < nwords; j++) 
    {
        clsidx = idx4class[j]; 
        if (prvcls != clsidx)
        {
            if (prvcls >= 0)
                (*m_classInfoLocal)(1, prvcls) = j;
            prvcls = clsidx;
            (*m_classInfoLocal)(0, prvcls) = j;
        }
    }
    (*m_classInfoLocal)(1, prvcls) = nwords;

    m_classInfoLocal->TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    int oldDeviceId = clsinfo->GetDeviceId();
    // caution, SetValue changes m_classInfoLocal from GPU to CPU, may change this behavior later
    clsinfo->SetValue(*m_classInfoLocal); 
    clsinfo->TransferFromDeviceToDevice(clsinfo->GetDeviceId(), oldDeviceId, true);

    m_clsinfoRead = true;
}

template<class ElemType>
bool SequenceReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{

    // get out if they didn't call StartMinibatchLoop() first
    if (m_mbSize == 0)
        return false;

    // check to see if we have changed epochs, if so we are done with this one.
    if (m_sequence.size() > 0 && m_mbStartSample > m_sequence[m_sequence.size()-1])
        return false;

    bool moreData = EnsureDataAvailable(m_mbStartSample);
    if (moreData == false)
        return false; 

    // figure which sweep of the randomization we are on
    size_t epochSample = m_mbStartSample % m_epochSize; // where the minibatch starts in this epoch
    size_t epochEnd = m_epochSize;
    size_t recordStart = m_totalSamples?m_mbStartSample%m_totalSamples:m_mbStartSample;

    // actual size is the size of the next seqence
    size_t actualmbsize = 0;

    // figure out the size of the next sequence
    if (m_seqIndex > 0 && m_seqIndex < m_sequence.size() && m_sequence.size() > 1)
    {
        actualmbsize = m_sequence[m_seqIndex] - m_sequence[m_seqIndex-1];   
    }
    else
    {
        actualmbsize = m_sequence[0];
    }

    if (actualmbsize > m_mbSize){
        Error("specified minibatch size %d is smaller than the actual minibatch size %d. memory can crash!", m_mbSize, actualmbsize);
    }

    // hit the end of the dataset, 
    if (!moreData)
    {
        // make sure we take into account hitting the end of the dataset (not wrapping around)
        actualmbsize = min(m_totalSamples-recordStart,actualmbsize);
    }

    // now get the labels
    const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];

    if (labelInfo.type == labelCategory)
    {
        memset(m_labelsBuffer,0,sizeof(ElemType)*labelInfo.dim*actualmbsize);
        memset(m_labelsIdBuffer,0,sizeof(IDataReader<ElemType>::LabelIdType)*actualmbsize);
    }
    else if (labelInfo.type != labelNone)
    {
        memset(m_labelsBuffer,0,sizeof(ElemType)*1*actualmbsize);        
    }

    if (actualmbsize > 0)
    {

        memset(m_featuresBuffer, 0, sizeof(ElemType)*actualmbsize*labelInfo.dim);

        //loop through all the samples
        int j = 0;
        Matrix<ElemType>& features = *matrices[m_featuresName];
        if (matrices.find(m_featuresName) != matrices.end())
        {
            if(features.GetMatrixType() == MatrixType::DENSE) 
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
            size_t jRand = jSample;
         
            // vector of feature data goes into matrix column
            size_t idx = m_featureData[jRand];
            m_featuresBuffer[j*labelInfo.dim + idx] = (ElemType)1; 

            if (matrices.find(m_featuresName) != matrices.end())
                features.SetValue(idx, j, (ElemType)1); 
        }

        GetLabelOutput(matrices, m_mbStartSample, actualmbsize);
        GetInputToClass(matrices);
        GetClassInfo(matrices);

        // make sure that the sequence index matches our end index
        assert(m_sequence[m_seqIndex] == m_mbStartSample+actualmbsize);
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
    try{
        // get the features array
        if (matrices.find(m_featuresName) == matrices.end())
        {
            Matrix<ElemType>& nbs = *matrices[L"numberobs"];
            int curDevId = nbs.GetDeviceId();
            nbs.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
            nbs(0,0) = actualmbsize;
            nbs.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);
            for (size_t i = 0; i < actualmbsize; i++)
            {
                std::wstring ws = msra::strfun::wstrprintf (L"feature%d", i);
                Matrix<ElemType>& features = *matrices[ws];
                features.SetValue(labelInfo.dim, 1, &m_featuresBuffer[i*labelInfo.dim],matrixFlagNormal);
            }
        }
    }catch(...)
    {
        Error("features size might not be sufficiently large. The asked minibatch size is %s. check minibatchSize in the feature definition"  ,actualmbsize);
    }

    try
    {
        if (labelInfo.type == labelCategory)
        {
            if (matrices.find(m_labelsName[labelInfoOut]) == matrices.end())
            {
                for (size_t i = 0; i < actualmbsize; i++)
                {
                    std::wstring ws = msra::strfun::wstrprintf (L"label%d", i);
                    Matrix<ElemType>* labels = matrices[ws]; 
                    labels->SetValue(labelInfo.dim, 1, &m_labelsBuffer[i * labelInfo.dim],matrixFlagNormal);
                }
            }
        }
        else if (labelInfo.type != labelNone)
        {
            Matrix<ElemType>* labels = matrices[m_labelsName[labelInfoOut]];
            labels->SetValue(1, actualmbsize,m_labelsBuffer,matrixFlagNormal);
        }
    }catch(...)
    {
        Error("cannot find matrices for %s", m_labelsName[labelInfoOut]);
    }

    // we read some records, so process them
    return true;
}

template<class ElemType>
void SequenceReader<ElemType>::OrganizeClass()
{
    //allocate auxiliary class variables (for faster search when normalizing probability at output layer)
    int cl, i;
    for (i=0; i<class_size; i++) {
        class_cn.push_back(0); 
    }

    for (i=0; i<nwords; i++) {
        cl=idx4class[i];
        class_words[cl].push_back(i); 
        class_cn[cl]++;
    }

    for (i=0; i<class_size; i++) {
        if (class_cn[i] == 0) {
            Error ("ERROR: class is empty");
        }
    }
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType>
const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& SequenceReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetLabelMapping(sectionName);
    }
    const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];

    return labelInfo.mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType>
void SequenceReader<ElemType>::SetLabelMapping(const std::wstring& sectionName, const std::map<typename IDataReader<ElemType>::LabelIdType, typename LabelType>& labelMapping)
{
    if (m_cachingReader)
    {
        Error("Cannot set mapping table when the caching reader is being used");
    }
    LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];

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
template<class ElemType>
bool SequenceReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    if (m_cachingReader)
    {
        return m_cachingReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
    }
    Error("GetData not supported in SequenceReader");
    return false;
}

// instantiate all the combinations we expect to be used
template class SequenceReader<double>; 
template class SequenceReader<float>;

template<class ElemType>
void BatchSequenceReader<ElemType>::Init(const ConfigParameters& readerConfig)
{
    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    // NOTE: probably want to re-enable at some point

    // initialize the cache
    //InitCache(readerConfig);
    //m_readerConfig = readerConfig;

    //// if we have a cache, no need to parse the test files...
    //if (m_cachingReader)
    //    return;

    std::vector<std::wstring> features;
    std::vector<std::wstring> labels;
    GetFileConfigNames(readerConfig, features, labels);
    if (features.size() > 0)
    {
        m_featuresName = features[0];
    }

    if (labels.size() == 2)
    {
        for (int index = labelInfoMin; index < labelInfoMax; ++index)
        {
            m_labelsName[index] = labels[index];
        }
    }
    else
        Error("two label definitions (in and out) required for Sequence Reader");

    ConfigParameters featureConfig = readerConfig(m_featuresName,"");
    ConfigParameters labelConfig[2] = {readerConfig(m_labelsName[0],""),readerConfig(m_labelsName[1],"")};

    class_size = 0;
    m_featureDim = featureConfig("dim");
    for (int index = labelInfoMin; index < labelInfoMax; ++index)
    {
        m_labelInfo[index].idMax = 0; 
        m_labelInfo[index].beginSequence = labelConfig[index]("beginSequence", "");
        m_labelInfo[index].endSequence = labelConfig[index]("endSequence", "");

        // determine label type desired
        std::string labelType(labelConfig[index]("labelType","Category"));
        if (labelType == "Category")
        {
            m_labelInfo[index].type = labelCategory;
        }
        else if (labelType == "NextWord")
        {
            // in this case, it's all identical to the Input labels, except the data type
            m_labelInfo[index].type = labelNextWord;
            m_labelInfo[index].dim = m_labelInfo[labelInfoIn].dim;
        }
        else if (labelType == "None")
        {
            m_labelInfo[index].type = labelNone;
            m_labelInfo[index].dim = 0;   // override for no labels
        }
        
        // if we have labels, we need a label Mapping file, it will be a file with one label per line
        if (m_labelInfo[index].type != labelNone)
        {
            std::wstring wClassFile = readerConfig("wordclass", "");
            nwords = labelConfig[index]("labelDim");
            if (wClassFile != L""){
                ReadClassInfo(wClassFile  , false);
            }

            std::vector<string> arrayLabels;
            std::wstring labelPath = labelConfig[index]("labelMappingFile");
            if (fexists(labelPath))
            {
                LoadLabelFile(labelPath, arrayLabels);
                for (int i=0; i < arrayLabels.size(); ++i)
                {
                    LabelType label = arrayLabels[i];
                    m_labelInfo[index].mapIdToLabel[i] = label;
                    m_labelInfo[index].mapLabelToId[label] = i;
                }
                m_labelInfo[index].idMax = (LabelIdType)arrayLabels.size();
                m_labelInfo[index].mapName = labelPath;
            }
            else
            {
                if (wClassFile != L""){
                    ReadClassInfo(wClassFile  , false);
                    int iMax = -1, i; 
                    for (auto ptr = word4idx.begin(); ptr != word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first; 
                        i = ptr->second; 
                        iMax = max(i, iMax);
                        m_labelInfo[index].mapIdToLabel[i] = label;
                        m_labelInfo[index].mapLabelToId[label] = i;
                    }
                    m_labelInfo[index].idMax = (LabelIdType)(iMax+1);

                    OrganizeClass();

                }
                m_labelInfo[index].mapName = labelPath;

                m_labelInfo[index].fileToWrite = labelPath;
            }
        }

        m_labelInfo[index].dim = labelConfig[index]("labelDim");

        // update dimension if the file says it's bigger
        if (m_labelInfo[index].dim < m_labelInfo[index].idMax)
        {
            m_labelInfo[index].dim = m_labelInfo[index].idMax;
        }
    }

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = m_seqIndex = 0;
    m_endReached = false;
    m_readNextSampleLine = 0;
    m_readNextSample = 0;
    m_traceLevel = readerConfig("traceLevel","0");
    m_parser.SetTraceLevel(m_traceLevel);

    if (readerConfig.Exists("randomize"))
    {
        string randomizeString = readerConfig("randomize");
        if (randomizeString == "None")
        {
            ;
        }
        else if (randomizeString == "Auto")
        {
            ;
        }
        else
        {
            ;//readerConfig("randomize");
        }
    }
    else
    {
        ; //randomizeAuto;
    }

    // The input data is a combination of the label Data and extra feature dims together
//    m_featureCount = m_featureDim + m_labelInfo[labelInfoIn].dim;
    m_featureCount = 1; 

    std::wstring m_file = readerConfig("file");
    if (m_traceLevel > 0)
        fprintf(stderr, "reading sequence file %ws\n", m_file.c_str());

    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseInit(m_file.c_str(), m_featureDim, labelIn.dim, labelOut.dim, labelIn.beginSequence, labelIn.endSequence, labelOut.beginSequence, labelOut.endSequence);

    mBlgSize = readerConfig("nbruttsineachrecurrentiter", "1");
}

template<class ElemType>
void BatchSequenceReader<ElemType>::Reset()
{
    mProcessed.clear();
    mToProcess.clear();
    mLastProcssedSentenceId = 0;
    mPosInSentence = 0;
    mLastPosInSentence = 0;
    mNumRead = 0;

    if (m_labelTemp.size() > 0)
        m_labelTemp.clear();
    if (m_featureTemp.size() > 0)
        m_featureTemp.clear();
    m_parser.mSentenceIndex2SentenceInfo.clear();
}

template<class ElemType>
void BatchSequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory();    // free the memory used by the SequenceReader
    }

    // if we are reading from the cache, do so now and return
    if (m_cachingReader)
    {
        m_cachingReader->StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
        return;
    } 

    if (m_featuresBuffer==NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];
        m_featuresBuffer = new ElemType[mbSize*labelInfo.dim];
        memset(m_featuresBuffer,0,sizeof(ElemType)*mbSize*labelInfo.dim);
    }

    if (m_labelsBuffer==NULL)
    {
        const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];
        if (labelInfo.type == labelCategory)
        {
            m_labelsBuffer = new ElemType[labelInfo.dim*mbSize];
            memset(m_labelsBuffer,0,sizeof(ElemType)*labelInfo.dim*mbSize);
            m_labelsIdBuffer = new IDataReader<ElemType>::LabelIdType[mbSize];
            memset(m_labelsIdBuffer,0,sizeof(IDataReader<ElemType>::LabelIdType)*mbSize);
        }
        else if (labelInfo.type != labelNone)
        {
            m_labelsBuffer = new ElemType[mbSize];
            memset(m_labelsBuffer,0,sizeof(ElemType)*mbSize);
            m_labelsIdBuffer = NULL;
        }
    }      

    m_featuresBufferRow = new size_t[mbSize];
    m_featuresBufferRowIdx = new size_t[mbSize];

    m_labelsIdBufferRow = new size_t[2*mbSize];
    m_labelsBlock2Id = new size_t[2*mbSize];
    m_labelsBlock2UniqId = new size_t[2*mbSize];

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
    size_t epochSize = m_epochSize == requestDataSize?1000:m_epochSize;
    m_epoch = epoch;
    m_mbStartSample = epoch*m_epochSize;

    // allocate room for the data
    m_featureData.reserve(m_featureCount*epochSize);
    if (m_labelInfo[labelInfoOut].type == labelCategory)
        m_labelIdData.reserve(epochSize); 
    else if (m_labelInfo[labelInfoOut].type != labelNone)
        m_labelData.reserve(epochSize);
    m_sequence.reserve(m_seqIndex); // clear out the sequence array
    /// this is too complicated for LM 
    // SetupEpoch(); 
    /// use the LMSetupEpoch() instead
    LMSetupEpoch();

    m_clsinfoRead = false; 
    m_idx2clsRead = false; 

    const LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    const LabelInfo& labelOut = m_labelInfo[labelInfoOut];
    m_parser.ParseReset(); 

    Reset();
}

template<class ElemType>
size_t BatchSequenceReader<ElemType>::FindNextSentences(size_t numRead)
{  
    size_t sln = 0;

    if (numRead == 0) return 0;

    if (mProcessed.size() == 0)
    {
        mProcessed.resize(numRead, false);
    }

    if (mToProcess.size() > 0)
    {
        bool allDone = false; 
        for (int s = 0; s < mToProcess.size(); s++)
        {
            int mp = mToProcess[s];
            if (mProcessed[mp])
            {
                mLastProcssedSentenceId = mp;
                mLastPosInSentence = 0;
                allDone = true;
                break;
            }
        }
        if (allDone)
        {
            mToProcess.clear();
        }
    }

    if (mToProcess.size() > 0)
    {
        sln = m_parser.mSentenceIndex2SentenceInfo[mToProcess[0]].sLen;
        return sln;
    }

    for (int seq = mLastProcssedSentenceId ; seq < numRead; seq++)
    {
        if (mProcessed[seq]) continue;
        
        if (sln == 0)
        {
            sln = m_parser.mSentenceIndex2SentenceInfo[seq].sLen;
        }
        if (sln == m_parser.mSentenceIndex2SentenceInfo[seq].sLen &&
            mProcessed[seq] == false && mToProcess.size() < mBlgSize)
            mToProcess.push_back(seq);

        if (mToProcess.size() == mBlgSize) break;
    }

    return sln;
}

template<class ElemType>
bool BatchSequenceReader<ElemType>::EnsureDataAvailable(size_t mbStartSample)
{
    bool bDataIsThere = true; 

    m_featureData.clear();
    m_labelIdData.clear();

    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];

    bool nextWord = false;
    if (m_labelInfo[labelInfoOut].type == labelNextWord)
    {
        nextWord = true;
    }
    LabelInfo& labelInfo = m_labelInfo[nextWord?labelInfoIn:labelInfoOut];

    // see how many we already read
    int sequencesRead = 0;
    std::vector<SequencePosition> seqPos;
    
    size_t sLn = FindNextSentences(mNumRead);
    if (sLn == 0)
    {
        Reset();

        mNumRead = m_parser.Parse(CACHE_BLOG_SIZE, &m_labelTemp, &m_featureTemp, &seqPos);
        if (mNumRead == 0) return false;

        std::random_shuffle(m_parser.mSentenceIndex2SentenceInfo.begin(), m_parser.mSentenceIndex2SentenceInfo.end());

        m_readNextSampleLine += mNumRead;
        sLn = FindNextSentences(mNumRead);
    }

    /// add one minibatch 
    int i = mLastPosInSentence; 
    int j = 0;
    // exclude the last token since it is the last label to be predicted
    for (i = mLastPosInSentence; j < m_mbSize &&  i < sLn-1; i++ , j++)
    {
        for (int k = 0; k < mToProcess.size(); k++)
        {
            size_t seq = mToProcess[k];
            size_t label = m_parser.mSentenceIndex2SentenceInfo[seq].sBegin + i;

            // labelIn should be a category label 
            LabelType labelValue = m_labelTemp[label++];

            // to-do, should ignore <s>, check the sentence ending is </s> 
            // need to remove <s> from the training set
            // allocate and initialize the next chunck of featureData
            if (labelIn.type == labelCategory)
            {
                LabelIdType index = GetIdFromLabel(labelValue, labelIn);

                // use the found value, and set the appropriate location to a 1.0
                assert(labelIn.dim > index); // if this goes off labelOut dimension is too small
                m_featureData.push_back(index);
            }
            else
            {
                Error("Input label expected to be a category label");
            }

            // now get the output label
            if (m_labelInfo[labelInfoOut].type == labelCategory)
            {
                labelValue = m_labelTemp[label++];
            }
            else if (nextWord)
            {
                // this is the next word (label was incremented above)
                labelValue = m_labelTemp[label];
                if (!_strcmpi(labelValue.c_str(), m_labelInfo[labelInfoIn].endSequence.c_str()))
                {
                    labelValue = labelInfo.endSequence;
                }
            }
            else
            {
                Error("Invalid output label type, expected Category, or Next Word");
            }

            // get the ID from the label
            LabelIdType id = GetIdFromLabel(labelValue, labelInfo);
            m_labelIdData.push_back(id);

            m_totalSamples ++;
        }
    }

    mLastPosInSentence = i;

    return bDataIsThere;
}

template<class ElemType>
size_t BatchSequenceReader<ElemType>::NumberSlicesInEachRecurrentIter()
{
    size_t sz = mToProcess.size();
    return sz; 
}

template<class ElemType>
void BatchSequenceReader<ElemType>::SetNbrSlicesEachRecurrentIter(const size_t mz)
{
    mBlgSize = mz;
}

template<class ElemType>
bool BatchSequenceReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{

    // get out if they didn't call StartMinibatchLoop() first
    if (m_mbSize == 0)
        return false;

    bool moreData = EnsureDataAvailable(m_mbStartSample);
    if (moreData == false)
        return false; 

    // actual size is the size of the next seqence
    size_t actualmbsize = 0;

    // figure out the size of the next sequence
    actualmbsize = m_labelIdData.size() ; 
    if (actualmbsize > m_mbSize * mToProcess.size()){
        Error("specified minibatch size %d is smaller than the actual minibatch size %d. memory can crash!", m_mbSize, actualmbsize);
    }

    // now get the labels
    const LabelInfo& labelInfo = m_labelInfo[( m_labelInfo[labelInfoOut].type == labelNextWord)?labelInfoIn:labelInfoOut];

    if (actualmbsize > 0)
    {

        //loop through all the samples
        Matrix<ElemType>& features = *matrices[m_featuresName];
        if (matrices.find(m_featuresName) != matrices.end())
        {
            if(features.GetMatrixType() == MatrixType::DENSE) 
            {
                features.Resize(labelInfo.dim, actualmbsize, false);
                features.SetValue(0);
            }
            else 
            {
                features.Resize(labelInfo.dim, actualmbsize, actualmbsize);
                features.Reset();
            }
        }
      
        // copy m_featureData to matrix, for CPU
        if(features.GetCurrentMatrixLocation() == CPU )
        {
            for (size_t j = 0; j < actualmbsize; ++j)
            {
                // vector of feature data goes into matrix column
                size_t idx = m_featureData[j];

                if (matrices.find(m_featuresName) != matrices.end())
                    features.SetValue(idx, j, (ElemType)1); 
            }
        }
        else // for GPU
        {
            if (matrices.find(m_featuresName) != matrices.end())
            {
                m_indexer.clear();
                size_t size = m_featureData.size();

                for(int i = 0; i < size; i++) 
                {
                    m_featuresBufferRow[i] = (size_t)m_featureData[i];                    
                    if(m_indexer.find(m_featuresBufferRow[i]) == m_indexer.end()) 
                    {
                        m_indexer[m_featuresBufferRow[i]] = m_indexer.size();
                    }
                    m_featuresBufferRowIdx[i] = m_indexer[m_featuresBufferRow[i]];
                }               
                features.SetMatrixFromCSCFormat(m_featuresBufferRow, m_featuresBufferRowIdx, size, m_indexer.size());
            }
        }
                
        // TODO: move these two methods to startMiniBatchLoop()
        GetInputToClass(matrices);
        GetClassInfo(matrices);
        GetLabelOutput(matrices, 0, actualmbsize);

        // go to the next sequence
        m_seqIndex++;
    } 
    else
        return false; 

    // now transfer to the GPU as needed
    try{
        // get the features array
        if (matrices.find(m_featuresName) == matrices.end())
        {
            Matrix<ElemType>& nbs = *matrices[L"numberobs"];
            int curDevId = nbs.GetDeviceId();
            nbs.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
            nbs(0,0) = actualmbsize;
            nbs.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);
            for (size_t i = 0; i < actualmbsize; i++)
            {
                std::wstring ws = msra::strfun::wstrprintf (L"feature%d", i);
                Matrix<ElemType>& features = *matrices[ws];
                features.SetValue(labelInfo.dim, 1, &m_featuresBuffer[i*labelInfo.dim],matrixFlagNormal);
            }
        }
    }catch(...)
    {
        Error("features size might not be sufficiently large. The asked minibatch size is %s. check minibatchSize in the feature definition"  ,actualmbsize);
    }

    // we read some records, so process them
    return true;
}

template<class ElemType>
void BatchSequenceReader<ElemType>::SetSentenceEnd(int wrd, int pos, int actualMbSize)
{
    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelIdType index = GetIdFromLabel(labelIn.endSequence.c_str(), labelIn);

    if (pos == actualMbSize - 1) 
    {
        if (wrd == index)
            mSentenceEnd = true;
        else
            mSentenceEnd = false; 
    }
}

template<class ElemType>
void BatchSequenceReader<ElemType>::SetSentenceBegin(int wrd, int pos, int actualMbSize)
{
    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelIdType index = GetIdFromLabel(labelIn.beginSequence.c_str(), labelIn);

    if (pos == 0) 
    {
        if (wrd == index)
            mSentenceBegin = true;
        else
            mSentenceBegin = false; 
    }
}

template<class ElemType>
void BatchSequenceReader<ElemType>::SetSentenceEndInBatch(vector<size_t> &sentenceEnd)
{
    sentenceEnd.resize(mToProcess.size());
    if (mSentenceBegin)
    {
        sentenceEnd.assign(mToProcess.size(), 0);
    }
    else
    {
        sentenceEnd.assign(mToProcess.size(), m_mbSize+2);
    }
}

template<class ElemType>
bool BatchSequenceReader<ElemType>::DataEnd(EndDataType endDataType)
{
    bool ret = false;
    switch (endDataType)
    {
    case endDataNull:
        assert(false);
        break;
    case endDataEpoch:
    case endDataSet:
        ret = !EnsureDataAvailable(m_mbStartSample);
        break;
    case endDataSentence:  // for fast reader each minibatch is considered a "sentence", so always true
        if (mSentenceEnd)
        {
            for (auto ptr = mToProcess.begin(); ptr != mToProcess.end(); ptr++)
                mProcessed[*ptr] = true;
        }
        ret = mSentenceEnd;
        break;
    }
    return ret;

}

template<class ElemType>
void BatchSequenceReader<ElemType>::GetLabelOutput(std::map<std::wstring, 
    Matrix<ElemType>*>& matrices, 
    size_t m_mbStartSample, size_t actualmbsize)
{
    size_t j = 0;
    Matrix<ElemType>* labels = matrices[m_labelsName[labelInfoOut]]; 
    if (labels == nullptr) return;
    
    if(labels->GetMatrixType() == MatrixType::DENSE) 
    {
        labels->Resize(nwords + class_size, actualmbsize, false);
        labels->SetValue(0);
    }
    else 
    {
        labels->Resize(nwords + class_size, actualmbsize, 2*actualmbsize);
        labels->Reset();
    }

    if(labels->GetCurrentMatrixLocation() == CPU) {
        for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
        {
            // pick the right sample with randomization if desired
            size_t jRand = jSample;

            int    wrd = m_labelIdData[jRand];
            int    clsidx = idx4class[wrd]; 

            labels->SetValue(wrd, j, 1); 

            SetSentenceEnd(wrd, j, actualmbsize);
            SetSentenceBegin(wrd, j, actualmbsize);

            if (class_size > 0)
                labels->SetValue(nwords + clsidx, j, 1); 
        }
    }
    else // GPU
    {
        m_indexer.clear();
        Matrix<ElemType>* id2cls = matrices[STRIDX2CLS];
        int p = 0;
        int b = 0; 
        int nz = 0;
        
        for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
        {
            // pick the right sample with randomization if desired
            size_t jRand = jSample;         
            int    wrd = m_labelIdData[jRand];
            int    clsidx = idx4class[wrd];         
            SetSentenceEnd(wrd, j, actualmbsize);
            SetSentenceBegin(wrd, j, actualmbsize);

            int start[2];
            int end[2];
            int target[2];
            int blockId[2];

            start[0] = (*m_classInfoLocal)(0, clsidx);
            end[0] = (*m_classInfoLocal)(1, clsidx);
            target[0] = wrd;
            blockId[0] = clsidx;
            start[1] = nwords;
            end[1] = nwords + (*m_classInfoLocal).GetNumCols(); 
            target[1] = nwords + clsidx;
            blockId[1] = -1;

            for(int i = 0; i < 2; i++) 
            {
                m_labelsIdBufferRow[p] = target[i];
                int len = end[i] - start[i];
                
                if(m_indexer.find(blockId[i]) == m_indexer.end()) 
                {
                    m_indexer[blockId[i]] = b;                    
                    b += len;
                }   
                m_labelsBlock2Id[p] = nz;
                m_labelsBlock2UniqId[p] = m_indexer[blockId[i]];
                nz += len;
                p++;
            }
        }
        Matrix<ElemType>* clsinfo = matrices[CLASSINFO];
        
        labels->SetMatrixFromLabelAndClass(m_labelsIdBufferRow, m_labelsBlock2Id, m_labelsBlock2UniqId, 2*actualmbsize, nz, b);
    }
}

template class BatchSequenceReader<double>; 
template class BatchSequenceReader<float>;

}}}