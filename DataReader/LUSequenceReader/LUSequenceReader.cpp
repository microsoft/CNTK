//
// <copyright file="LUSequenceReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// LUSequenceReader.cpp : Defines the exported functions for the DLL application.
//


#include "stdafx.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "LUSequenceReader.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#include <fstream>
#include <random>       // std::default_random_engine
#include "fileutil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template<class ElemType>
bool LUSequenceReader<ElemType>::ReadRecord(size_t /*readSample*/)
{
    return false; // not used
}

// RecordsToRead - Determine number of records to read to populate record buffers
// mbStartSample - the starting sample from which to read
// tail - we are checking for possible remainer records to read (default false)
// returns - true if we have more to read, false if we hit the end of the dataset
template<class ElemType>
size_t LUSequenceReader<ElemType>::RecordsToRead(size_t mbStartSample, bool tail)
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
/* return value used to be unsigned */
typename IDataReader<ElemType>::LabelIdType LUSequenceReader<ElemType>::GetIdFromLabel(const std::string& labelValue, LabelInfo& labelInfo)
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

// GetIdFromLabel - get an Id from a Label
// mbStartSample - the starting sample we are ensureing are good
// endOfDataCheck - check if we are at the end of the dataset (no wraparound)
// returns - true if we have more to read, false if we hit the end of the dataset
template<class ElemType>
bool LUSequenceReader<ElemType>::GetIdFromLabel(const vector<string>& labelValue, LabelInfo& labelInfo, vector<LabelIdType>& val)
{
    val.clear();

    for (size_t i = 0; i < labelValue.size(); i++)
    {
        auto found = labelInfo.mapLabelToId.find(labelValue[i]);

        // not yet found, add to the map
        if (found != labelInfo.mapLabelToId.end())
        {
            val.push_back(found->second);
        }
        else
            RuntimeError("LUSequenceReader::GetIdFromLabel: cannot find value %s to map to id. Check input and output mapping file. Check if all input/output symbols are defined in the input/output mapping/list files.", labelValue[i].c_str());
    }
    return true;
}

template<class ElemType>
/*IDataReader<ElemType>::LabelIdType*/ bool LUSequenceReader<ElemType>::CheckIdFromLabel(const std::string& labelValue, const LabelInfo& labelInfo, unsigned & labelId)
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

// UpdateDataVariables - Update variables that depend on the dataset being completely read
template<class ElemType>
void LUSequenceReader<ElemType>::UpdateDataVariables()
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
void LUSequenceReader<ElemType>::WriteLabelFile()
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
void LUSequenceReader<ElemType>::LoadLabelFile(const std::wstring &filePath, std::vector<LabelType>& retLabels)
{
    File file(filePath, fileOptionsRead);

    // initialize with file name
    std::string path = msra::strfun::utf8(filePath);
    auto location = path.find_last_of("/\\");
    if (location != npos)
        path = path.substr(location+1);
    
    // read the entire file into a string
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
void LUSequenceReader<ElemType>::Destroy()
{
    delete this;
}

// Init - Reader Initialize for multiple data sets
// config - [in] configuration parameters for the datareader
// Sample format below:
//# Parameter values for the reader
//reader=[
//  # reader to use
//  readerType=LUSequenceReader
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
void LUSequenceReader<ElemType>::Init(const ConfigParameters& readerConfig)
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
        RuntimeError("two label definitions (in and out) required for Sequence Reader");

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
            std::wstring wClassFile = labelConfig[index]("token", "");
            if (wClassFile != L""){
                ReadLabelInfo(wClassFile  , m_labelInfo[index].word4idx, m_labelInfo[index].idx4word);
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
                    int iMax = -1, i; 
                    for (auto ptr = m_labelInfo[index].word4idx.begin(); ptr != m_labelInfo[index].word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first; 
                        i = ptr->second; 
                        iMax = max(i, iMax);
                        m_labelInfo[index].mapIdToLabel[i] = label;
                        m_labelInfo[index].mapLabelToId[label] = i;
                    }
                    m_labelInfo[index].idMax = (LabelIdType)(iMax+1);

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

    mRandomize = false; 
    if (readerConfig.Exists("randomize"))
    {
        string randomizeString = readerConfig("randomize");
        if (randomizeString == "None")
        {
            mRandomize = false;
        }
        else if (randomizeString == "Auto")
        {
            mRandomize = true;
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

    m_seed = 0;

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
void LUSequenceReader<ElemType>::ReadWord(char *word, FILE *fin)
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
void LUSequenceReader<ElemType>::ChangeMaping(const map<string, string>& maplist, 
                                              const string & unkstr,
                                              map<string, int> & word4idx)
{
    auto punk = word4idx.find(unkstr);
    for(auto ptr = word4idx.begin(); ptr != word4idx.end(); ptr++)
    {
        string wrd = ptr->first;
        int idx = -1; 
        if (maplist.find(wrd) != maplist.end())
        {
            string mpp = maplist.find(wrd)->second; 
            idx = word4idx[mpp];
        }
        else
        {
            if (punk == word4idx.end())
            {
                RuntimeError("check unk list is missing ");
            }
            idx = punk->second;
        }

        word4idx[wrd] = idx;
    }
}

template<class ElemType>
void LUSequenceReader<ElemType>::ReadLabelInfo(const wstring & vocfile, 
                                                map<string, int> & word4idx,
                                                map<int, string>& idx4word)
{
    char strFileName[MAX_STRING];
    char stmp[MAX_STRING];
    string strtmp; 
    size_t sz;
    int b;
    class_size  = 0;

    wcstombs_s(&sz, strFileName, 2048, vocfile.c_str(), vocfile.length());

    FILE * vin;
    vin = fopen(strFileName, "rt") ;

    if (vin == nullptr)
    {
        RuntimeError("cannot open word class file");
    }
    b = 0;
    while (!feof(vin)){
        fscanf_s(vin, "%s\n", stmp, _countof(stmp));
        word4idx[stmp] = b;
        idx4word[b++]= stmp;
    }
    fclose(vin);

}

// InitCache - Initialize the caching reader if cache files exist, otherwise the writer
// readerConfig - reader configuration
template<class ElemType>
void LUSequenceReader<ElemType>::InitCache(const ConfigParameters& readerConfig)
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
LUSequenceReader<ElemType>::~LUSequenceReader()
{
    ReleaseMemory();
    delete m_cachingReader;
    delete m_cachingWriter;
}

// ReleaseMemory - release the memory footprint of LUSequenceReader
// used when the caching reader is taking over
template<class ElemType>
void LUSequenceReader<ElemType>::ReleaseMemory()
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
    m_featureWordContext.clear();
    m_labelIdData.clear();
    m_labelData.clear();
    m_sequence.clear();
}

template<class ElemType>
void LUSequenceReader<ElemType>::LMSetupEpoch()
{
    m_readNextSampleLine = m_readNextSample = m_epochStartSample = m_mbStartSample = m_seqIndex = 0;
}

// utility function to round an integer up to a multiple of size
inline size_t RoundUp(size_t value, size_t size) 
{
    return ((value + size -1)/size)*size;
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of Samples, etc.)
//     NOTE: for sequence data, this will be the MAX size of a sequence, as every sequence could be a different length
// epoch - [in] epoch number for this loop, if > 0 the requestedEpochSamples must be specified (unless epoch zero was completed this run)
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType>
void LUSequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory();    // free the memory used by the LUSequenceReader
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
            m_labelsIdBuffer = new LabelIdType[mbSize];
            memset(m_labelsIdBuffer,0,sizeof(LabelIdType)*mbSize);
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

    m_parser.ParseReset(); 
}


template<class ElemType>
bool LUSequenceReader<ElemType>::SentenceEnd()
{
    // this is after getMinibatch size, which has increased m_seqIndex by 1
    // so the real index is m_seqIndex - 1; 
    int seqIndex = (int)m_seqIndex - 1; 

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
            RuntimeError("cannot find sentence begining label");

        if (m_labelIdData[jEnd] == index )
            return true;
        else 
            return false;
    }
    return false; 
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType>
const std::map<typename IDataReader<ElemType>::LabelIdType, typename IDataReader<ElemType>::LabelType>& LUSequenceReader<ElemType>::GetLabelMapping(const std::wstring& sectionName)
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
void LUSequenceReader<ElemType>::SetLabelMapping(const std::wstring& /*sectionName*/, const std::map<typename IDataReader<ElemType>::LabelIdType, LabelType>& labelMapping)
{
    if (m_cachingReader)
    {
        RuntimeError("Cannot set mapping table when the caching reader is being used");
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
bool LUSequenceReader<ElemType>::GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart)
{
    if (!m_cachingReader)
        RuntimeError("GetData not supported in LUSequenceReader");
    return m_cachingReader->GetData(sectionName, numRecords, data, dataBufferSize, recordStart);
}

// instantiate all the combinations we expect to be used
template class LUSequenceReader<double>; 
template class LUSequenceReader<float>;

template<class ElemType>
void BatchLUSequenceReader<ElemType>::LoadWordMapping(const ConfigParameters& readerConfig)
{
    mWordMappingFn = readerConfig("wordmap", "");
    string si, so;
    if (mWordMappingFn != "")
    {
        ifstream fp;
        fp.open(mWordMappingFn.c_str());
        while (!fp.eof())
        {
            fp >> si >> so;
            mWordMapping[si] = so;
        }
        fp.close();
    }
    mUnkStr = readerConfig("unk", "<unk>");
}

template<class ElemType>
void BatchLUSequenceReader<ElemType>::Init(const ConfigParameters& readerConfig)
{
    // See if the user wants caching
    m_cachingReader = NULL;
    m_cachingWriter = NULL;

    LoadWordMapping(readerConfig);

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
        RuntimeError("two label definitions (in and out) required for Sequence Reader");

    ConfigParameters featureConfig = readerConfig(m_featuresName,"");
    ConfigParameters labelConfig[2] = {readerConfig(m_labelsName[0],""),readerConfig(m_labelsName[1],"")};

    class_size = 0;
    m_featureDim = featureConfig("dim");
    for (int index = labelInfoMin; index < labelInfoMax; ++index)
    {
        m_labelInfo[index].idMax = 0; 
        m_labelInfo[index].beginSequence = labelConfig[index]("beginSequence", "");
        m_labelInfo[index].endSequence = labelConfig[index]("endSequence", "");
        m_labelInfo[index].busewordmap = labelConfig[index]("usewordmap", "false");

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
            std::wstring wClassFile = labelConfig[index]("token", "");
            if (wClassFile != L""){
                ReadLabelInfo(wClassFile  , m_labelInfo[index].word4idx, m_labelInfo[index].idx4word);
            }
            if (m_labelInfo[index].busewordmap)
                ChangeMaping(mWordMapping, mUnkStr, m_labelInfo[index].word4idx);


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
                    int iMax = -1, i; 
                    for (auto ptr = m_labelInfo[index].word4idx.begin(); ptr != m_labelInfo[index].word4idx.end(); ptr++)
                    {
                        LabelType label = ptr->first; 
                        i = ptr->second; 
                        iMax = max(i, iMax);
                        m_labelInfo[index].mapIdToLabel[i] = label;
                        m_labelInfo[index].mapLabelToId[label] = i;
                    }
                    m_labelInfo[index].idMax = (LabelIdType)(iMax+1);

                }
                m_labelInfo[index].mapName = labelPath;

                m_labelInfo[index].fileToWrite = labelPath;
            }
        }

        m_labelInfo[index].dim = m_labelInfo[index].idMax;
    }

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = m_seqIndex = 0;
    m_endReached = false;
    m_readNextSampleLine = 0;
    m_readNextSample = 0;
    m_traceLevel = readerConfig("traceLevel","0");
    m_parser.SetTraceLevel(m_traceLevel);
    ConfigArray wContext = readerConfig("wordContext", "0");
    intargvector wordContext = wContext;
    m_wordContext = wordContext;
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
void BatchLUSequenceReader<ElemType>::Reset()
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
void BatchLUSequenceReader<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    // if we aren't currently caching, see if we can use a cache
    if (!m_cachingReader && !m_cachingWriter)
    {
        InitCache(m_readerConfig);
        if (m_cachingReader)
            ReleaseMemory();    // free the memory used by the LUSequenceReader
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
            m_labelsIdBuffer = new LabelIdType[mbSize];
            memset(m_labelsIdBuffer,0,sizeof(LabelIdType)*mbSize);
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

    m_parser.ParseReset(); 

    Reset();
}

template<class ElemType>
size_t BatchLUSequenceReader<ElemType>::FindNextSentences(size_t numRead)
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
            size_t mp = mToProcess[s];
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

    for (size_t seq = mLastProcssedSentenceId ; seq < numRead; seq++)
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
bool BatchLUSequenceReader<ElemType>::EnsureDataAvailable(size_t /*mbStartSample*/)
{
    bool bDataIsThere = true; 

    m_featureData.clear();
    m_labelIdData.clear();
    m_featureWordContext.clear();

    // now get the labels
    LabelInfo& featIn = m_labelInfo[labelInfoIn];
    LabelInfo& labelIn = m_labelInfo[labelInfoOut];

    // see how many we already read
    std::vector<SequencePosition> seqPos;

    {
        size_t sLn = FindNextSentences(mNumRead);
        if (sLn == 0)
        {
            Reset();

            mNumRead = m_parser.Parse(CACHE_BLOG_SIZE, &m_labelTemp, &m_featureTemp, &seqPos);
            if (mNumRead == 0) return false;

            if (mRandomize)
            {
                unsigned seed = m_seed;
                std::shuffle(m_parser.mSentenceIndex2SentenceInfo.begin(), m_parser.mSentenceIndex2SentenceInfo.end(), std::default_random_engine(seed));
                m_seed++;
            }

            m_readNextSampleLine += mNumRead;
            sLn = FindNextSentences(mNumRead);
        }

        /// add one minibatch 
        size_t i = mLastPosInSentence; 
        size_t j = 0;

        for (i = mLastPosInSentence; j < m_mbSize &&  i < sLn; i++ , j++)
        {
            for (int k = 0; k < mToProcess.size(); k++)
            {
                size_t seq = mToProcess[k];
                size_t label = m_parser.mSentenceIndex2SentenceInfo[seq].sBegin + i;
                std::vector<std::vector<LabelIdType>> tmpCxt;

                for (int i_cxt = 0; i_cxt < m_wordContext.size() ; i_cxt++)
                {
                    // to-do, should ignore <s>, check the sentence ending is </s> 
                    // need to remove <s> from the training set
                    // allocate and initialize the next chunck of featureData
                    if (featIn.type == labelCategory)
                    {
                        vector<LabelIdType> index ;
                        int ilabel = (int) (label + m_wordContext[i_cxt]);
                        if (ilabel < m_parser.mSentenceIndex2SentenceInfo[seq].sBegin )
                        {
                            GetIdFromLabel(m_featureTemp[m_parser.mSentenceIndex2SentenceInfo[seq].sBegin], featIn, index);
                        }
                        else if (ilabel >= m_parser.mSentenceIndex2SentenceInfo[seq].sEnd)
                        {
                            GetIdFromLabel(m_featureTemp[m_parser.mSentenceIndex2SentenceInfo[seq].sEnd-1], featIn, index);
                        }
                        else
                        {
                            GetIdFromLabel(m_featureTemp[ilabel], featIn, index);
                        }
                        if (i_cxt == 0)
                        {
                            m_featureData.push_back(index);
                        }
                        tmpCxt.push_back(index);
                    }
                    else
                    {
                        RuntimeError("Input label expected to be a category label");
                    }
                }

                m_featureWordContext.push_back(tmpCxt);

                // now get the output label
                LabelIdType id = GetIdFromLabel(m_labelTemp[label], labelIn);
                m_labelIdData.push_back(id);


                m_totalSamples ++;
                label++;
            }
        }

        mLastPosInSentence = i;
    }

    return bDataIsThere;
}

template<class ElemType>
size_t BatchLUSequenceReader<ElemType>::NumberSlicesInEachRecurrentIter()
{
    size_t sz = mToProcess.size();
    return sz; 
}

template<class ElemType>
void BatchLUSequenceReader<ElemType>::SetNbrSlicesEachRecurrentIter(const size_t mz)
{
    mBlgSize = mz;
}

template<class ElemType>
bool BatchLUSequenceReader<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
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
        RuntimeError("specified minibatch size %d is smaller than the actual minibatch size %d. memory can crash!", m_mbSize, actualmbsize);
    }

    // now get the labels
    const LabelInfo& featInfo = m_labelInfo[labelInfoIn];

    if (actualmbsize > 0)
    {

        //loop through all the samples
        Matrix<ElemType>& features = *matrices[m_featuresName];
        if (matrices.find(m_featuresName) != matrices.end())
        {
//            assert(features.GetMatrixType() == MatrixType::SPARSE);
            //features.Resize(featInfo.dim, actualmbsize, true);
            features.Resize(featInfo.dim * m_wordContext.size(), actualmbsize, true);
            features.SetValue(0);
        }

        for (size_t j = 0; j < actualmbsize; ++j)
        {
            size_t wrd = 0;
            // vector of feature data goes into matrix column
            for (size_t jj = 0; jj < m_featureWordContext[j].size(); jj++)
            {
                /// this support context dependent inputs since words or vector of words are placed
                /// in different slots
                for (size_t ii = 0; ii < m_featureWordContext[j][jj].size(); ii++)
                {
                    /// this can support bag of words, since words are placed in the same slot
                    size_t idx = m_featureWordContext[j][jj][ii];

                    if (matrices.find(m_featuresName) != matrices.end())
                    {
                        assert(idx < featInfo.dim);
                        features.SetValue(idx + jj * featInfo.dim, j, (ElemType)1); 
                    }
                }
            }

            assert(m_featureData[j].size() == 1);
            wrd = m_featureData[j][0];
            SetSentenceEnd(wrd, j, actualmbsize);
            SetSentenceBegin(wrd, j, actualmbsize);
        }

        GetLabelOutput(matrices, m_mbStartSample, actualmbsize);

        // go to the next sequence
        m_seqIndex++;
    } 
    else
        return false; 

    // we read some records, so process them
    return true;
}

template<class ElemType>
void BatchLUSequenceReader<ElemType>::GetLabelOutput(std::map<std::wstring, 
    Matrix<ElemType>*>& matrices, 
    size_t /*m_mbStartSample*/, size_t actualmbsize)
{
    const LabelInfo& labelInfo = m_labelInfo[labelInfoOut];
    Matrix<ElemType>* labels = matrices[m_labelsName[labelInfoOut]];
    if (labels == nullptr) return;
    
    labels->Resize(labelInfo.dim, actualmbsize);
    labels->SetValue(0);
    for (size_t j = 0; j < actualmbsize; ++j)
    {
        int    wrd = m_labelIdData[j];
        
        labels->SetValue(wrd, j, 1); 

    }

}

template<class ElemType>
void BatchLUSequenceReader<ElemType>::SetSentenceEndInBatch(vector<size_t> &sentenceEnd)
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
void BatchLUSequenceReader<ElemType>::SetSentenceEnd(int wrd, int pos, int actualMbSize)
{
    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelIdType index = GetIdFromLabel(labelIn.endSequence.c_str(), labelIn);

    if (pos == actualMbSize - 1) 
    {
        if (wrd == (int)index)
            mSentenceEnd = true;
        else
            mSentenceEnd = false; 
    }
}

template<class ElemType>
void BatchLUSequenceReader<ElemType>::SetSentenceBegin(int wrd, int pos, int /*actualMbSize*/)
{
    // now get the labels
    LabelInfo& labelIn = m_labelInfo[labelInfoIn];
    LabelIdType index = GetIdFromLabel(labelIn.beginSequence.c_str(), labelIn);

    if (pos == 0) 
    {
        if (wrd == (int)index)
            mSentenceBegin = true;
        else
            mSentenceBegin = false; 
    }
}


template<class ElemType>
bool BatchLUSequenceReader<ElemType>::DataEnd(EndDataType endDataType)
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

template class BatchLUSequenceReader<double>; 
template class BatchLUSequenceReader<float>;

}}}
