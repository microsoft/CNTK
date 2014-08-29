//
// <copyright file="UCIReader.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// UCIReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "File.h"
#define DATAREADER_EXPORTS  // creating the exports here
#include "DataReader.h"
#include "UCIReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType, typename LabelType>
size_t UCIReader<ElemType, LabelType>::RandomizeSweep(size_t epochSample)
{
    size_t randomRangePerEpoch = (m_epochSize+m_randomizeRange-1)/m_randomizeRange;
    return m_epoch*randomRangePerEpoch + epochSample/m_randomizeRange;
}

// ReadLine - Read a line
// readSample - sample to read in global sample space
// returns - true if we successfully read a record, otherwise false
template<class ElemType, typename LabelType>
bool UCIReader<ElemType, LabelType>::ReadRecord(size_t readSample)
{
    bool readRecord = false;
    File& file = *m_file;
    if (!file.IsEOF())
    {
        LabelType label;
        //std::wstring wstr;
        //file.GetLine(wstr);
        if (m_labelFirst && m_labelType != labelNone)
        {
            file >> label;
        }

        // get the sample index in this epoch (not global sample)
        size_t epochSample = readSample % m_epochSize;
        size_t idxFeature = epochSample*m_featureCount;

        int cntFeatures=0;
        bool read = true;   // set the read amount to something valid
        ElemType* feature = &m_featureData[idxFeature];
        while (cntFeatures < m_featureCount && read)
        {
            ElemType elem;

            // try and get an element, if it doesn't work (reading a string as an number, etc.) 
            // read will return 0, and we exit the loop
            read = file.TryGetText(elem);
            if (read)
                *feature++ = elem;
            ++cntFeatures;
        }
        
        // get end of line if it exists
        bool eol = file.EndOfLineOrEOF(true);

        // if label is last pop it off the vector
        if (!m_labelFirst && m_labelType != labelNone)
        {
            file >> label;
            if (!file.EndOfLineOrEOF(true))
                ERROR("end of line/file not found after label");
        }

        readRecord = true;

        // add the new values to the arrays 
        if (m_labelType == labelCategory)
        {
            // check to see if we have seen this label before
            auto value = m_mapLabelToId.find(label);
            LabelIdType labelId;
            if (value == m_mapLabelToId.end())
            {
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
            m_labelIdData[epochSample] = labelId;
        }
        else if (m_labelType != labelNone)
        {
            m_labelData[epochSample] = label;
        }
    }
    return readRecord; // we read a record
}

// EnsureDataAvailable - Read enough lines so we can request a minibatch starting as requested
// mbStartSample - the starting sample we are ensureing are good
// numberRead - [out] returns the actual number read
// returns - true if we have more to read, false if we hit the end of the dataset
template<class ElemType, typename LabelType>
bool UCIReader<ElemType, LabelType>::EnsureDataAvailable(size_t mbStartSample, size_t& numberRead)
{
    assert(mbStartSample >= m_epochStartSample);
    // determine how far ahead we need to read
    bool randomize = Randomize();
    // need to read to the end of the next minibatch
    size_t epochSample = mbStartSample;
    epochSample %= m_epochSize;

    // determine number left to read for this epoch
    size_t numberToRead = m_epochSize - epochSample;
    // we will take either a minibatch or the number left in the epoch
    numberToRead = min(numberToRead, m_mbSize);

    size_t randomRangePerEpoch = 1;
    if (randomize)
    {
        size_t randomizeSweep = RandomizeSweep(epochSample);
        // if first read or read takes us to another randomization range
        // we need to read at least randomization range records
        if (m_randomizeRange != randomizeAuto &&   // if we are randomizing and know the range
            randomizeSweep != m_randomordering.CurrentSeed()) // the range has changed since last time
        {
            numberToRead = m_randomizeRange;
        }
    }

    // check to see if we have the proper records read already
    if (m_readNextSample >= mbStartSample+numberToRead && mbStartSample >= m_epochStartSample)
        return true;

    // read in the samples
    File& file = *m_file;
    numberRead=0;
    bool readRecords = true;
    while (readRecords && numberRead < numberToRead)
    {
        size_t next = numberRead+1;
        if (!(next% 10000))
            fprintf(stderr,"#");
        else if (!(next% 1000))
            fprintf(stderr,"+");
        else if (!(next % 100))
            fprintf(stderr, ".");
        readRecords = ReadRecord(m_readNextSample);
        if (readRecords)
        {
            numberRead++;
            ++m_readNextSample;
            if (!m_endReached)
                ++m_totalSamples;   // total number of records in the dataset
        }
    }

    // if we hit the end of the records, we now have the total number of Samples in the dataset
    if (!readRecords)
    {
        UpdateDataVariables();
    }
    return readRecords;
}

// UpdateDataVariables - Update variables that depend on the dataset being completely read
template<class ElemType, typename LabelType>
void UCIReader<ElemType, LabelType>::UpdateDataVariables()
{
    // if we already reached the end before no need to set again.
    if (m_endReached)
        return;

    // get the size of the dataset
    assert(m_totalSamples*m_featureCount >= m_featureData.size());

    if (m_epochSize == requestDataSize)
        m_epochSize = m_totalSamples;

    // make sure randomization range is within the sample bounds
    if (m_randomizeRange > m_epochSize)
    {
        m_randomizeRange = m_epochSize;
        m_randomordering.resize(m_randomizeRange,m_randomizeRange);
    }

    // update the label dimension if it is not big enough, add something on
    if (m_labelType == labelCategory && m_labelIdMax > m_labelDim)
        m_labelDim = m_labelIdMax;  // update the label dimensions if different

    // we got to the end of the dataset
    m_endReached = true;
}

// Reader Initialize
// vdim - [out] number of elements in a single Sample of feature values (single precision values) 
// udim - [out] number of columns in the label matrix 
// filepaths - [in] and array of file paths to necessary files, it is variable depending on the reader 
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific 
// randomize - number of samples to randomize, defaults to randomizeAuto
template<class ElemType, typename LabelType>
void UCIReader<ElemType, LabelType>::Init(size_t& vdim, size_t& udim, const std::vector<std::wstring>& filepaths, const ConfigParameters& config)
{

    // initialize all the variables
    m_mbStartSample = m_epoch = m_totalSamples = m_epochStartSample = 0;
    m_labelIdMax = m_labelDim = 0; 
    m_partialMinibatch = m_labelFirst = m_endReached = false;
    m_labelType = labelCategory;
    m_featureCount = vdim;
    m_readNextSample = 0;

    // set the feature count to at least one (we better have one feature...)
    assert (m_featureCount != 0);

    fprintf(stderr, "reading uci file %ws", filepaths[0].c_str());
    m_file = new File(filepaths[0], fileOptionsRead | fileOptionsText | fileOptionsSequential);
    File& file = *m_file;

    ConfigParameters readerConfig = config("reader");
    if (readerConfig.Exists("randomize"))
    {
        string randomizeString = readerConfig("randomize");
        if (randomizeString == "None")
        {
            m_randomizeRange = randomizeNone;
        }
        else if (randomizeString == "Auto")
        {
            m_randomizeRange = randomizeAuto;
        }
        else
        {
            m_randomizeRange = readerConfig("randomize");
        }
    }
    else
    {
        m_randomizeRange = randomizeAuto;
    }

    // determine if we have first or last label
    std::string labelFirst(readerConfig("labelPosition","First"));
    m_labelFirst = labelFirst == "First";

    // determine if we partial minibatches are desired
    std::string minibatchMode(readerConfig("minibatchMode","Partial"));
    m_partialMinibatch = minibatchMode == "Parital";

    // determine if we partial minibatches are desired
    std::string labelType(readerConfig("labelType","Category"));
    if (labelType == "Category")
    {
        m_labelType = labelCategory;
    }
    else if (labelType == "Regression")
    {
        m_labelType = labelRegression;
    }
    else if (labelType == "None")
    {
        m_labelType = labelNone;
    }

    // if we know the size of the randomization now, resize, otherwise wait until we know the epochSize in StartMinibatchLoop()
    if (Randomize() && m_randomizeRange != randomizeAuto)
        m_randomordering.resize(m_randomizeRange, m_randomizeRange);

    // if the value they passed in as udim is not big enough, add something on
    if (udim < m_labelIdMax)
        udim = m_labelIdMax;
    m_labelDim = (LabelIdType)udim;
}

// destructor - virtual so it gets called properly 
template<class ElemType, typename LabelType>
UCIReader<ElemType, LabelType>::~UCIReader()
{
    delete m_file;
}

//SetupEpoch - Setup the proper position in the file, and other variable settings to start a particular epoch
template<class ElemType, typename LabelType>
void UCIReader<ElemType, LabelType>::SetupEpoch()
{
    size_t sweep = 0;
    size_t sweepsPerDS = 1;
    // if we know the total number of records
    if (m_endReached)
    {
        sweepsPerDS = (m_totalSamples+m_epochSize-1)/m_epochSize;
        sweep = m_epoch / sweepsPerDS;
    }
    else
    {   // don't know yet, haven't reached the end
        sweepsPerDS = m_epoch+1;
    }

    // if we need to start in the middle of the dataset, we better already be there
    if (m_epoch % sweepsPerDS != 0)
    {
        // make sure we are in the correct location already for mid-dataset epochs
        fprintf(stderr, "starting epoch %d midway through file at position %ld\n", m_epoch, m_mbStartSample);
        assert(m_mbStartSample % m_epochSize == 0);
        m_epochStartSample = m_mbStartSample;
        // future, we would need to seek to the proper location
    }
    else 
    {
        // starting over in the dataset
        m_readNextSample = m_epochStartSample = m_mbStartSample = m_epoch * m_epochSize;
        if (sweepsPerDS > 1)
        {
            // restarting an epoch at the beginning of the dataset
            fprintf(stderr, "restarting file read, for epoch %d\n", m_epoch);
            m_file->SetPosition(0);
        }
        else if (m_epoch > 0) // if we have read the data once already
        {
            assert(m_totalSamples <= m_epochSize);
            assert(m_featureData.size()/m_featureCount >= m_totalSamples);
            fprintf(stderr, "all data already resident for epoch %d\n", m_epoch);
            // move the read pointer to the end since we have everything already in memory.
            m_readNextSample += m_totalSamples;    
        }
    }
}

//StartMinibatchLoop - Startup a minibatch loop 
// mbSize - [in] size of the minibatch (number of Samples, etc.)
// epoch - [in] epoch number for this loop
// requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
template<class ElemType, typename LabelType>
void UCIReader<ElemType, LabelType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
{
    m_mbSize = mbSize;
    if (requestedEpochSamples == requestDataSize)
    {
        if (m_endReached)
        {
            m_epochSize = m_totalSamples;
        }
    }
    else
    {
        m_epochSize = requestedEpochSamples;
    }
    
    // set the randomization range for randomizationAuto
    // or if it's invalid less than the minibatch size, we need to make it at least minibatch size
    if (m_randomizeRange == randomizeAuto 
        || (m_randomizeRange != randomizeNone && m_randomizeRange < mbSize))
    {
        if (m_epochSize != requestDataSize)
        {
            m_randomizeRange = m_epochSize;
			m_randomordering.resize(m_randomizeRange, m_randomizeRange);
        } 
        else if (m_randomizeRange < mbSize)
		{
			m_randomizeRange = max(m_randomizeRange, m_mbSize);
			m_randomordering.resize(m_randomizeRange, m_randomizeRange);
		}
    }

    m_epoch = epoch;
    m_mbStartSample = epoch*m_epochSize;
    SetupEpoch();

    // allocate room for the data
    m_featureData.resize(m_featureCount*m_epochSize);
    if (m_labelType == labelCategory)
        m_labelIdData.resize(m_epochSize);
    else if (m_labelType != labelNone)
        m_labelData.resize(m_epochSize);
}

// GetMinibatch - Get the next minibatch 
// features - [out] returns minibatch in passed in matrix, will resize and replace existing data. Number of columns returned may be less than requested mbSize if end of dataset has been reached 
// labels - [out] returns matrix of label values as normalized integers (0-x) for class labels, and  will replace existing data. 
// return - true if we read some records to process, otherwise false;
template<class ElemType, typename LabelType>
bool UCIReader<ElemType, LabelType>::GetMinibatch(Matrix<ElemType>& features, Matrix<ElemType>& labels)
{
    // get out if they didn't call StartMinibatchLoop() first
    if (m_mbSize == 0)
        return false;

    // check to see if we have changed epochs, if so we are done with this one.
    if (m_mbStartSample / m_epochSize != m_epoch)
        return false;

    bool randomize = Randomize();
    size_t recordsRead = 0;
    bool moreData = EnsureDataAvailable(m_mbStartSample, recordsRead);

    // figure which sweep of the randomization we are on
    size_t epochSample = m_mbStartSample % m_epochSize; // where the minibatch starts in this epoch
    size_t samplesExtra = m_totalSamples % m_epochSize; // extra samples at the end of an epoch
    size_t epochsDS = (m_totalSamples+m_epochSize-1)/m_epochSize; // how many epochs per dataset
    size_t randomizeSet = randomize?RandomizeSweep(epochSample):0;
    const auto & tmap = m_randomordering(randomizeSet);
    size_t epochEnd = m_epochSize;

    // actual size is either what requested, or total number of samples read so far
    size_t actualmbsize = min(m_totalSamples, m_mbSize);   // it may still return less if at end of sweep

    // if we have extra records at the end of the dataset
    // and we are in the epoch where they would occur
    if (samplesExtra && !((m_epoch+1)%epochsDS))
    {
        epochEnd = samplesExtra; 
    }

    // check for an odd sized last minibatch
    if (epochSample + actualmbsize > epochEnd)
    {   
        actualmbsize = epochEnd - epochSample;
    }

    // hit the end of the dataset, so see how many records we REALLY got
    if (!moreData)
    {
        // we started a new epoch and hit the end of the file before we read any records so reset the epoch and keep going
        // we know this is the case when our epoch starts on the same sample as the next read sample and we are at the beginning.
        if (epochSample == 0 && m_epochStartSample == m_readNextSample)
        {
            SetupEpoch();
            moreData = EnsureDataAvailable(m_mbStartSample, recordsRead);
        }
        // if we are out of records return now
        else if (actualmbsize == 0)
        {
            return false;
        }
    }

    // if they don't want partial minibatches, skip and return
    if (actualmbsize < m_mbSize && !m_partialMinibatch)
    {
        m_mbStartSample += actualmbsize;
        return false;
    }

    // resize the features array to be big enough
    features.Resize(m_featureCount, actualmbsize);

    if (m_labelType == labelCategory)
    {
        // make the label array big enough, this should be a sparse array when that is supported
        labels.Resize(m_labelDim, actualmbsize);
        labels.SetValue((ElemType)0);
    }
    else if (m_labelType != labelNone)
    {
        labels.Resize(1, actualmbsize);
    }

    // loop through and copy data to matrix
    int j = 0; // vector of vectors of feature data
    // determine randomization base index
    size_t randBase;
    if (randomize)
        randBase = epochSample - epochSample%m_randomizeRange;

    for (size_t jSample = m_mbStartSample; j < actualmbsize; ++j, ++jSample)
    {
        // pick the right sample with randomization if desired
        size_t jRand = randomize?(randBase + tmap[jSample%m_randomizeRange]):jSample;
        jRand %= m_epochSize;
        size_t sampleCount = m_featureData[jRand*m_featureCount];
         
        // vector of feature data goes into matrix column
        for (int i = 0;i < m_featureCount; ++i)
        {
            features(i, j) = m_featureData[jRand*m_featureCount+i];
        }

        if (m_labelType == labelCategory)
        {
            // they all have to be in dimensions
            assert(m_labelIdData[jRand] < m_labelDim);    
            labels(m_labelIdData[jRand], j) = (ElemType)1;
        }
        else if (m_labelType != labelNone)
        {
            // how do we support string labels?
            labels(0, j) = m_labelData[jRand];
        }
    }

    // advance to the next minibatch
    m_mbStartSample += actualmbsize;

    // we read some records, so process them
    return true;
}

// GetLabelMapping - Gets the label mapping from integer index to label type 
// returns - a map from numeric datatype to native label type 
template<class ElemType, typename LabelType>
const std::map<unsigned, LabelType>& UCIReader<ElemType, LabelType>::GetLabelMapping( )
{
    return m_mapIdToLabel;
}

// SetLabelMapping - Sets the label mapping from integer index to label 
// labelMapping - mapping table from label values to IDs (must be 0-n)
// note: for tasks with labels, the mapping table must be the same between a training run and a testing run 
template<class ElemType, typename LabelType>
void UCIReader<ElemType, LabelType>::SetLabelMapping(const std::map<unsigned, LabelType>& labelMapping)
{
    m_mapIdToLabel = labelMapping;
    m_mapLabelToId.clear();
    for each (std::pair<unsigned, LabelType> var in labelMapping)
    {
        m_mapLabelToId[var.second] = var.first;
    }
}

// instantiate all the combinations we expect to be used
//template class UCIReader<double, std::wstring>; 
//template class UCIReader<float, std::wstring>;
template class UCIReader<float, int>;
template class UCIReader<double, int>;
template class UCIReader<float, float>;
template class UCIReader<double, double>;
}}}