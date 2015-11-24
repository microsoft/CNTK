//
// <copyright file="HTKMLFWriter.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// HTKMLFReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "basetypes.h"

#include "htkfeatio.h"                  // for reading HTK features

//#ifndef __unix__
#include "ssematrix.h"
//#endif
//#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
//#include "simplesenonehmm.h"            // for MMI scoring
//#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

//#include "rollingwindowsource.h"        // minibatch sources
//#include "utterancesource.h"
//#include "readaheadsource.h"
//#include "chunkevalsource.h"
//#include "minibatchiterator.h"

#define DATAWRITER_EXPORTS  // creating the exports here
#include "DataWriter.h"
#include "commandArgUtil.h"
#include "HTKMLFWriter.h"

#include "commandArgUtil.h"
#ifdef LEAKDETECT
#include <vld.h> // for memory leak detection
#endif



namespace Microsoft { namespace MSR { namespace CNTK {

    // Create a Data Writer
    //DATAWRITER_API IDataWriter* DataWriterFactory(void)

    template<class ElemType>
    void HTKMLFWriter<ElemType>::Init(const ConfigParameters& writerConfig)
    {
        m_tempArray = nullptr;
        m_tempArraySize = 0;

        vector<wstring> scriptpaths;
        vector<wstring> filelist;
        size_t numFiles;
        size_t firstfilesonly = SIZE_MAX;   // set to a lower value for testing

        ConfigArray outputNames = writerConfig(L"outputNodeNames","");
        if (outputNames.size()<1)
            RuntimeError("writer needs at least one outputNodeName specified in config");

        foreach_index(i, outputNames) // inputNames should map to node names
        {
            ConfigParameters thisOutput = writerConfig(outputNames[i]);

            if (thisOutput.Exists("dim"))
                udims.push_back(thisOutput(L"dim"));
            else
                RuntimeError("HTKMLFWriter::Init: writer need to specify dim of output");
            if (thisOutput.Exists("file"))
                scriptpaths.push_back(thisOutput(L"file"));
            else if (thisOutput.Exists("scpFile"))
                scriptpaths.push_back(thisOutput(L"scpFile"));
            else
                RuntimeError("HTKMLFWriter::Init: writer needs to specify scpFile for output");
            
            if (thisOutput.Exists("Kaldicmd"))
            {
                kaldicmd.push_back(thisOutput(L"Kaldicmd"));
            }
 
            outputNameToIdMap[outputNames[i]]= i;
            outputNameToDimMap[outputNames[i]]=udims[i];
            wstring type = thisOutput(L"type","Real");
            if (type == L"Real")
            {
                outputNameToTypeMap[outputNames[i]] = OutputTypes::outputReal;
            }
            else
            {
                throw std::runtime_error ("HTKMLFWriter::Init: output type for writer output expected to be Real");
            }
        }

        numFiles=0;
        foreach_index(i,scriptpaths)
        {
            filelist.clear();
            std::wstring scriptPath = scriptpaths[i];
            fprintf(stderr, "HTKMLFWriter::Init: reading output script file %S ...", scriptPath.c_str());
            size_t n = 0;
            for (msra::files::textreader reader(scriptPath); reader && filelist.size() <= firstfilesonly/*optimization*/; )
            {
                filelist.push_back (reader.wgetline());
                n++;
            }

            fprintf (stderr, " %zd entries\n", n);

            if (i==0)
                numFiles=n;
            else
                if (n!=numFiles)
                    throw std::runtime_error (msra::strfun::strprintf ("HTKMLFWriter:Init: number of files in each scriptfile inconsistent (%d vs. %d)", numFiles,n));

            outputFiles.push_back(filelist);
        }
        outputFileIndex=0;
        sampPeriod=100000;

    }

    template<class ElemType>
    void HTKMLFWriter<ElemType>::Destroy()
    {
        delete [] m_tempArray;
        m_tempArray = nullptr;
        m_tempArraySize = 0;
    }

    template<class ElemType>
    void HTKMLFWriter<ElemType>::GetSections(std::map<std::wstring, SectionType, nocase_compare>& /*sections*/)
    {
    }

    template<class ElemType>
    bool HTKMLFWriter<ElemType>::SaveData(size_t /*recordStart*/, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t /*numRecords*/, size_t /*datasetSize*/, size_t /*byteVariableSized*/)
    {
        
        if (kaldicmd.size() == 0)
        {
            //std::map<std::wstring, void*, nocase_compare>::iterator iter;
            if (outputFileIndex>=outputFiles[0].size())
                RuntimeError("index for output scp file out of range...");

            for (auto iter = matrices.begin();iter!=matrices.end(); iter++)
            {
                wstring outputName = iter->first;
                Matrix<ElemType>& outputData = *(static_cast<Matrix<ElemType>*>(iter->second));
                size_t id = outputNameToIdMap[outputName];
                size_t dim = outputNameToDimMap[outputName];
                wstring outFile = outputFiles[id][outputFileIndex];
            
                assert(outputData.GetNumRows()==dim); dim;

                SaveToKaldiFile(outFile,outputData);
            }

            outputFileIndex++;
        } else
        {
            if (outputFileIndex>=outputFiles[0].size())
                RuntimeError("index for output scp file out of range...");

            for (auto iter = matrices.begin();iter!=matrices.end(); iter++)
            {
                wstring outputName = iter->first;
                Matrix<ElemType>& outputData = *(static_cast<Matrix<ElemType>*>(iter->second));
                size_t id = outputNameToIdMap[outputName];
                size_t dim = outputNameToDimMap[outputName];
                wstring outFile = outputFiles[id][outputFileIndex];
                string wfea = "ark:" + msra::strfun::utf8(outFile);
                
                wfea = msra::strfun::utf8(kaldicmd[0]);
                kaldi::BaseFloatMatrixWriter feature_writer(wfea);
                kaldi::Matrix<kaldi::BaseFloat> nnet_out_host;
   
                assert(outputData.GetNumRows()==dim); dim;
                const std::string outputPath = msra::strfun::utf8(outFile);
                const std::string file_key = removeExtension(basename(outputPath));

                nnet_out_host.Resize(outputData.GetNumCols(), outputData.GetNumRows());
                outputData.CopyToArray(m_tempArray, m_tempArraySize);
                ElemType * pValue = m_tempArray;

           
                for (int j=0; j< outputData.GetNumCols(); j++)
                {
                    for (int i=0; i<outputData.GetNumRows(); i++)
                    {
                        nnet_out_host(j,i) = (float)*pValue++;                
                        if (nnet_out_host(j,i) > 50)
                        {
                            nnet_out_host(j,i)  = -(float)log(1.0/outputData.GetNumCols());
                        }


                    }
                }
                
                fprintf (stderr, "evaluate: writing %zd frames of %s\n", outputData.GetNumCols(), wfea.c_str());
                feature_writer.Write(file_key, nnet_out_host);

 
            }

            outputFileIndex++;

        }
        return true;
    }

    template<class ElemType>
    void HTKMLFWriter<ElemType>::SaveToFile(std::wstring& outputFile, Matrix<ElemType>& outputData)
    {
        msra::dbn::matrix output;
        output.resize(outputData.GetNumRows(),outputData.GetNumCols());
        outputData.CopyToArray(m_tempArray, m_tempArraySize);
        ElemType * pValue = m_tempArray;

        for (int j=0; j< outputData.GetNumCols(); j++)
            {
                for (int i=0; i<outputData.GetNumRows(); i++)
                {
                    output(i,j) = (float)*pValue++;                
                }
            }
            
        const size_t nansinf = output.countnaninf();
        if (nansinf > 0)
            fprintf (stderr, "chunkeval: %d NaNs or INF detected in '%S' (%d frames)\n", (int) nansinf, outputFile.c_str(), (int) output.cols());
        // save it
        msra::files::make_intermediate_dirs (outputFile);
        msra::util::attempt (5, [&]()
        {
            msra::asr::htkfeatwriter::write (outputFile, "USER", this->sampPeriod, output);
        });
                        
        fprintf (stderr, "evaluate: writing %zd frames of %S\n", output.cols(), outputFile.c_str());


    }

    template<class ElemType>
    void HTKMLFWriter<ElemType>::SaveToKaldiFile(std::wstring& outputFile, Matrix<ElemType>& outputData)
    {
        msra::dbn::matrix output;
        output.resize(outputData.GetNumRows(),outputData.GetNumCols());
        outputData.CopyToArray(m_tempArray, m_tempArraySize);
        ElemType * pValue = m_tempArray;

        for (int j=0; j< outputData.GetNumCols(); j++)
            {
                for (int i=0; i<outputData.GetNumRows(); i++)
                {
                    output(i,j) = (float)*pValue++;                
                }
            }
            
        const size_t nansinf = output.countnaninf();
        if (nansinf > 0)
            fprintf (stderr, "chunkeval: %d NaNs or INF detected in '%S' (%d frames)\n", (int) nansinf, outputFile.c_str(), (int) output.cols());
        // save it
        msra::files::make_intermediate_dirs (outputFile);
        msra::util::attempt (5, [&]()
        {
            msra::asr::htkfeatwriter::writeKaldi (outputFile, "USER", this->sampPeriod, output, sizeof(ElemType));
        });
                        
        fprintf (stderr, "evaluate: writing %zd frames of %S\n", output.cols(), outputFile.c_str());
    }



    template<class ElemType>
    void HTKMLFWriter<ElemType>::SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& /*labelMapping*/)
    {
    }
   
    template class HTKMLFWriter<float>;
    template class HTKMLFWriter<double>;

}}}
