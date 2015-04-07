

#include "CppUnitTest.h"
#include "Eval.h"
#include "../../DataReader/HTKMLFReader/htkfeatio.h"
#include <memory>
#include <string>
#include <vector>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace Microsoft::MSR::CNTK;
using msra::asr::htkfeatreader;

typedef void(*GetEvalProc)(IEvaluateModel<float>** peval);

namespace EvalUnitTest
{
    class HTKUtteranceReader    // designed for read one utterance 
    {
    public:
        HTKUtteranceReader() : m_verbosity(0), m_contextWindow(5), m_frameshift(0){

        }
        void SetContextWindow(size_t windowSize) { m_contextWindow = windowSize; }
        void SetFrameShift(int shift){ m_frameshift = shift;    }
        size_t  GetUtteranceData(wstring filename, vector<float>& utterance)
            // for a D (dim) X N (frame) utterance (specified by filename) and a window size H 
            // utterance returns a D*(2*H+1) * N vector 
            // caller don't need to resize utterance 
        {

            //========================================
            // Sec. 1 get basic info 
            //========================================
            htkfeatreader reader; 
            htkfeatreader::parsedpath path(filename);
            
            string featKind; 
            size_t featDim; 
            size_t nframes;
            unsigned int featPeriod = 0; 
            reader.getinfo(path, featKind, featDim, featPeriod, nframes);

            //========================================
            // Sec. 2 resize output vector 
            //========================================
            size_t expandedDim = (2 * m_contextWindow + 1) * featDim;       // dimension after contex expanding 
            size_t totalDim = expandedDim * nframes;                        // dimension of the vector we want to feed to CNTK eval
            utterance.resize(totalDim, 0.0f);
            //========================================
            // Sec. 3 get the feature matrix 
            //========================================
            vector<vector<float> > featMatrix;
            featMatrix.resize(nframes);
            for (size_t n = 0; n < nframes; n++)
            {
                reader.read(featMatrix[n]);
            }
            //========================================
            // Sec. 4 context expansion
            //========================================
            if (m_contextWindow > 0)
            {
                vector<float>  expandedfeat;
                expandedfeat.resize(expandedDim);

                for (size_t n = 0; n < nframes; n++)
                {
                    
                    size_t despos = 0; 
                    for (size_t ileft = m_contextWindow-1; ileft >=0; ileft--, despos ++ )
                    {
                        size_t srcpos = max(n - ileft, 0);                         
                        copy(featMatrix[srcpos].begin(), featMatrix[srcpos].end(), expandedfeat.begin() + featDim *despos);
                    }
                    for (size_t iRight = 1; iRight < m_contextWindow; iRight++, despos++)
                    {
                        size_t srcpos = min(n + iRight, nframes);
                        copy(featMatrix[srcpos].begin(), featMatrix[srcpos].end(), expandedfeat.begin() + featDim *despos); 
                    }
                    copy(expandedfeat.begin(), expandedfeat.end(), utterance.begin() + expandedDim*n);
                }
            }
            else
            {
                size_t idx = 0;
                for (auto v : featMatrix)
                {
                    copy(v.begin(), v.end(), utterance.begin() + expandedDim*idx);
                    idx++;
                }
            }



            //========================================
            // Sec. 5 frame shift 
            //========================================
            if (m_frameshift > 0)
            {
                assert(m_frameshift < nframes);  // TODO: throw an exception
                /*
                for (size_t f = m_frameshift; f < nframes; f++)
                {
                    copy(featMatrix[f].begin(), featMatrix[f].end(), utterance.begin() + (f-m_frameshift)*expandedDim);
                }
                for (size_t r = 0; r < m_frameshift; r++)
                {
                    copy(featMatrix[nframes - 1].begin(), featMatrix[nframes - 1].end(), utterance.begin()+(nframes-m_frameshift+r)*expandedDim );
                }*/
                for (size_t ipos = nframes - m_frameshift -1; ipos >= 0; ipos++)
                {
                    auto src = utterance.begin() + ipos * expandedDim;
                    auto des = src + (m_frameshift)*expandedDim;
                    copy(src, src + expandedDim, des);
                }

                auto src = utterance.begin() + m_frameshift*expandedDim;
                for (size_t ipos = 0; ipos < m_frameshift; ipos++)
                {
                    auto des = utterance.begin() + ipos * expandedDim;
                    copy(src, src + expandedDim, des);
                }
            }
            return nframes;

        }

    private: 
        size_t m_contextWindow;         // =5 means 5:1:5 TODO support asymmetric context window
        size_t m_verbosity; 
        short  m_frameshift;            // f>0  means x[0..n-1] -> [ x[f..n-1], x[n-1], ... ] 
                                        // f<0  means x[0..n-1] -> [ x[0], x[0],... x[f..n-1] ] 
                                        // frameshift is applied after context windows expansion 
    };



    TEST_CLASS(CNTKEvalUnitTest)
    {
    private:
        bool LoadEvalutor()
        {
            try
            {
                Plugin plug;
                GetEvalProc p_Processor = (GetEvalProc)plug.Load(m_strEvalModule, m_strGetEvalProcessor);
                IEvaluateModel<float>* pEval = nullptr;
                p_Processor(&pEval);
                m_pEval.reset(pEval);
            }
            catch (std::exception& e)
            {
                printf("ERROR: %s\n", e.what());
                Assert::Fail(L"Load Module failed");
            }
            return true;
        }

    public:
        TEST_CLASS_INITIALIZE(ConfigEvalUnitTest)
        {

        }

        TEST_METHOD(LoadModule)
        {
            Assert::IsTrue(LoadEvalutor());
        }

        TEST_METHOD(RNNModelTest)
        {
            Assert::IsTrue(LoadEvalutor());
            m_pEval->Init("deviceID=-1");
            m_pEval->LoadModel(m_strRNNModel);

            //========================================
            // Sec. 1 query node dimension
            //========================================
            map<wstring, size_t> inputDimensions;
            m_pEval->GetNodeDimensions(inputDimensions, nodeInput);
            map<wstring, size_t> outputDimensions; 
            m_pEval->GetNodeDimensions(outputDimensions, nodeOutput);

            //========================================
            // Sec. 2 load in feature files and expected output files 
            //========================================            
            HTKUtteranceReader reader; 
            reader.SetFrameShift(0);
            reader.SetContextWindow(0);
            vector<float> utterData; 
            size_t nframe=reader.GetUtteranceData(m_strRNNModelInputOutputPairs.first, utterData);
            vector<float> expectedRes;
            size_t nframeLLR = reader.GetUtteranceData(m_strRNNModelInputOutputPairs.second, expectedRes);
            Assert::AreEqual(nframe, nframeLLR, L"ERROR: number of input feature vectors mismatched with the number of output vectors");

            //========================================
            // Sec. 3 setup inputs and outputs 
            //========================================
            Assert::IsTrue(inputDimensions.find(L"features") != inputDimensions.end());
            Assert::IsTrue(outputDimensions.find(L"ScaledLogLikelihood") != outputDimensions.end());

            map<wstring, vector<float>* > inputs; 
            inputs[L"features"] = &utterData;
            map<wstring, vector<float>* > outputs; 
            size_t outputDim = nframe * outputDimensions[L"ScaledLogLikelihood"];
            vector<float> outputData(outputDim,0.0f);
            outputs[L"ScaledLogLikelihood"] = &outputData;
            m_pEval->Evaluate(inputs, outputs);

            //========================================
            // Sec. 4 CompareResults 
            //========================================
            wstring msg = msra::strfun::wstrprintf(L"Output dimensions (%d vs %d) do not match", outputs.size(), expectedRes.size());
            Assert::AreEqual(expectedRes.size(), outputs[L"ScaledLogLikelihood"]->size(), L"Output dimension does not match");
            
            for (size_t i = 0; i < expectedRes.size(); i++)
            {
                Assert::AreEqual(expectedRes[i], (*outputs[L"ScaledLogLikelihood"])[i], 0.01f);
            }

        }


    private:
        shared_ptr<IEvaluateModel<float>>   m_pEval;
        static const wstring    m_strEvalModule;
        static const  string    m_strGetEvalProcessor;
        static const wstring    m_strRNNModel; 
        static const pair<wstring, wstring>   m_strRNNModelInputOutputPairs;


    };

    const wstring  CNTKEvalUnitTest::m_strEvalModule = L"CNTKEval";           // the module (.dll) name 
    const  string  CNTKEvalUnitTest::m_strGetEvalProcessor = "GetEvalF";     // which function to get eval 
    const wstring  CNTKEvalUnitTest::m_strRNNModel = L"..\\..\\..\\AutoTest\\resources\\rnn.model.40";
    const pair<wstring, wstring>  CNTKEvalUnitTest::m_strRNNModelInputOutputPairs = make_pair(L"..\\..\\..\\AutoTest\\resources\\LFB87dim.mfc", L"..\\..\\..\\AutoTest\\resources\\LFB87dim.pos");
    

}