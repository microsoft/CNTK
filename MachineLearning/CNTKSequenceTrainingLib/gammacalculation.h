#pragma once

#include <unordered_map>
#include "simplesenonehmm.h"
#include "latticearchive.h"
#include "latticesource.h"
#include "ssematrix.h"
#include "Matrix.h"

#pragma warning (disable: 4127) // conditional expression is constant

namespace msra { namespace lattices {
    template<class ElemType>
    class GammaCalculation
    {
        bool cpumode;
    public:
        GammaCalculation() : cpumode(false)
        {
            initialmark = false;
            lmf = 14.0f; // Note that 9 was best for Fisher  --these should best be configurable
            wp = 0.0f;
            amf = 14.0f;
            boostmmifactor = 0.0f;
            seqsMBRmode = false;
        }
        ~GammaCalculation()
        {

        }

        void init(msra::asr::simplesenonehmm hset, int DeviceId)
        {
            m_deviceid = DeviceId;
            if (!initialmark)
            {
                m_hset = hset;
                m_maxframenum = 0;

                // prep for parallel implementation (CUDA)
                parallellattice.setdevice(DeviceId);
                    
                if (parallellattice.enabled())                   // send hmm set to GPU if GPU computation enabled
                    parallellattice.entercomputation(m_hset, mbrclassdef);       // cache senone2classmap if mpemode 
                initialmark = true;
            }
        }
            
            
        void calgammaformb(Microsoft::MSR::CNTK::Matrix<ElemType>& functionValues, std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> &lattices, const Microsoft::MSR::CNTK::Matrix<ElemType>& loglikelihood,
            Microsoft::MSR::CNTK::Matrix<ElemType>&  labels, Microsoft::MSR::CNTK::Matrix<ElemType>& gammafromlattice, std::vector<size_t> &uids, std::vector<size_t> &boundaries,
            size_t samplesInRecurrentStep, std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> pMBLayout, std::vector<size_t> &extrauttmap, bool doreferencealign)
        {
            //check total frame number to be added ?
            //int deviceid = loglikelihood.GetDeviceId();
            size_t boundaryframenum;
            std::vector<size_t> validframes;
            validframes.assign(samplesInRecurrentStep, 0);
            ElemType objectValue = 0.0;
            //convert from Microsoft::MSR::CNTK::Matrix to  msra::math::ssematrixbase
            size_t numrows = loglikelihood.GetNumRows();
            size_t numcols = loglikelihood.GetNumCols();                
            Microsoft::MSR::CNTK::Matrix<ElemType> tempmatrix(m_deviceid);
                
            //copy loglikelihood to pred
            if (numcols > pred.cols())
            {
                pred.resize(numrows, numcols);
                dengammas.resize(numrows, numcols);
            }

            if (doreferencealign)
                labels.SetValue((ElemType)(0.0f));
                
            size_t mbsize = numcols / samplesInRecurrentStep;                
            if (samplesInRecurrentStep > 1)
            {
                assert(extrauttmap.size() == lattices.size());
                assert(mbsize == pMBLayout->GetNumTimeSteps());
            }
                
            size_t mapi = 0;
            size_t mapframenum = 0;
            //cal gamma for each utterance
            size_t ts = 0;
            //size_t ts_uid = 0;                
            for (size_t i = 0; i < lattices.size(); i++)
            {
                const size_t numframes = lattices[i]->getnumframes();

                msra::dbn::matrixstripe predstripe(pred, ts, numframes);           // logLLs for this utterance                    
                msra::dbn::matrixstripe dengammasstripe(dengammas, ts, numframes); // denominator gammas

                                        
                if (samplesInRecurrentStep == 1)  //one channel 
                {
                    tempmatrix = loglikelihood.ColumnSlice(ts, numframes);
                    if (m_deviceid == CPUDEVICE)
                    {
                        CopyFromCNTKMatrixToSSEMatrix(tempmatrix, numframes, predstripe);
                    }
                    else
                        parallellattice.setloglls(tempmatrix);
                }
                else                   //multi channel
                {
                    //get frame number for each utterance
                    mapi = extrauttmap[i];
                        
                    for (size_t j = validframes[mapi]; j < mbsize; j++)
                    {
                        if (pMBLayout->Is(mapi,j, MinibatchPackingFlags::SequenceEnd))
                        {
                            mapframenum = j - validframes[mapi] + 1;
                            break;
                        }
                    }                    

                        
                    assert(numframes == mapframenum);

                    if (numframes > tempmatrix.GetNumCols())
                        tempmatrix.Resize(numrows, numframes);

                    for (size_t nframe = 0; nframe < numframes; nframe++)
                    {
                        Microsoft::MSR::CNTK::Matrix<ElemType> columndata = loglikelihood.ColumnSlice((nframe + validframes[mapi])*samplesInRecurrentStep + mapi, 1);
                        tempmatrix.SetColumn(columndata, nframe);
                    }

                    //if (doreferencealign || m_deviceid == CPUDEVICE)
                    {
                        CopyFromCNTKMatrixToSSEMatrix(tempmatrix, numframes, predstripe);
                    }

                    if (m_deviceid != CPUDEVICE)
                    {                            
                        parallellattice.setloglls(tempmatrix);
                    }
                }
                    
                array_ref<size_t> uidsstripe(&uids[ts], numframes);
                    
                if (doreferencealign)
                {
                    boundaryframenum = numframes;                        
                }
                else
                    boundaryframenum = 0;
                array_ref<size_t> boundariesstripe(&boundaries[ts], boundaryframenum);                    
                    
                double numavlogp = 0;
                foreach_column(t, dengammasstripe)     // we do not allocate memory for numgamma now, should be the same as numgammasstripe
                {
                    const size_t s = uidsstripe[t ];
                    numavlogp += predstripe(s, t) / amf;
                }
                numavlogp /= numframes;
                    
                //auto_timer dengammatimer;
                double denavlogp = lattices[i]->second.forwardbackward(parallellattice,
                    (const msra::math::ssematrixbase &) predstripe, (const msra::asr::simplesenonehmm &) m_hset,
                    (msra::math::ssematrixbase &) dengammasstripe, (msra::math::ssematrixbase &) gammasbuffer/*empty, not used*/,
                    lmf, wp, amf, boostmmifactor, seqsMBRmode, uidsstripe, boundariesstripe);
                objectValue += (ElemType)((numavlogp - denavlogp) * numframes);
                   
                if (samplesInRecurrentStep == 1)
                {
                    tempmatrix = gammafromlattice.ColumnSlice(ts, numframes);
                }

                //copy gamma to tempmatrix
                if (m_deviceid == CPUDEVICE)
                {
                    CopyFromSSEMatrixToCNTKMatrix(dengammas, numrows, numframes, tempmatrix, gammafromlattice.GetDeviceId());
                }
                else
                    parallellattice.getgamma(tempmatrix);

                // set gamma for multi channel
                if (samplesInRecurrentStep > 1)
                {
                    Microsoft::MSR::CNTK::Matrix<ElemType> gammaFromLatticeForCurrentParallelUtterance = gammafromlattice.ColumnSlice(mapi + (validframes[mapi] * samplesInRecurrentStep), ((numframes - 1) * samplesInRecurrentStep) + 1);
                    gammaFromLatticeForCurrentParallelUtterance.CopyColumnsStrided(tempmatrix, numframes, 1, samplesInRecurrentStep);
                }

                if (doreferencealign)
                {
                    for (size_t nframe = 0; nframe < numframes; nframe++)
                    {
                        size_t uid = uidsstripe[nframe];
                        if (samplesInRecurrentStep > 1)
                            labels(uid, (nframe + validframes[mapi])*samplesInRecurrentStep + mapi) = 1.0;
                        else
                            labels(uid, ts+nframe) = 1.0;
                    }
                }
                if (samplesInRecurrentStep > 1)
                    validframes[mapi] += numframes;
                fprintf(stderr, "dengamma value %f\n", denavlogp);
                ts += numframes;
            }       
            functionValues.SetValue(objectValue);
        }

    private:
        // Helper methods for copying between ssematrix objects and CNTK matrices

        // Helper methods for copying to and from CNTK matrices
        void CopyFromCNTKMatrixToSSEMatrix(const Microsoft::MSR::CNTK::Matrix<ElemType>& src, size_t numCols, msra::math::ssematrixbase& dest)
        {
            if (!std::is_same<ElemType, float>::value)
            {
                LogicError("Cannot copy between a SSE matrix and a non-float type CNTK Matrix object!");
            }

            size_t numRows = src.GetNumRows();

            // TODO: Used piined memory for faster copies?
            // TODO: Can we cache the scratch buffer and reuse across copies
            ElemType* cpuSrcCopy = src.CopyToArray();
            if ((dest.getcolstride() == dest.rows()) && (numRows == dest.rows()))
            {
                memcpy(&dest(0, 0), (float*)cpuSrcCopy, sizeof(ElemType) * numRows * numCols);
            }
            else
            {
                // We need to copy columnwise
                for (size_t i = 0; i < numCols; ++i)
                {
                    memcpy(&dest(0, i), (float*)(cpuSrcCopy + (i * numRows)), sizeof(ElemType) * numRows);
                }
            }

            delete[] cpuSrcCopy;
        }

        void CopyFromSSEMatrixToCNTKMatrix(const msra::math::ssematrixbase& src, size_t numRows, size_t numCols, Microsoft::MSR::CNTK::Matrix<ElemType>& dest, int deviceId)
        {
            if (!std::is_same<ElemType, float>::value)
            {
                LogicError("Cannot copy between a SSE matrix and a non-float type CNTK Matrix object!");
            }

            // TODO: Used piined memory for faster copies?
            // TODO: Can we cache the scratch buffer and reuse across copies
            ElemType* cpuDestCopy = new ElemType[numRows * numCols];
            if ((src.getcolstride() == src.rows()) && (numRows == src.rows()))
            {
                memcpy((float*)cpuDestCopy, &src(0, 0), sizeof(float) * numRows * numCols);
            }
            else
            {
                // We need to copy columnwise
                for (size_t i = 0; i < numCols; ++i)
                {
                    memcpy((float*)(cpuDestCopy + (i * numRows)), &src(0, i), sizeof(float) * numRows);
                }
            }

            dest.SetValue(numRows, numCols, deviceId, cpuDestCopy, 0);
            delete[] cpuDestCopy;
        }
            
    protected:
        msra::asr::simplesenonehmm m_hset;
        msra::lattices::lattice::parallelstate parallellattice;
        msra::lattices::mbrclassdefinition mbrclassdef = msra::lattices::senone;    // defines the unit for minimum bayesian risk
        bool initialmark;
        msra::dbn::matrix dengammas;
        msra::dbn::matrix pred;
        int m_deviceid;  //-1: cpu
        size_t m_maxframenum;
        float lmf ; // Note that 9 was best for Fisher  --these should best be configurable
        float wp ;
        float amf;
        msra::dbn::matrix gammasbuffer;
        vector<size_t> boundary;
        float boostmmifactor;
        bool seqsMBRmode;
    };
}}
