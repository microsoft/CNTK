#pragma once

#include <unordered_map>
#include "simplesenonehmm.h"
#include "latticearchive.h"
#include "latticesource.h"
#include "ssematrix.h"
#include "Matrix.h"

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
                if (DeviceId == CPUDEVICE)
                    parallellattice.setmode(true);
                else
                {
                        
                    parallellattice.setmode(false);
                }
                        
                    
                if (parallellattice.enabled())                   // send hmm set to GPU if GPU computation enabled
                    parallellattice.entercomputation(m_hset, mbrclassdef);       // cache senone2classmap if mpemode 
                initialmark = true;
            }
        }
        //void init(msra::asr::simplesenonehmm hset, int DeviceId);
            
            
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
                assert(mbsize == pMBLayout->GetSize());
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
                        ElemType *datap = tempmatrix.CopyToArray();
                        memcpy(&predstripe(0, 0), (float *)datap, sizeof(float)*numrows*numframes);
                        delete datap;
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
                        ElemType *datap = tempmatrix.CopyToArray();
                        memcpy(&predstripe(0, 0), (float *)datap, sizeof(float)*numrows*numframes);
                        delete datap;
                    }
                    if (m_deviceid != CPUDEVICE)
                    {                            
                        parallellattice.setloglls(tempmatrix);
                    }

                    /*size_t pnumrow = logllmatrix.GetNumRows();
                    size_t pnumcol = logllmatrix.GetNumCols();
                    logllmatrix.Print("data value 1", 0, min(10, pnumrow) - 1, 0, min(10, pnumcol) - 1);
                    logllmatrix.Print("data value 2", 0, min(10, pnumrow) - 1, pnumcol - 11, pnumcol - 1);
                    float fnorm = (float)(logllmatrix.FrobeniusNorm());
                    fprintf(stderr, "fnorm %f\n", fnorm);*/
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
                    ElemType * outp = new ElemType[numrows*numframes];
                    memcpy((float *)outp, &dengammas(0, 0), sizeof(float)*numrows*numframes);                        
                    tempmatrix.SetValue(numrows, numframes, outp, 0, gammafromlattice.GetDeviceId());
                    delete outp;
                }
                else
                    parallellattice.getgamma(tempmatrix);

                // set gamma for multi channel
                if (samplesInRecurrentStep > 1)
                {                        
                    for (size_t nframe = 0; nframe < numframes; nframe++)
                    {
                        Microsoft::MSR::CNTK::Matrix<ElemType> columndata = tempmatrix.ColumnSlice(nframe, 1);
                        gammafromlattice.SetColumn(columndata, (nframe + validframes[mapi])*samplesInRecurrentStep + mapi);
                    }
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
                /*if (samplesInRecurrentStep == 1)
                    ts += numframes;
                else
                    ts = (i+1) * mbsize;*/
                ts += numframes;
            }       
            functionValues.SetValue(objectValue);
            // parallellattice.release(false);
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
