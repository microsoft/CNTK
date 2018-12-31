#pragma once

#include <unordered_map>
#include "simplesenonehmm.h"
#include "latticearchive.h"
#include "latticesource.h"
#include "ssematrix.h"
#include "Matrix.h"
#include "CUDAPageLockedMemAllocator.h"

#include <memory>
#include <vector>

#pragma warning(disable : 4127) // conditional expression is constant

namespace msra { namespace lattices {

struct SeqGammarCalParam
{
    double amf;
    double lmf;
    double wp;
    double bMMIfactor;
    bool sMBRmode;
    SeqGammarCalParam()
    {
        amf = 14.0;
        lmf = 14.0;
        wp = 0.0;
        bMMIfactor = 0.0;
        sMBRmode = false;
    }
};

template <class ElemType>
class GammaCalculation
{
    bool cpumode;

public:
    GammaCalculation()
        : cpumode(false)
    {
        initialmark = false;
        lmf = 7.0f; // Note that 9 was best for Fisher  --these should best be configurable
        wp = 0.0f;
        amf = 7.0f;
        boostmmifactor = 0.0f;
        seqsMBRmode = false;
    }
    ~GammaCalculation()
    {
    }

    // ========================================
    // Sec. 1 init functions
    // ========================================
    void init(msra::asr::simplesenonehmm hset, int DeviceId)
    {
        m_deviceid = DeviceId;
        if (!initialmark)
        {
            m_hset = hset;
            m_maxframenum = 0;

            // prep for parallel implementation (CUDA)
            parallellattice.setdevice(DeviceId);
            if (parallellattice.enabled())                             // send hmm set to GPU if GPU computation enabled
                parallellattice.entercomputation(m_hset, mbrclassdef); // cache senone2classmap if mpemode
            initialmark = true;
        }
    }

    // ========================================
    // Sec. 2 set functions
    // ========================================
    void SetGammarCalculationParams(const SeqGammarCalParam& gammarParam)
    {
        lmf = (float) gammarParam.lmf;
        amf = (float) gammarParam.amf;
        wp = (float) gammarParam.wp;
        seqsMBRmode = gammarParam.sMBRmode;
        boostmmifactor = (float) gammarParam.bMMIfactor;
    }

    // ========================================
    // Sec. 3 calculation functions
    // ========================================
    void calgammaformb(Microsoft::MSR::CNTK::Matrix<ElemType>& functionValues,
                       std::vector<std::shared_ptr<const msra::dbn::latticepair>>& lattices,
                       const Microsoft::MSR::CNTK::Matrix<ElemType>& loglikelihood,
                       Microsoft::MSR::CNTK::Matrix<ElemType>& labels,
                       Microsoft::MSR::CNTK::Matrix<ElemType>& gammafromlattice,
                       std::vector<size_t>& uids, std::vector<size_t>& boundaries,
                       size_t samplesInRecurrentStep, /* numParallelUtterance ? */
                       std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> pMBLayout,
                       std::vector<size_t>& extrauttmap,
                       bool doreferencealign)
    {
        // check total frame number to be added ?
        // int deviceid = loglikelihood.GetDeviceId();
        size_t boundaryframenum;
        std::vector<size_t> validframes; // [s] cursor pointing to next utterance begin within a single parallel sequence [s]
        validframes.assign(samplesInRecurrentStep, 0);
        ElemType objectValue = 0.0;
        // convert from Microsoft::MSR::CNTK::Matrix to  msra::math::ssematrixbase
        size_t numrows = loglikelihood.GetNumRows();
        size_t numcols = loglikelihood.GetNumCols();
        Microsoft::MSR::CNTK::Matrix<ElemType> tempmatrix(m_deviceid);

        // copy loglikelihood to pred
        if (numcols > pred.cols())
        {
            pred.resize(numrows, numcols);
            dengammas.resize(numrows, numcols);
        }

        if (doreferencealign)
            labels.SetValue((ElemType)(0.0f));

        size_t T = numcols / samplesInRecurrentStep; // number of time steps in minibatch
        if (samplesInRecurrentStep > 1)
        {
            assert(extrauttmap.size() == lattices.size());
            assert(T == pMBLayout->GetNumTimeSteps());
        }

        size_t mapi = 0; // parallel-sequence index for utterance [i]
        // cal gamma for each utterance
        size_t ts = 0;
        for (size_t i = 0; i < lattices.size(); i++)
        {
            const size_t numframes = lattices[i]->getnumframes();

            msra::dbn::matrixstripe predstripe(pred, ts, numframes);           // logLLs for this utterance
            msra::dbn::matrixstripe dengammasstripe(dengammas, ts, numframes); // denominator gammas

            if (samplesInRecurrentStep == 1) // no sequence parallelism
            {
                tempmatrix = loglikelihood.ColumnSlice(ts, numframes);
                // if (m_deviceid == CPUDEVICE)
                {
                    CopyFromCNTKMatrixToSSEMatrix(tempmatrix, numframes, predstripe);
                }

                if (m_deviceid != CPUDEVICE)
                    parallellattice.setloglls(tempmatrix);
            }
            else // multiple parallel sequences
            {
                // get number of frames for the utterance
                mapi = extrauttmap[i]; // parallel-sequence index; in case of >1 utterance within this parallel sequence, this is in order of concatenation

                // scan MBLayout for end of utterance
                size_t mapframenum = SIZE_MAX; // duration of utterance [i] as determined from MBLayout
                for (size_t t = validframes[mapi]; t < T; t++)
                {
                    // TODO: Adapt this to new MBLayout, m_sequences would be easier to work off.
                    if (pMBLayout->IsEnd(mapi, t))
                    {
                        mapframenum = t - validframes[mapi] + 1;
                        break;
                    }
                }

                // must match the explicit information we get from the reader
                if (numframes != mapframenum)
                    LogicError("gammacalculation: IsEnd() not working, numframes (%d) vs. mapframenum (%d)", (int) numframes, (int) mapframenum);
                assert(numframes == mapframenum);

                if (numframes > tempmatrix.GetNumCols())
                    tempmatrix.Resize(numrows, numframes);

                Microsoft::MSR::CNTK::Matrix<ElemType> loglikelihoodForCurrentParallelUtterance = loglikelihood.ColumnSlice(mapi + (validframes[mapi] * samplesInRecurrentStep), ((numframes - 1) * samplesInRecurrentStep) + 1);
                tempmatrix.CopyColumnsStrided(loglikelihoodForCurrentParallelUtterance, numframes, samplesInRecurrentStep, 1);

                // if (doreferencealign || m_deviceid == CPUDEVICE)
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
            foreach_column (t, dengammasstripe) // we do not allocate memory for numgamma now, should be the same as numgammasstripe
            {
                const size_t s = uidsstripe[t];
                numavlogp += predstripe(s, t) / amf;
            }
            numavlogp /= numframes;

            // auto_timer dengammatimer;
            double denavlogp = lattices[i]->second.forwardbackward(parallellattice,
                                                                   (const msra::math::ssematrixbase&) predstripe, (const msra::asr::simplesenonehmm&) m_hset,
                                                                   (msra::math::ssematrixbase&) dengammasstripe, (msra::math::ssematrixbase&) gammasbuffer /*empty, not used*/,
                                                                   lmf, wp, amf, boostmmifactor, seqsMBRmode, uidsstripe, boundariesstripe);
            objectValue += (ElemType)((numavlogp - denavlogp) * numframes);

            if (samplesInRecurrentStep == 1)
            {
                tempmatrix = gammafromlattice.ColumnSlice(ts, numframes);
            }

            // copy gamma to tempmatrix
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
                        labels(uid, (nframe + validframes[mapi]) * samplesInRecurrentStep + mapi) = 1.0;
                    else
                        labels(uid, ts + nframe) = 1.0;
                }
            }
            if (samplesInRecurrentStep > 1)
                validframes[mapi] += numframes; // advance the cursor within the parallel sequence
            fprintf(stderr, "dengamma value %f\n", denavlogp);
            ts += numframes;
        }
        functionValues.SetValue(objectValue);
    }


    // Calculate CTC score
    // totalScore (output): total CTC score at element (0,0)
    // prob (input): the posterior output from the network (log softmax of right)
    // maxIndexes (input): indexes of max elements in label input vectors
    // maxValues (input): values of max elements in label input vectors
    // labels (input): 1-hot vector with frame-level phone labels
    // CTCPosterior (output): CTC posterior
    // blankTokenId (input): id of the blank token. If specified as SIZE_MAX, will be replaced with (numberOfLabels - 1)
    // delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference. This using the original time information to enforce that CTC tokens only get aligned within a time margin.
    //      Setting this parameter smaller will result in shorted delay between label output during decoding, yet may hurt accuracy.
    //      delayConstraint=-1 means no constraint
    void doCTC(Microsoft::MSR::CNTK::Matrix<ElemType>& totalScore, 
        const Microsoft::MSR::CNTK::Matrix<ElemType>& prob, 
        const Microsoft::MSR::CNTK::Matrix<ElemType>& maxIndexes,
        const Microsoft::MSR::CNTK::Matrix<ElemType>& maxValues,
        Microsoft::MSR::CNTK::Matrix<ElemType>& CTCPosterior, 
        const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> pMBLayout, 
        size_t blankTokenId,
        int delayConstraint = -1)
    {
        const auto numParallelSequences = pMBLayout->GetNumParallelSequences();
        const auto numSequences = pMBLayout->GetNumSequences();
        const size_t numRows = prob.GetNumRows();
        const size_t numCols = prob.GetNumCols();
        m_deviceid = prob.GetDeviceId();
        Microsoft::MSR::CNTK::Matrix<ElemType> matrixPhoneSeqs(CPUDEVICE);
        Microsoft::MSR::CNTK::Matrix<ElemType> matrixPhoneBounds(CPUDEVICE);
        std::vector<std::vector<size_t>> allUttPhoneSeqs;
        std::vector<std::vector<size_t>> allUttPhoneBounds;
        int maxPhoneNum = 0;
        std::vector<size_t> phoneSeq;
        std::vector<size_t> phoneBound;

        if (blankTokenId == SIZE_MAX)
            blankTokenId = numRows - 1;

        size_t mbsize = numCols / numParallelSequences;

        // Prepare data structures from the reader
        // the position of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
        std::vector<size_t> uttBeginFrame;
        // the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
        std::vector<size_t> uttFrameNum;
        // the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
        std::vector<size_t> uttPhoneNum;
        // map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
        std::vector<size_t> uttToChanInd;
        uttBeginFrame.reserve(numSequences);
        uttFrameNum.reserve(numSequences);
        uttPhoneNum.reserve(numSequences);
        uttToChanInd.reserve(numSequences);
        size_t seqId = 0;
        for (const auto& seq : pMBLayout->GetAllSequences())
        {
            if (seq.seqId == GAP_SEQUENCE_ID)
                continue;

            assert(seq.seqId == seqId);
            seqId++;
            uttToChanInd.push_back(seq.s);
            size_t numFrames = seq.GetNumTimeSteps();
            uttBeginFrame.push_back(seq.tBegin);
            uttFrameNum.push_back(numFrames);

            // Get the phone list and boundaries
            phoneSeq.clear();
            phoneSeq.push_back(SIZE_MAX);
            phoneBound.clear();
            phoneBound.push_back(0);
            int prevPhoneId = -1;
            size_t startFrameInd = seq.tBegin * numParallelSequences + seq.s;
            size_t endFrameInd   = seq.tEnd   * numParallelSequences + seq.s;
            size_t frameCounter = 0;
            for (auto frameInd = startFrameInd; frameInd < endFrameInd; frameInd += numParallelSequences, frameCounter++) 
            {
                // Labels are represented as 1-hot vectors for each frame
                // If the 1-hot vectors may have either value 1 or 2 at the position of the phone corresponding to the frame:
                //      1 means the frame is within phone boundary
                //      2 means the frame is the phone boundary
                if (maxValues(0, frameInd) == 2) 
                {
                    prevPhoneId = (size_t)maxIndexes(0, frameInd);

                    phoneSeq.push_back(blankTokenId);
                    phoneBound.push_back(frameCounter);
                    phoneSeq.push_back(prevPhoneId);
                    phoneBound.push_back(frameCounter);
                }
            }
            phoneSeq.push_back(blankTokenId);
            phoneBound.push_back(numFrames);
            phoneSeq.push_back(SIZE_MAX);
            phoneBound.push_back(numFrames);

            allUttPhoneSeqs.push_back(phoneSeq);
            allUttPhoneBounds.push_back(phoneBound);

            uttPhoneNum.push_back(phoneSeq.size());

            if (phoneSeq.size() > maxPhoneNum)
                maxPhoneNum = phoneSeq.size();
        }

        matrixPhoneSeqs.Resize(maxPhoneNum, numSequences);
        matrixPhoneBounds.Resize(maxPhoneNum, numSequences);
        for (size_t i = 0; i < numSequences; i++)
        {
            for (size_t j = 0; j < allUttPhoneSeqs[i].size(); j++)
            {
                matrixPhoneSeqs(j, i) = (ElemType)allUttPhoneSeqs[i][j];
                matrixPhoneBounds(j, i) = (ElemType)allUttPhoneBounds[i][j];
            }
        }

        // Once these matrices populated, move them to the active device
        matrixPhoneSeqs.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid);
        matrixPhoneBounds.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid);

        // compute alpha, beta and CTC scores
        Microsoft::MSR::CNTK::Matrix<ElemType> alpha(m_deviceid);
        Microsoft::MSR::CNTK::Matrix<ElemType> beta(m_deviceid);
        CTCPosterior.AssignCTCScore(prob, alpha, beta, matrixPhoneSeqs, matrixPhoneBounds, totalScore, uttToChanInd, uttBeginFrame,
            uttFrameNum, uttPhoneNum, numParallelSequences, mbsize, blankTokenId, delayConstraint, /*isColWise=*/true );
        
        Microsoft::MSR::CNTK::Matrix<ElemType> rowSum(m_deviceid);
        rowSum.Resize(1, numCols);

        // Normalize the CTC scores
        CTCPosterior.VectorSum(CTCPosterior, rowSum, /*isColWise=*/true);
        CTCPosterior.RowElementDivideBy(rowSum);
    }


    // Calculate CTC score
    // totalScore (output): total CTC score at element (0,0)
    // prob (input): the posterior output from the network (log softmax of right)
    // maxIndexes (input): indexes of max elements in label input vectors
    // maxValues (input): values of max elements in label input vectors
    // labels (input): 1-hot vector with frame-level phone labels
    // CTCPosterior (output): CTC posterior
    // blankTokenId (input): id of the blank token. If specified as SIZE_MAX, will be replaced with (numberOfLabels - 1)
    // delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference. This using the original time information to enforce that CTC tokens only get aligned within a time margin.
    //      Setting this parameter smaller will result in shorted delay between label output during decoding, yet may hurt accuracy.
    //      delayConstraint=-1 means no constraint
    void twodimForwardBackward(Microsoft::MSR::CNTK::Matrix<ElemType>& totalScore,
        Microsoft::MSR::CNTK::Matrix<ElemType>& F,
        Microsoft::MSR::CNTK::Matrix<ElemType>& G,
        Microsoft::MSR::CNTK::Matrix<ElemType>& mergedinput,
        const Microsoft::MSR::CNTK::Matrix<ElemType>& maxIndexes,
        Microsoft::MSR::CNTK::Matrix<ElemType>& m_derivative,
        const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> pMBLayout,
        const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> phoneMBLayout,
        size_t blankTokenId)

    {
        const auto numParallelSequences = pMBLayout->GetNumParallelSequences();
        const auto numPhoneParallelSequences = phoneMBLayout->GetNumParallelSequences();
        const auto numSequences = pMBLayout->GetNumSequences();
        //assert(numParallelSequences==phoneMBLayout->GetNumParallelSequences());
        assert(numSequences==phoneMBLayout->GetNumSequences());
        const size_t numRows = F.GetNumRows();
        const size_t numCols = F.GetNumCols();
        const size_t numPhoneCols = G.GetNumCols();

        size_t maxFrameNum = numCols / numParallelSequences;
        size_t maxPhoneNum = numPhoneCols / numPhoneParallelSequences;

        // Prepare data structures from the reader
        // the position of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
        std::vector<size_t> uttFrameBeginIdx, uttPhoneBeginIdx;
        // the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
        std::vector<size_t> uttFrameNum;
        // the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
        std::vector<size_t> uttPhoneNum;
        // map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
        std::vector<size_t> uttFrameToChanInd, uttPhoneToChanInd;
        std::vector<size_t> phoneSeq;
        std::vector<std::vector<size_t>> allUttPhoneSeqs;

        uttFrameNum.reserve(numSequences);
        uttPhoneNum.reserve(numSequences);
        uttFrameToChanInd.reserve(numSequences);
        uttPhoneToChanInd.reserve(numSequences);
        uttFrameBeginIdx.reserve(numSequences);
        uttPhoneBeginIdx.reserve(numSequences);


        //get utt information, such as channel map id and utt begin frame, utt frame num, utt phone num for frame and phone respectively....
        size_t seqId = 0;   //frame
        size_t totalframenum = 0, totalphonenum=0;
        for (const auto& seq : pMBLayout->GetAllSequences())
        {
            if (seq.seqId == GAP_SEQUENCE_ID)
            {
                continue;
            }
            assert(seq.seqId == seqId);
            seqId++;
            uttFrameToChanInd.push_back(seq.s);            
            size_t numFrames = seq.GetNumTimeSteps();
            uttFrameBeginIdx.push_back(seq.tBegin);
            uttFrameNum.push_back(numFrames);
            totalframenum += numFrames;
        }
        seqId = 0;  //phone
        for (const auto& seq : phoneMBLayout->GetAllSequences())
        {
            if (seq.seqId == GAP_SEQUENCE_ID)
            {
                continue;
            }
            assert(seq.seqId == seqId);
            seqId++;
            uttPhoneToChanInd.push_back(seq.s);
            size_t numFrames = seq.GetNumTimeSteps();
            uttPhoneBeginIdx.push_back(seq.tBegin);
            uttPhoneNum.push_back(numFrames);
            totalphonenum += numFrames;
            size_t startFrameInd = seq.tBegin * numPhoneParallelSequences + seq.s;
            size_t endFrameInd = seq.tEnd   * numPhoneParallelSequences + seq.s;
            size_t frameCounter = 0;
            phoneSeq.clear();
            for (auto frameInd = startFrameInd; frameInd < endFrameInd; frameInd += numPhoneParallelSequences, frameCounter++)
            {
                phoneSeq.push_back((size_t)maxIndexes(0, frameInd));                
            }
            allUttPhoneSeqs.push_back(phoneSeq);
        }
        // for cpu 
        m_deviceid_gpu = maxIndexes.GetDeviceId();
        m_deviceid = m_deviceid_gpu;
        Microsoft::MSR::CNTK::Matrix<ElemType> matrixPhoneSeqs(CPUDEVICE);
        //Microsoft::MSR::CNTK::Matrix<ElemType> matrixPhoneBounds(CPUDEVICE);
        // copy phone seq to matrix
        matrixPhoneSeqs.Resize(maxPhoneNum, numSequences);
        //matrixPhoneBounds.Resize(maxPhoneNum, numSequences);
        for (size_t i = 0; i < numSequences; i++)
        {
            for (size_t j = 0; j < allUttPhoneSeqs[i].size(); j++)
            {
                matrixPhoneSeqs(j, i) = (ElemType)allUttPhoneSeqs[i][j];
          //      matrixPhoneBounds(j, i) = (ElemType)allUttPhoneBounds[i][j];
            }
        }

        // Once these matrices populated, move them to the active device
        matrixPhoneSeqs.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid);
        //matrixPhoneBounds.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid);

        //calculate the memory need for f*g
        std::vector<size_t> uttBeginForOutputditribution;
        uttBeginForOutputditribution.reserve(numSequences);
        
        size_t totalcol = 0;
        for (size_t s = 0; s < numSequences; s++)
        {
            uttBeginForOutputditribution.push_back(totalcol);
            totalcol += uttFrameNum[s] * uttPhoneNum[s];            
        }

        //compute f+g
        Microsoft::MSR::CNTK::Matrix<ElemType> matrixOutputDistribution(m_deviceid_gpu);
        /*G.TransferFromDeviceToDevice(m_deviceid_gpu, CPUDEVICE);
        for (size_t uttId = 0; uttId < numSequences; uttId++)
        {
            for (size_t u = 0; u < uttPhoneNum[uttId]; u++)
            {
                size_t phonePosInMB = (u + uttPhoneBeginIdx[uttId])*numPhoneParallelSequences + uttPhoneToChanInd[uttId];
                size_t phoneId = allUttPhoneSeqs[uttId][u];
                for (size_t k = 0; k < G.GetNumRows(); k++)
                {
                    if (k == phoneId)
                        G.SetValue(k, phonePosInMB, 0.0);
                    else
                        G.SetValue(k, phonePosInMB, (float)LZERO);
                }
            }
            
        }
        G.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);*/
        //G.SetValue(0.0);
        //F.Print("H");
        //G.Print("G");
        
        //matrixOutputDistribution.Resize(numRows, totalcol);
        //matrixOutputDistribution.AssignUserOp1(F, G, uttFrameToChanInd, uttPhoneToChanInd, uttFrameBeginIdx, uttPhoneBeginIdx, uttBeginForOutputditribution, uttFrameNum, uttPhoneNum, 
        //   totalcol, numParallelSequences, numPhoneParallelSequences);
        //matrixOutputDistribution.Print("h");
        //log softmax of f+g
        mergedinput.InplaceLogSoftmax(true);
        //Microsoft::MSR::CNTK::Matrix<ElemType> logsoftmax(m_deviceid_gpu);
        //logsoftmax.AssignLogSoftmaxOf(mergedinput,true);
        //matrixOutputDistribution.Print("prob");
        // forward backward to compute alpha, beta derivaitves
        Microsoft::MSR::CNTK::Matrix<ElemType> alpha(m_deviceid_gpu);
        Microsoft::MSR::CNTK::Matrix<ElemType> beta(m_deviceid_gpu);
        m_derivative.TransferToDeviceIfNotThere(m_deviceid_gpu);
        m_derivative.AssignRNNTScore(mergedinput, alpha, beta, matrixPhoneSeqs, matrixPhoneSeqs, uttFrameToChanInd, uttFrameBeginIdx, uttBeginForOutputditribution, uttPhoneToChanInd, uttPhoneBeginIdx,
            uttFrameNum, uttPhoneNum, numParallelSequences, numPhoneParallelSequences, maxPhoneNum, maxFrameNum, totalScore, blankTokenId, -1,true);
        ElemType finalscore = 0;
        //m_derivative.Print("RNNT");
        finalscore =  totalScore.Get00Element();
        //fprintf(stderr, "finalscore:%f\n", finalscore);
        if (finalscore > 50 || finalscore < 0)
        {
            for (size_t i = 0; i < uttFrameNum.size(); i++)
            {
                fprintf(stderr, "framenum:%d\n", (int)(uttFrameNum[i]));                
            }
            matrixPhoneSeqs.Print("phone seq");
            //matrixPhoneBounds.Print("phone bound");
        }
        /*alpha.Print("alpha");
        beta.Print("beta");
        prob.Print("prob");*/
       
        //m_derivative.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);
        //matrixOutputDistribution.ReleaseMemory();

        //compute derivatives for F and G
        /*if (m_derivativeForF.GetDeviceId() != CPUDEVICE)
            printf("m_derivativeForF before is in GPU");
        if (m_derivativeForG.GetDeviceId() != CPUDEVICE)
            printf("m_derivativeForG before is in GPU");*/
        
        /*m_derivativeForF.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);
        m_derivativeForG.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);
        m_derivativeForF.SetValue(0.0);
        m_derivativeForG.SetValue(0.0);
        m_derivativeForF.AssignUserOp2(RNNTPosterior, uttFrameToChanInd, uttPhoneToChanInd, uttFrameBeginIdx, uttPhoneBeginIdx, uttBeginForOutputditribution, uttFrameNum, uttPhoneNum,
            numParallelSequences, numPhoneParallelSequences, maxFrameNum, maxPhoneNum, 0);
        m_derivativeForG.AssignUserOp2(RNNTPosterior, uttFrameToChanInd, uttPhoneToChanInd, uttFrameBeginIdx, uttPhoneBeginIdx, uttBeginForOutputditribution, uttFrameNum, uttPhoneNum,
            numParallelSequences, numPhoneParallelSequences, maxFrameNum, maxPhoneNum, 1);*/

        //do derivative norm
        /*Microsoft::MSR::CNTK::Matrix<ElemType> tempMatrix1(m_deviceid_gpu), tempMatrix2(m_deviceid_gpu);
        Microsoft::MSR::CNTK::Matrix<ElemType>::VectorSum(m_derivativeForF, tempMatrix1, false);
        tempMatrix2.AssignVectorNorm2Of(tempMatrix1,true);
        //tempMatrix1.Print("sum of F");
        //tempMatrix2.Print("norm of F");
        //fprintf(stderr, "framenum %d phonenum %d\n", (int) totalframenum, (int) totalphonenum);
        ElemType norm_coef = (ElemType) 10.0/ tempMatrix2.Get00Element();
        Microsoft::MSR::CNTK::Matrix<ElemType>::Scale(norm_coef, m_derivativeForF);
        Microsoft::MSR::CNTK::Matrix<ElemType>::Scale(norm_coef * (ElemType) totalphonenum / (ElemType)totalframenum, m_derivativeForG);*/

        /*//tempMatrix2.Print("norm of F");
        Microsoft::MSR::CNTK::Matrix<ElemType>::VectorSum(m_derivativeForG, tempMatrix1, false);
        //tempMatrix1.Print("sum of G");
        tempMatrix2.AssignVectorNorm2Of(tempMatrix1, true);
        tempMatrix2.Print("norm of G");
        if (m_derivativeForF.GetDeviceId() != CPUDEVICE)
            printf("m_derivativeForF after is in GPU");
        if (m_derivativeForG.GetDeviceId() != CPUDEVICE)
            printf("m_derivativeForG after is in GPU");
        m_derivativeForF.Print("derivative for F");
        m_derivativeForG.Print("derivative for G");

        m_derivativeForF.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);
        m_derivativeForG.TransferFromDeviceToDevice(CPUDEVICE, m_deviceid_gpu);
        printf("finish gamma");*/

    }

private:
    // Helper methods for copying between ssematrix objects and CNTK matrices
    void CopyFromCNTKMatrixToSSEMatrix(const Microsoft::MSR::CNTK::Matrix<ElemType>& src, size_t numCols, msra::math::ssematrixbase& dest)
    {
        if (!std::is_same<ElemType, float>::value)
        {
            LogicError("Cannot copy between a SSE matrix and a non-float type CNTK Matrix object!");
        }

        size_t numRows = src.GetNumRows();
        const Microsoft::MSR::CNTK::Matrix<ElemType> srcSlice = src.ColumnSlice(0, numCols);
        if ((m_intermediateCUDACopyBuffer == nullptr) || (m_intermediateCUDACopyBufferSize < srcSlice.GetNumElements()))
        {
            m_intermediateCUDACopyBuffer = AllocateIntermediateBuffer(srcSlice.GetDeviceId(), srcSlice.GetNumElements());
            m_intermediateCUDACopyBufferSize = srcSlice.GetNumElements();
        }

        ElemType* pBuf = m_intermediateCUDACopyBuffer.get();
        srcSlice.CopyToArray(pBuf, m_intermediateCUDACopyBufferSize);
        if (pBuf != m_intermediateCUDACopyBuffer.get())
        {
            LogicError("Unexpected re-allocation of destination CPU buffer in Matrix::CopyToArray!");
        }

        if ((dest.getcolstride() == dest.rows()) && (numRows == dest.rows()))
        {
            memcpy(&dest(0, 0), (float*) pBuf, sizeof(ElemType) * numRows * numCols);
        }
        else
        {
            // We need to copy columnwise
            for (size_t i = 0; i < numCols; ++i)
            {
                memcpy(&dest(0, i), (float*) (pBuf + (i * numRows)), sizeof(ElemType) * numRows);
            }
        }
    }

    void CopyFromSSEMatrixToCNTKMatrix(const msra::math::ssematrixbase& src, size_t numRows, size_t numCols, Microsoft::MSR::CNTK::Matrix<ElemType>& dest, int deviceId)
    {
        if (!std::is_same<ElemType, float>::value)
        {
            LogicError("Cannot copy between a SSE matrix and a non-float type CNTK Matrix object!");
        }

        size_t numElements = numRows * numCols;
        if ((m_intermediateCUDACopyBuffer == nullptr) || (m_intermediateCUDACopyBufferSize < numElements))
        {
            m_intermediateCUDACopyBuffer = AllocateIntermediateBuffer(deviceId, numElements);
            m_intermediateCUDACopyBufferSize = numElements;
        }

        if ((src.getcolstride() == src.rows()) && (numRows == src.rows()))
        {
            memcpy((float*) m_intermediateCUDACopyBuffer.get(), &src(0, 0), sizeof(float) * numRows * numCols);
        }
        else
        {
            // We need to copy columnwise
            for (size_t i = 0; i < numCols; ++i)
            {
                memcpy((float*) (m_intermediateCUDACopyBuffer.get() + (i * numRows)), &src(0, i), sizeof(float) * numRows);
            }
        }

        dest.SetValue(numRows, numCols, deviceId, m_intermediateCUDACopyBuffer.get(), 0);
    }

    // TODO: This function is duplicate of the one in HTLMLFReader.
    // This should be moved to a common utils library and removed from here as well as HTLMLFReader
    std::unique_ptr<Microsoft::MSR::CNTK::CUDAPageLockedMemAllocator>& GetCUDAAllocator(int deviceID)
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
            m_cudaAllocator.reset(new Microsoft::MSR::CNTK::CUDAPageLockedMemAllocator(deviceID));
        }

        return m_cudaAllocator;
    }

    // TODO: This function is duplicate of the one in HTLMLFReader.
    // This should be moved to a common utils library and removed from here as well as HTLMLFReader
    std::shared_ptr<ElemType> AllocateIntermediateBuffer(int deviceID, size_t numElements)
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

protected:
    msra::asr::simplesenonehmm m_hset;
    msra::lattices::lattice::parallelstate parallellattice;
    msra::lattices::mbrclassdefinition mbrclassdef = msra::lattices::senone; // defines the unit for minimum bayesian risk
    bool initialmark;
    msra::dbn::matrix dengammas;
    msra::dbn::matrix pred;
    int m_deviceid; // -1: cpu
    int m_deviceid_gpu;
    size_t m_maxframenum;
    float lmf; // Note that 9 was best for Fisher  --these should best be configurable
    float wp;
    float amf;
    msra::dbn::matrix gammasbuffer;
    std::vector<size_t> boundary;
    float boostmmifactor;
    bool seqsMBRmode;

private:
    std::unique_ptr<Microsoft::MSR::CNTK::CUDAPageLockedMemAllocator> m_cudaAllocator;
    std::shared_ptr<ElemType> m_intermediateCUDACopyBuffer;
    size_t m_intermediateCUDACopyBufferSize;
};

}}
