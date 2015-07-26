#include "basetypes.h"
#include "htkfeatio_utils.h"
#include "KaldiSequenceTrainingIO.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Constructor.
    template<class ElemType>
    KaldiSequenceTrainingIO<ElemType>::KaldiSequenceTrainingIO(
        const wstring& denlatRspecifier, const wstring& aliRspecifier,
        const wstring& transModelFilename, const wstring& silencePhoneStr,
        const wstring& trainCriterion,
        ElemType oldAcousticScale, ElemType acousticScale,
        ElemType lmScale, bool oneSilenceClass, size_t numberOfuttsPerMinibatch)
    {
        using namespace msra::asr;
        assert(denlatRspecifier != L"");
        assert(aliRspecifier != L"");
        m_denlatReader = new kaldi::RandomAccessCompactLatticeReader(
            trimmed(fileToStr(toStr(denlatRspecifier))));
        m_aliReader = new kaldi::RandomAccessInt32VectorReader(
            trimmed(fileToStr(toStr(aliRspecifier))));
        ReadKaldiObject(toStr(transModelFilename), &m_transModel);
        m_oldAcousticScale = oldAcousticScale;
        m_acousticScale = acousticScale;
        m_lmScale = lmScale;
        m_trainCriterion = trainCriterion;
        m_oneSilenceClass = oneSilenceClass;
        m_numUttsPerMinibatch = numberOfuttsPerMinibatch;
        m_needLikelihood = true;
        m_currentObj = 0;
        m_minibatchIndex = 1;
        m_lastCompleteMinibatch.assign(m_numUttsPerMinibatch, 0);
        if (!kaldi::SplitStringToIntegers(toStr(silencePhoneStr),
                                          ":", false, &m_silencePhones))
        {
            LogicError("Invalid silence phone sequence.\n");
        }
        if (m_trainCriterion != L"mpfe" && m_trainCriterion != L"smbr")
        {
            LogicError("Supported sequence training criterion: mpfe, smbr.\n");
        }
    }

    // Destructor.
    template<class ElemType>
    KaldiSequenceTrainingIO<ElemType>::~KaldiSequenceTrainingIO()
    {
        if (m_denlatReader != NULL)
        {
            delete m_denlatReader;
            m_denlatReader = NULL;
        }
        if (m_aliReader != NULL)
        {
            delete m_aliReader;
            m_aliReader = NULL;
        }
    }

    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::ComputeDerivative(
        const wstring& uttID)
    {
        assert(m_uttPool.find(uttID) != m_uttPool.end());
        assert(m_uttPool[uttID].hasDerivative == false);
        Matrix<ElemType>& logLikelihood = m_uttPool[uttID].logLikelihood;

        std::string uttIDStr = msra::asr::toStr(uttID);

        // Sanity check.
        if (m_transModel.NumPdfs() != logLikelihood.GetNumRows())
        {
            RuntimeError("Number of labels in logLikelihood does not match that"
                         " in the Kaldi model for utterance %S: %d v.s. %d\n",
                         uttID.c_str(), logLikelihood.GetNumRows(),
                         m_transModel.NumPdfs());
        }

        // Reads alignment.
        if (!m_aliReader->HasKey(uttIDStr))
        {
            RuntimeError("Alignment not found for utterance %s\n",
                         uttIDStr.c_str());
        }
        const std::vector<int32> ali = m_aliReader->Value(uttIDStr);
        if (ali.size() != logLikelihood.GetNumCols())
        {
            RuntimeError("Number of frames in logLikelihood does not match that"
                         " in the alignment for utterance %S: %d v.s. %d\n",
                         uttID.c_str(), logLikelihood.GetNumCols(), ali.size());
        }

        // Reads denominator lattice.
        if (!m_denlatReader->HasKey(uttIDStr))
        {
            RuntimeError("Denominator lattice not found for utterance %S\n",
                         uttID.c_str());
        }
        kaldi::CompactLattice clat = m_denlatReader->Value(uttIDStr);
        fst::CreateSuperFinal(&clat);  /* One final state with weight One() */
        kaldi::Lattice lat;
        fst::ConvertLattice(clat, &lat);

        // Does a first path of acoustic scaling. Typically this sets the old
        // acoustic scale to 0.
        if (m_oldAcousticScale != 1.0)
        {
            fst::ScaleLattice(fst::AcousticLatticeScale(m_oldAcousticScale),
                              &lat);
        }

        // Topsort lattice.
        kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
        if (!(props & fst::kTopSorted))
        {
            if (fst::TopSort(&lat) == false)
            {
                RuntimeError("Cycles detected in lattice: %S\n", uttID.c_str());
            }
        }

        // Gets time information for the lattice.
        std::vector<kaldi::int32> stateTimes;
        kaldi::int32 maxTime = kaldi::LatticeStateTimes(lat, &stateTimes);
        if (maxTime != logLikelihood.GetNumCols())
        {
            RuntimeError("Number of frames in the logLikelihood does not match"
                         " that in the denominator lattice for utterance %S\n",
                         uttID.c_str(), logLikelihood.GetNumRows(), maxTime);
        }

        // Does lattice acoustic rescoring with the new posteriors from the
        // neural network.
        LatticeAcousticRescore(stateTimes, logLikelihood, &lat);

        // Second pass acoustic and language model scale.
        if (m_acousticScale != 1.0 || m_lmScale != 1.0)
        {
            fst::ScaleLattice(fst::LatticeScale(m_lmScale, m_acousticScale),
                              &lat);
        }

        // Forward-backward on the lattice.
        kaldi::Posterior post;
        ElemType thisObj = 0;
        if (m_trainCriterion == L"smbr")
        {
            thisObj = kaldi::LatticeForwardBackwardMpeVariants(
                m_transModel, m_silencePhones, lat,
                ali, "smbr", m_oneSilenceClass, &post);
        }
        else if (m_trainCriterion == L"mpfe")
        {
            thisObj = kaldi::LatticeForwardBackwardMpeVariants(
                m_transModel, m_silencePhones, lat,
                ali, "mpfe", m_oneSilenceClass, &post);
        }

        kaldi::ConvertPosteriorToPdfs(m_transModel,
                                      post, &(m_uttPool[uttID].posterior));

        // Uses "expected error rate" instead of "expected accuracy".
        m_uttPool[uttID].objective = logLikelihood.GetNumCols() - thisObj;

        assert(m_uttPool[uttID].posterior.size() == logLikelihood.GetNumCols());

        return true;
    }

    template<class ElemType>
    void KaldiSequenceTrainingIO<ElemType>::LatticeAcousticRescore(
        const std::vector<kaldi::int32>& stateTimes,
        const Matrix<ElemType>& logLikelihood, kaldi::Lattice* lat) const
    {
        std::vector<std::vector<kaldi::int32>> timeStateMap(
            logLikelihood.GetNumCols());
        size_t num_states = lat->NumStates();
        for (size_t s = 0; s < num_states; s++)
        {
            assert(stateTimes[s] >= 0
                   && stateTimes[s] <= logLikelihood.GetNumCols());
            if (stateTimes[s] < logLikelihood.GetNumCols())
            {
                timeStateMap[stateTimes[s]].push_back(s);
            }
        }

        for (size_t t = 0; t < logLikelihood.GetNumCols(); ++t)
        {
            for (size_t i = 0; i < timeStateMap[t].size(); ++i)
            {
                kaldi::int32 state = timeStateMap[t][i];
                for (fst::MutableArcIterator<kaldi::Lattice> aiter(lat, state);
                     !aiter.Done(); aiter.Next())
                {
                    kaldi::LatticeArc arc = aiter.Value();
                    kaldi::int32 trans_id = arc.ilabel;
                    if (trans_id != 0)
                    {
                        kaldi::int32 pdf_id =
                            m_transModel.TransitionIdToPdf(trans_id);
                        arc.weight.SetValue2(-logLikelihood(pdf_id, t)
                                             + arc.weight.Value2());
                        aiter.SetValue(arc);
                    }
                }
                // Checks final state.
                kaldi::LatticeWeight final = lat->Final(state);
                if (final != kaldi::LatticeWeight::Zero())
                {
                    final.SetValue2(0.0);
                    lat->SetFinal(state, final);
                }
            }
        }
    }

    template<class ElemType>
    void KaldiSequenceTrainingIO<ElemType>::ProcessUttInfo(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag,
        std::vector<std::vector<std::pair<wstring, std::pair<size_t, size_t>>>>* uttInfoInMinibatch) const
    {
        assert(uttInfoInMinibatch != NULL);
        assert(uttInfo.size() == m_numUttsPerMinibatch);
        assert(sentenceBegin.GetNumRows() == m_numUttsPerMinibatch);
        assert(minibatchPackingFlag.size() == sentenceBegin.GetNumCols());
        uttInfoInMinibatch->clear();
        uttInfoInMinibatch->resize(uttInfo.size());
        for (size_t i = 0; i < uttInfo.size(); ++i)
        {
            size_t startFrameIndexInMinibatch = 0;
            size_t numFrames = 0;
            for (size_t j = 0; j < sentenceBegin.GetNumCols(); ++j)
            {
                if (((size_t)sentenceBegin(i, j) & NO_LABEL) == NO_LABEL)
                {
                    continue;
                }
                if (((size_t)sentenceBegin(i, j) & NO_FEATURE) == NO_FEATURE)
                {
                    continue;
                }
                numFrames += 1;
                if ((((size_t)sentenceBegin(i, j) & SEQUENCE_END) == SEQUENCE_END)
                         || j == sentenceBegin.GetNumCols() - 1)
                {
                    size_t uttIndex = (*uttInfoInMinibatch)[i].size();
                    wstring uttID = uttInfo[i][uttIndex].first;
                    (*uttInfoInMinibatch)[i].push_back(
                        make_pair(uttID, make_pair(startFrameIndexInMinibatch, numFrames)));
                    startFrameIndexInMinibatch = j + 1;
                    numFrames = 0;
                }
            }
            assert(uttInfo[i].size() == (*uttInfoInMinibatch)[i].size());
        }
    }

    // Suppose we have a, b, c 3 streams, the <logLikelihoodIn> is the in the
    // following format:
    // 1: a11 b11 c11 a12 b12 c12...
    // 2: a21 b21 c21 a22 b22 c22...
    // 3: a31 b31 c31 a32 b32 c32...
    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::SetLikelihood(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& logLikelihoodIn,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag)
    {
        assert(m_needLikelihood == true);
        std::vector<std::vector<
            std::pair<wstring, std::pair<size_t, size_t>>>> uttInfoInMinibatch;
        ProcessUttInfo(uttInfo, sentenceBegin,
                       minibatchPackingFlag, &uttInfoInMinibatch);

        // Checks if we need to move data to CPU.
        Matrix<ElemType> logLikelihood(logLikelihoodIn);
        if (logLikelihood.GetDeviceId() >= 0)
        {
            logLikelihood.TransferFromDeviceToDevice(
                logLikelihood.GetDeviceId(), CPUDEVICE, true, false, false);
        }

        bool minibatchComplete = true;
        size_t currentMBSize = minibatchPackingFlag.size();
        for (size_t i = 0; i < uttInfo.size(); ++i)
        {
            assert(uttInfo[i].size() == uttInfoInMinibatch[i].size());
            for (size_t j = 0; j < uttInfo[i].size(); ++j)
            {
                wstring uttID = uttInfo[i][j].first;
                if (m_uttPool.find(uttID) == m_uttPool.end())
                {
                    UtteranceDerivativeUnit tmpUttUnit;
                    tmpUttUnit.hasDerivative = false;
                    tmpUttUnit.uttLength = uttInfo[i][j].second;
                    tmpUttUnit.progress = 0;
                    tmpUttUnit.streamID = i;
                    tmpUttUnit.logLikelihood.Resize(m_transModel.NumPdfs(),
                                                    tmpUttUnit.uttLength);
                    m_uttPool[uttID] = tmpUttUnit;
                }

                // Sets the likelihood and computes derivatives.
                assert(m_uttPool.find(uttID) != m_uttPool.end());
                if (m_uttPool[uttID].hasDerivative == false)
                {
                    assert(uttID == uttInfoInMinibatch[i][j].first);
                    size_t startFrame = uttInfoInMinibatch[i][j].second.first;
                    size_t numFrames = uttInfoInMinibatch[i][j].second.second;
                    assert(m_uttPool[uttID].progress + numFrames
                           <= m_uttPool[uttID].uttLength);

                    // Sets the likelihood.
                    for (size_t k = 0; k < numFrames; ++k)
                    {
                        m_uttPool[uttID].logLikelihood.SetColumn(
                            logLikelihood.ColumnSlice(
                            (startFrame + k) * m_numUttsPerMinibatch + i, 1),
                            m_uttPool[uttID].progress + k);
                    }

                    m_uttPool[uttID].progress += numFrames;
                    if (m_uttPool[uttID].progress == m_uttPool[uttID].uttLength)
                    {
                        ComputeDerivative(uttID);
                        m_uttPool[uttID].hasDerivative = true;
                        m_uttPool[uttID].progress = 0;
                        if (startFrame + numFrames == currentMBSize)
                        {
                            m_lastCompleteMinibatch[m_uttPool[uttID].streamID]
                                = m_minibatchIndex;
                        }
                        else
                        {
                            m_lastCompleteMinibatch[m_uttPool[uttID].streamID]
                                = m_minibatchIndex - 1;
                        }
                    }
                }
            }
        }

        // Checks if we are ready to provide derivatives.
        m_minCompleteMinibatchIndex = *std::min_element(
            m_lastCompleteMinibatch.begin(), m_lastCompleteMinibatch.end());
        m_needLikelihood = (m_minCompleteMinibatchIndex >= 1) ? false : true;
        m_minibatchIndex += 1;
    }

    // Suppose we have a, b, c 3 streams, the <derivativesOut> should be in the
    // following format:
    // 1: a11 b11 c11 a12 b12 c12...
    // 2: a21 b21 c21 a22 b22 c22...
    // 3: a31 b31 c31 a32 b32 c32...
    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::GetDerivative(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& sentenceBegin,
        const std::vector<MinibatchPackingFlag>& minibatchPackingFlag,
        Matrix<ElemType>* derivativesOut)
    {
        assert(derivativesOut != NULL);
        std::vector<std::vector<
            std::pair<wstring, std::pair<size_t, size_t>>>> uttInfoInMinibatch;
        ProcessUttInfo(uttInfo, sentenceBegin,
                       minibatchPackingFlag, &uttInfoInMinibatch);

        Matrix<ElemType> derivatives(CPUDEVICE);
        derivatives.Resize(m_transModel.NumPdfs(),
            sentenceBegin.GetNumCols() * sentenceBegin.GetNumRows());
        derivatives.SetValue(0);

        m_currentObj = 0;
        for (size_t i = 0; i < uttInfo.size(); ++i)
        {
            assert(uttInfo[i].size() == uttInfoInMinibatch[i].size());
            for (size_t j = 0; j < uttInfo[i].size(); ++j)
            {
                wstring uttID = uttInfo[i][j].first;

                // Checks if we have derivatives.
                if (m_uttPool.find(uttID) == m_uttPool.end()
                    || (m_uttPool.find(uttID) != m_uttPool.end()
                        && m_uttPool[uttID].hasDerivative == false))
                {
                    RuntimeError("Derivatives are not ready for utterance:"
                                 " %S\n", uttID.c_str());
                }

                // Assign the derivatives.
                assert(uttID == uttInfoInMinibatch[i][j].first);
                size_t startFrame = uttInfoInMinibatch[i][j].second.first;
                size_t startFrameInUtt = m_uttPool[uttID].progress;
                size_t numFrames = uttInfoInMinibatch[i][j].second.second;
                for (size_t k = 0; k < numFrames; ++k)
                {
                    size_t posStart = startFrameInUtt + k;
                    for (size_t l = 0;
                        l < m_uttPool[uttID].posterior[posStart].size(); ++l)
                    {
                        size_t pdf_id =
                            m_uttPool[uttID].posterior[posStart][l].first;
                        assert(pdf_id < m_transModel.NumPdfs());
                        derivatives(pdf_id,
                            (startFrame + k) * m_numUttsPerMinibatch + i) -=
                            m_uttPool[uttID].posterior[posStart][l].second;
                    }
                }
                m_currentObj += m_uttPool[uttID].objective
                    * numFrames / m_uttPool[uttID].uttLength;
                m_uttPool[uttID].progress += numFrames;
                assert(m_uttPool[uttID].progress <= m_uttPool[uttID].uttLength);
                if (m_uttPool[uttID].progress == m_uttPool[uttID].uttLength)
                {
                    m_uttPool.erase(uttID);
                }
            }
        }

        // Checks if we need to move data to GPU.
        if (derivativesOut->GetDeviceId() >= 0)
        {
            derivatives.TransferFromDeviceToDevice(
                CPUDEVICE, derivativesOut->GetDeviceId(), true, false, false);
        }
        derivativesOut->SetValue(derivatives);

        // Keeps the utterance information so we can check next time when we
        // gives the objectives.
        m_currentUttInfo = uttInfo;

        // Checks if we need to read more loglikelihoods.
        m_needLikelihood = false;
        m_minCompleteMinibatchIndex -= 1;
        if (m_minCompleteMinibatchIndex <= 0)
        {
            m_needLikelihood = true;
            m_minibatchIndex = 1;
            m_lastCompleteMinibatch.assign(m_numUttsPerMinibatch, 0);

            // Un-do the logLikelihood for partial utterances.
            for (auto iter = m_uttPool.begin(); iter != m_uttPool.end(); ++iter)
            {
                if (iter->second.hasDerivative == false)
                {
                    iter->second.progress = 0;
                }
            }
        }
        return true;
    }

    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::GetObjective(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        Matrix<ElemType>* objectivesIn)
    {
        assert(objectivesIn != NULL);

        // Checks utterance information.
        bool match = true;
        if (uttInfo.size() == m_currentUttInfo.size())
        {
            for (size_t i = 0; i < uttInfo.size(); ++i)
            {
                if (uttInfo[i].size() != m_currentUttInfo[i].size())
                {
                    match = false;
                    break;
                }
                for (size_t j = 0; j < uttInfo[i].size(); ++j)
                {
                    if (uttInfo[i][j].first != m_currentUttInfo[i][j].first ||
                        uttInfo[i][j].second != m_currentUttInfo[i][j].second)
                    {
                        match = false;
                        break;
                    }
                }
            }
        }
        else
        {
            match = false;
        }
        if (!match)
        {
            RuntimeError("Current objective does not correspond to the"
                         " minibatch utterance information, perhaps you did not"
                         " run GetObjective() right after GetDerivatives()?");
        }

        // Sets the objectives...
        objectivesIn->Resize(1, 1);
        objectivesIn->SetValue(m_currentObj);

        return true;
    }

    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::HasLatticeAndAlignment(
        const wstring& uttID) const
    {
        if(m_aliReader == false || m_denlatReader == false)
        {
            fprintf(stderr, "WARNING: lattice or alignemnt reader has not been"
                            " set up yet.\n");
            return false;
        }

        std::string uttIDStr = msra::asr::toStr(uttID);
        if(!m_aliReader->HasKey(uttIDStr) || !m_denlatReader->HasKey(uttIDStr))
        {
            return false;
        }
        return true;
    }

    template class KaldiSequenceTrainingIO<float>;
    template class KaldiSequenceTrainingIO<double>;
}}}
