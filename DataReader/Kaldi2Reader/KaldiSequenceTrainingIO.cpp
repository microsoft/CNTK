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
        ElemType lmScale, bool oneSilenceClass)
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
        m_objective = 0;
        m_posteriors.clear();
        if (!kaldi::SplitStringToIntegers(toStr(silencePhoneStr),
                                          ":", false, &m_silencePhones))
        {
            LogicError("Invalid silence phone sequence.\n");
        }
        if (m_trainCriterion != L"mpfe" && m_trainCriterion != L"smbr")
        {
            LogicError("Supported sequence training criterion are: mpfe, smbr.\n");
        }
        m_derivRead = false;
        m_objRead = false;
        m_currentUttHasDeriv = false;
        m_currentUttID = L"";
        m_currentUttLength = 0;
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
    bool KaldiSequenceTrainingIO<ElemType>::HasDerivatives(const wstring& uttID)
    {
        if (uttID == m_currentUttID && m_currentUttHasDeriv)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template<class ElemType>
    bool KaldiSequenceTrainingIO<ElemType>::ComputeDerivatives(
        const wstring& uttID, const Matrix<ElemType>& logLikelihood)
    {
        std::string uttIDStr = msra::asr::toStr(uttID);

        // Sanity check.
        if (m_transModel.NumPdfs() != logLikelihood.GetNumRows())
        {
            RuntimeError("Number of labels in logLikelihood does not match that in the Kaldi model for utterance %S: %d v.s. %d\n", uttID.c_str(), logLikelihood.GetNumRows(), m_transModel.NumPdfs());
        }

        // Reads alignment.
        if (!m_aliReader->HasKey(uttIDStr))
        {
            RuntimeError("Alignment not found for utterance %s\n", uttIDStr.c_str());
        }
        const std::vector<int32> ali = m_aliReader->Value(uttIDStr);
        if (ali.size() != logLikelihood.GetNumCols())
        {
            RuntimeError("Number of frames in logLikelihood does not match that in the alignment for utterance %S: %d v.s. %d\n", uttID.c_str(), logLikelihood.GetNumCols(), ali.size());
        }

        // Reads denominator lattice.
        if (!m_denlatReader->HasKey(uttIDStr))
        {
            RuntimeError("Denominator lattice not found for utterance %S\n", uttID.c_str());
        }
        kaldi::CompactLattice clat = m_denlatReader->Value(uttIDStr);
        fst::CreateSuperFinal(&clat);  /* One final state with weight One() */
        kaldi::Lattice lat;
        fst::ConvertLattice(clat, &lat);

        // Does a first path of acoustic scaling. Typically this sets the old
        // acoustic scale to 0.
        if (m_oldAcousticScale != 1.0)
        {
            fst::ScaleLattice(fst::AcousticLatticeScale(m_oldAcousticScale), &lat);
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
            RuntimeError("Number of frames in the logLikelihood does not match that in the denominator lattice for utterance %S\n", uttID.c_str(), logLikelihood.GetNumRows(), maxTime);
        }

        // Does lattice acoustic rescoring with the new posteriors from the
        // neural network.
        LatticeAcousticRescore(stateTimes, logLikelihood, &lat);

        // Second pass acoustic and language model scale.
        if (m_acousticScale != 1.0 || m_lmScale != 1.0)
        {
            fst::ScaleLattice(fst::LatticeScale(m_lmScale, m_acousticScale), &lat);
        }

        // Forward-backward on the lattice.
        kaldi::Posterior post;
        ElemType thisObj = 0;
        if (m_trainCriterion == L"smbr")
        {
            thisObj = kaldi::LatticeForwardBackwardMpeVariants(
                m_transModel, m_silencePhones, lat, ali, "smbr", m_oneSilenceClass, &post);
        }
        else if (m_trainCriterion == L"mpfe")
        {
            thisObj = kaldi::LatticeForwardBackwardMpeVariants(
                m_transModel, m_silencePhones, lat, ali, "mpfe", m_oneSilenceClass, &post);
        }

        kaldi::ConvertPosteriorToPdfs(m_transModel, post, &m_posteriors);

        // Uses "expected error rate" instead of "expected accuracy".
        m_objective = logLikelihood.GetNumCols() - thisObj;

        assert(m_posteriors.size() == logLikelihood.GetNumCols());

        m_derivRead = false;
        m_objRead = false;
        m_currentUttHasDeriv = true;
        m_currentUttID = uttID;
        m_currentUttLength = logLikelihood.GetNumCols();
        return true;
    }

    template<class ElemType>
    void KaldiSequenceTrainingIO<ElemType>::LatticeAcousticRescore(
        const std::vector<kaldi::int32>& stateTimes,
        const Matrix<ElemType>& logLikelihood, kaldi::Lattice* lat)
    {
        // TODO(Guoguo): If we use GPUs, we may have to copy the <logLikelihood>
        // to CPUs first, as the lattice computation happens on CPUs. Otherwise
        // each call to get a single element in the matrix will be slow?
        std::vector<std::vector<kaldi::int32>> timeStateMap(logLikelihood.GetNumCols());
        size_t num_states = lat->NumStates();
        for (size_t s = 0; s < num_states; s++)
        {
            assert(stateTimes[s] >= 0 && stateTimes[s] <= logLikelihood.GetNumCols());
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
                for (fst::MutableArcIterator<kaldi::Lattice> aiter(lat, state); !aiter.Done(); aiter.Next())
                {
                    kaldi::LatticeArc arc = aiter.Value();
                    kaldi::int32 trans_id = arc.ilabel;
                    if (trans_id != 0)
                    {
                        kaldi::int32 pdf_id = m_transModel.TransitionIdToPdf(trans_id);
                        arc.weight.SetValue2(-logLikelihood(pdf_id, t) + arc.weight.Value2());
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
    void KaldiSequenceTrainingIO<ElemType>::GetDerivatives(size_t startFrame,
                                                           size_t endFrame,
                                                           const std::wstring& uttID,
                                                           Matrix<ElemType>& derivatives)
    {
        // Does some sanity check first.
        if (uttID != m_currentUttID)
        {
            RuntimeError("Requested utterance does not matched the utterance that we have computed derivatives for: %S v.s. %S\n", uttID.c_str(), m_currentUttID.c_str());
        }
        if (!m_currentUttHasDeriv)
        {
            RuntimeError("Derivatives have not been computed, you have to call KaldiSequenceTrainingIO::ComputeDerivative() before using it.\n");
        }
        assert(startFrame >= 0);
        assert(endFrame <= m_currentUttLength);

        // TODO(Guoguo): If we use GPUs, we may have to copy the <m_posteriors>
        // to CPUs first, as the lattice computation happens on CPUs. Otherwise
        // each call to get a single element in the matrix will be slow?
        derivatives.Resize(m_transModel.NumPdfs(), endFrame - startFrame);
        derivatives.SetValue(0);
        for (size_t t = startFrame; t < endFrame; ++t)
        {
            for (size_t i = 0; i < m_posteriors[t].size(); ++i)
            {
                size_t pdf_id = m_posteriors[t][i].first;
                assert(pdf_id < m_transModel.NumPdfs());
                derivatives(pdf_id, t - startFrame) -= m_posteriors[t][i].second; /* Flip the sign */
            }
        }

        // We've used up all the derivatives, reset it.
        if (endFrame >= m_currentUttLength)
        {
            m_derivRead = true;
            if (m_objRead)
            {
                m_currentUttID = L"";
                m_currentUttHasDeriv = false;
                m_currentUttLength = 0;
            }
        }
    }

    template<class ElemType>
    void KaldiSequenceTrainingIO<ElemType>::GetObjectives(size_t startFrame,
                                                          size_t endFrame,
                                                          const std::wstring& uttID,
                                                          Matrix<ElemType>& objectives)
    {
        // Does some sanity check first.
        if (uttID != m_currentUttID)
        {
            RuntimeError("Requested utterance does not matched the utterance that we have computed objectives for: %S v.s. %S\n", uttID.c_str(), m_currentUttID.c_str());
        }
        if (!m_currentUttHasDeriv)
        {
            RuntimeError("Objectives have not been computed, you have to call KaldiSequenceTrainingIO::ComputeDerivative() before using it.\n");
        }
        assert(startFrame >= 0);
        assert(endFrame <= m_currentUttLength);

        objectives.Resize(1, 1);
        objectives.SetValue(m_objective * static_cast<ElemType>(endFrame - startFrame) / static_cast<ElemType>(m_currentUttLength));

        // We've used up all the objectives, reset it.
        if (endFrame >= m_currentUttLength)
        {
            m_objRead = true;
            if (m_derivRead)
            {
                m_currentUttID = L"";
                m_currentUttHasDeriv = false;
                m_currentUttLength = 0;
            }
        }
    }

    template class KaldiSequenceTrainingIO<float>;
    template class KaldiSequenceTrainingIO<double>;
}}}
