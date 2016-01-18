#include "basetypes.h"
#include "htkfeatio_utils.h"
#include "KaldiSequenceTrainingDerivative.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Constructor.
template <class ElemType>
KaldiSequenceTrainingDerivative<ElemType>::KaldiSequenceTrainingDerivative(
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
template <class ElemType>
KaldiSequenceTrainingDerivative<ElemType>::~KaldiSequenceTrainingDerivative()
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

template <class ElemType>
bool KaldiSequenceTrainingDerivative<ElemType>::ComputeDerivative(
    const wstring& uttID,
    const Matrix<ElemType>& logLikelihood,
    Matrix<ElemType>* derivative,
    ElemType* objective)
{
    std::string uttIDStr = msra::asr::toStr(uttID);

    // Sanity check.
    if (m_transModel.NumPdfs() != logLikelihood.GetNumRows())
    {
        RuntimeError("Number of labels in logLikelihood does not match that"
                     " in the Kaldi model for utterance %S: %d v.s. %d\n",
                     uttID.c_str(), (int) logLikelihood.GetNumRows(),
                     (int) m_transModel.NumPdfs());
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
                     uttID.c_str(), (int) logLikelihood.GetNumCols(), (int) ali.size());
    }

    // Reads denominator lattice.
    if (!m_denlatReader->HasKey(uttIDStr))
    {
        RuntimeError("Denominator lattice not found for utterance %S\n",
                     uttID.c_str());
    }
    kaldi::CompactLattice clat = m_denlatReader->Value(uttIDStr);
    fst::CreateSuperFinal(&clat); /* One final state with weight One() */
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

    // Does lattice acoustic rescoring with the new posteriors from the
    // neural network.
    LatticeAcousticRescore(uttID, logLikelihood, &lat);

    // Second pass acoustic and language model scale.
    if (m_acousticScale != 1.0 || m_lmScale != 1.0)
    {
        fst::ScaleLattice(fst::LatticeScale(m_lmScale, m_acousticScale),
                          &lat);
    }

    // Forward-backward on the lattice.
    kaldi::Posterior post, pdfPost;
    if (m_trainCriterion == L"smbr")
    {
        *objective = kaldi::LatticeForwardBackwardMpeVariants(
            m_transModel, m_silencePhones, lat,
            ali, "smbr", m_oneSilenceClass, &post);
    }
    else if (m_trainCriterion == L"mpfe")
    {
        *objective = kaldi::LatticeForwardBackwardMpeVariants(
            m_transModel, m_silencePhones, lat,
            ali, "mpfe", m_oneSilenceClass, &post);
    }

    ConvertPosteriorToDerivative(post, derivative);
    assert(derivative->GetNumCols() == logLikelihood.GetNumCols());

    // Uses "expected error rate" instead of "expected accuracy".
    *objective = logLikelihood.GetNumCols() - *objective;

    return true;
}

template <class ElemType>
void KaldiSequenceTrainingDerivative<ElemType>::ConvertPosteriorToDerivative(
    const kaldi::Posterior& post,
    Matrix<ElemType>* derivative)
{
    kaldi::Posterior pdfPost;
    kaldi::ConvertPosteriorToPdfs(m_transModel, post, &pdfPost);

    derivative->Resize(m_transModel.NumPdfs(), pdfPost.size());
    derivative->SetValue(0);

    for (size_t t = 0; t < pdfPost.size(); ++t)
    {
        for (size_t i = 0; i < pdfPost[t].size(); ++i)
        {
            size_t pdf_id = pdfPost[t][i].first;
            assert(pdf_id < m_transModel.NumPdfs());
            // Flips the sign below.
            (*derivative)(pdf_id, t) -= pdfPost[t][i].second;
        }
    }
}

template <class ElemType>
void KaldiSequenceTrainingDerivative<ElemType>::LatticeAcousticRescore(
    const wstring& uttID,
    const Matrix<ElemType>& logLikelihood,
    kaldi::Lattice* lat) const
{
    // Gets time information for the lattice.
    std::vector<kaldi::int32> stateTimes;
    kaldi::int32 maxTime = kaldi::LatticeStateTimes(*lat, &stateTimes);
    if (maxTime != logLikelihood.GetNumCols())
    {
        RuntimeError("Number of frames in the logLikelihood does not match"
                     " that in the denominator lattice for utterance %S: %d vs. %d\n",
                     uttID.c_str(), (int) logLikelihood.GetNumRows(), (int) maxTime);
    }

    std::vector<std::vector<kaldi::int32>> timeStateMap(
        logLikelihood.GetNumCols());
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
            for (fst::MutableArcIterator<kaldi::Lattice> aiter(lat, state);
                 !aiter.Done(); aiter.Next())
            {
                kaldi::LatticeArc arc = aiter.Value();
                kaldi::int32 trans_id = arc.ilabel;
                if (trans_id != 0)
                {
                    kaldi::int32 pdf_id =
                        m_transModel.TransitionIdToPdf(trans_id);
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

template <class ElemType>
bool KaldiSequenceTrainingDerivative<ElemType>::HasResourceForDerivative(
    const wstring& uttID) const
{
    if (m_aliReader == false || m_denlatReader == false)
    {
        fprintf(stderr, "WARNING: lattice or alignemnt reader has not been"
                        " set up yet.\n");
        return false;
    }

    std::string uttIDStr = msra::asr::toStr(uttID);
    if (!m_aliReader->HasKey(uttIDStr) || !m_denlatReader->HasKey(uttIDStr))
    {
        return false;
    }
    return true;
}

template class KaldiSequenceTrainingDerivative<float>;
template class KaldiSequenceTrainingDerivative<double>;
} } }
