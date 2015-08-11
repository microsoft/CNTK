#pragma once

#include "kaldi.h"
#include "Matrix.h"
#include "UtteranceDerivativeComputationInterface.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class deals with the CTC training in CNTK.
template<class ElemType>
class CtcTrainingIO :
public UtteranceDerivativeComputationInterface<ElemType>
{
private:
    wstring m_trainCriterion;
    wstring m_currentUttID;
    kaldi::RandomAccessInt32VectorReader* m_labRspecifier;        /*label sequence*/

public:
    // Ctc Constructor.
    CtcTrainingIO( const wstring& labRspecifier, const wstring& trainCriterion);

    // Ctc Destructor.
    ~CtcTrainingIO();

    virtual bool ComputeDerivative(const wstring& uttID,
                                   const Matrix<ElemType>& logLikelihoodIn,
                                   Matrix<ElemType>* derivative,
                                   ElemType* objective);

    bool HasResourceForDerivative(const wstring& uttID) const;

protected:
    void ComputeCtcLatticeForward(
          Matrix<ElemType> &alpha,
          Matrix<ElemType> &prob,
          int row,
          std::vector<size_t> &labels);

    void ComputeCtcLatticeBackward(
          Matrix<ElemType> &beta,
          Matrix<ElemType> &prob,
          int row,
          std::vector<size_t> &labels);

    void ComputeCtcError(
            Matrix<ElemType> &ctc_err,
            Matrix<ElemType> &alpha,
            Matrix<ElemType> &beta,
            Matrix<ElemType> &log_nnet_out,
            std::vector<size_t> &labels,
            ElemType pzx);

    virtual bool ComputeDerivativeActual(const wstring& uttID,
                                   const Matrix<ElemType>& logLikelihoodIn,
                                   Matrix<ElemType>* derivative,
                                   ElemType* objective);

    virtual bool ComputeDerivativeNumerical(const wstring& uttID,
                                   const Matrix<ElemType>& logLikelihoodIn,
                                   Matrix<ElemType>* derivative,
                                   ElemType* objective);

    /*
    static const ElemType log_zero_ = -1e100;
    static const ElemType exp_limit_ = 709.78271289338397;
    static const ElemType log_inf_ = 1e100;
    static const ElemType max_ = 1.7976931348623157e+308;
    */

    static const ElemType log_zero_ = -1e30f;
    static const ElemType exp_limit_ = 88.722839f;
    static const ElemType log_inf_ = 1e30f;
    static const ElemType max_ = 3.4028235e+038f;

    ElemType AddAB(ElemType a, ElemType b);
    ElemType SubAB(ElemType a, ElemType b);
    ElemType ExpA(ElemType a);
    ElemType LogAPlusB(ElemType a, ElemType b);


};

}}}
