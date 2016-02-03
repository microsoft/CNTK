#pragma once

#include "Matrix.h"
#include "basetypes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class defines the interface for utterance derivative computation.
template <class ElemType>
class UtteranceDerivativeComputationInterface
{
public:
    // Computes derivative and objective for given utterance ID and
    // log-likelihood from neural network output.
    virtual bool ComputeDerivative(const wstring& /*uttID*/,
                                   const Matrix<ElemType>& /*logLikelihood*/,
                                   Matrix<ElemType>* /*derivative*/,
                                   ElemType* /*objective*/) = 0;

    // Returns true if we have resources to comptue the derivative, otherwise
    // returns false.
    virtual bool HasResourceForDerivative(const wstring& /*uttID*/) const = 0;
};
} } }
