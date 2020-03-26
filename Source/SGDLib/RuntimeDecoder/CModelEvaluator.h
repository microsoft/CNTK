#pragma once

#include "CMatrix.h"
#include "CVector.h"
#include "CModelParams.h"
#include "cntk_nodes.h"
#include "rnnt_nodes.h"

//
// Encapsulates the encoder, predictor, and joint networks of an RNNT
//
class CModelEvaluator
{
public:
    CModelEvaluator(const CModelParams& params)
        : m_params(params),
          m_pEncoder(make_unique_encoder(m_params)),
          m_pPredictor(make_unique_predictor(m_params)),
          m_pJoint(make_unique_joint(m_params)),
          Encoder(*m_pEncoder),
          Predictor(*m_pPredictor),
          Joint(*m_pJoint)
    {
    }

    void Reset()
    {
        Encoder.Reset();
        Predictor.Reset();
    }

private:
    const CModelParams& m_params;
    std::unique_ptr<IEncoder> m_pEncoder;
    std::unique_ptr<IPredictor> m_pPredictor;
    std::unique_ptr<IJoint> m_pJoint;

public:
    IEncoder& Encoder;
    IPredictor& Predictor;
    IJoint& Joint;
};
