#include "shape_inference.h"

namespace ONNXIR
{
    InferenceContext::InferenceContext(Node* p_node,
        const OpSignature* p_opSchema)
        : m_node(p_node),
        m_opSignature(p_opSchema)
    {
    }

    const Node* InferenceContext::GetNode() const
    {
        return m_node;
    }

    const OpSignature* InferenceContext::GetOp() const
    {
        return m_opSignature;
    }

    const std::vector<NodeArg>* InferenceContext::GetInputs() const
    {
        if (nullptr == m_node)
        {
            return nullptr;
        }
        return &(m_node->InputDefs());
    }

    std::vector<NodeArg>* InferenceContext::Mutable_Outputs()
    {
        if (nullptr == m_node)
        {
            return nullptr;
        }
        return &(m_node->Mutable_OutputDefs());
    }
}
