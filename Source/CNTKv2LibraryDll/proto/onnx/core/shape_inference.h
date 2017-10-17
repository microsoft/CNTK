#ifndef CORE_GRAPH_SHAPEINFERENCE_H
#define CORE_GRAPH_SHAPEINFERENCE_H

#include "graph.h"
#include "opsignature.h"

namespace ONNXIR
{

    // A context to contain information for shape inference function.
    // It includes the operator registry, input arguments definition,
    // and mutable output arguments, whose shapes needs to be filled.
    class InferenceContext
    {
    public:

        // TODO: Add input tensors into constructor.
        // TODO: An abstract tensor interface will be needed.
        // In some cases, node evaluation will be needed to get output shapes.
        InferenceContext(Node* p_node,
            const OpSignature* p_opSchema);

        const Node* GetNode() const;

        const OpSignature* GetOp() const;

        const std::vector<NodeArg>* GetInputs() const;

        std::vector<NodeArg>* Mutable_Outputs();

    private:

        Node* m_node;

        const OpSignature* m_opSignature;
    };

    // Shape inference function define.
    typedef std::function<Status(InferenceContext&)> ShapeInferenceFunc;
}
#endif