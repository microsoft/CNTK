#include "proto/onnx/core/common/CommonSTD.h"
#include "proto/onnx/core/graph/graph_transformer.h"

namespace ONNXIR
{

Status GraphTransformerManager::ApplyAll(Graph* graph)
{
    bool changed = false;
    for (unsigned step = 0; step < steps_; ++step)
    {
        for (auto& transformer : transformers_)
        {
            bool t_changed = false;
            Status s = transformer->Apply(graph, &t_changed);
            if (!s.IsOK())
                return s;
            changed = changed || t_changed;
        }
        if (!changed)
            break;
    }
    return Status::OK();
}

} // namespace ONNXIR
