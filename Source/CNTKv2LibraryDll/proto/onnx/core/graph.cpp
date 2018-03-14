#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>

#include "graph.h"
#include "op.h"
#include "utils.h"

using namespace ONNXIR::Utils;

namespace ONNXIR
{

#define NO_CHANGE_ON_SYNC_FLAG(...)                 \
    do {                                            \
        bool syncNeeded = m_graphProtoSyncNeeded;   \
        {__VA_ARGS__;}                               \
        m_graphProtoSyncNeeded = syncNeeded;        \
    } while (0)                                     \

    NodeArg::NodeArg(const std::string& p_name,
        const TypeProto* p_nodeArgType)
    {
        m_nodeArgInfo.set_name(p_name);
        // If the name is empty, it means the arg does not exist.
        m_exist = !(p_name.empty());
        if (nullptr != p_nodeArgType)
        {
            (*m_nodeArgInfo.mutable_type()) = *p_nodeArgType;
            m_type = OpUtils::ToType(m_nodeArgInfo.type());
        }
        else
        {
            m_type = nullptr;
        }
    }

    const std::string& NodeArg::Name() const
    {
        return m_nodeArgInfo.name();
    }

    const PTYPE& NodeArg::Type() const
    {
        return m_type;
    }

    const TensorShapeProto* NodeArg::Shape() const
    {
        if (!m_nodeArgInfo.has_type())
        {
            return nullptr;
        }

        auto typeCase = m_nodeArgInfo.type().value_case();
        switch (typeCase)
        {
        case TypeProto::kTensorType:
            return &(m_nodeArgInfo.type().tensor_type().shape());
        case TypeProto::kSequenceType:
        case TypeProto::kMapType:
        case TypeProto::VALUE_NOT_SET:
        default:
            return nullptr;
        }
    }

    void NodeArg::SetShape(const TensorShapeProto& p_shape)
    {
        if (!m_nodeArgInfo.has_type())
        {
            return;
        }

        auto typeCase = m_nodeArgInfo.type().value_case();
        switch (typeCase)
        {
        case TypeProto::kTensorType:
            *(m_nodeArgInfo.mutable_type()->mutable_tensor_type()->mutable_shape()) = p_shape;
            break;
        case TypeProto::kSequenceType:
        case TypeProto::kMapType:
        case TypeProto::VALUE_NOT_SET:
        default:
            return;
        }
    }

    const NodeArgInfo& NodeArg::ToProto() const
    {
        return m_nodeArgInfo;
    }

    void NodeArg::SetType(PTYPE p_type)
    {
        if (nullptr == p_type)
        {
            return;
        }

        m_type = p_type;
        *(m_nodeArgInfo.mutable_type())
            = OpUtils::ToTypeProto(p_type);
    }

    void NodeArg::SetType(const TypeProto& p_typeProto)
    {
        m_type = OpUtils::ToType(p_typeProto);
        *(m_nodeArgInfo.mutable_type()) = p_typeProto;
    }

    bool NodeArg::Exist() const
    {
        return m_exist;
    }

    Node::EdgeEnd::EdgeEnd(const Node& p_node, const NodeArg& p_nodeArg)
        : m_node(&p_node),
        m_nodeArg(&p_nodeArg)
    {
    }

    const Node* Node::EdgeEnd::GetNode() const
    {
        return m_node;
    }

    const NodeArg* Node::EdgeEnd::GetNodeArg() const
    {
        return m_nodeArg;
    }

    Node::NodeConstIterator::NodeConstIterator(
        std::set<const Node*>::const_iterator p_iter)
        : m_iter(p_iter)
    {
    }

    bool Node::NodeConstIterator::operator==(
        const NodeConstIterator& p_other) const
    {
        return m_iter == p_other.m_iter;
    }

    bool Node::NodeConstIterator::operator!=(
        const NodeConstIterator& p_other) const
    {
        return m_iter != p_other.m_iter;
    }

    void Node::NodeConstIterator::operator++()
    {
        ++m_iter;
    }

    const Node* Node::NodeConstIterator::operator*()
    {
        return *m_iter;
    }

    Node::Node(const Node& p_other)
    {
        m_name = p_other.m_name;
        m_opType = p_other.m_opType;
        m_domain = p_other.m_domain;
        m_inputDefs = p_other.m_inputDefs;
        m_inputs = p_other.m_inputs;
        m_inputNodes = p_other.m_inputNodes;
        m_controlInputs = p_other.m_controlInputs;
        m_outputDefs = p_other.m_outputDefs;
        m_outputNodes = p_other.m_outputNodes;
        m_device = p_other.m_device;
        m_attributes = p_other.m_attributes;
    }

    NODEINDEX Node::Index() const
    {
        return m_index;
    }

    const std::string& Node::Name() const
    {
        return m_name;
    }

    const std::string& Node::OpType() const
    {
        return m_opType;
    }

    const std::string& Node::Description() const
    {
        return m_description;
    }

    const std::string& Node::Domain() const
    {
        return m_domain;
    }

    const OperatorSchema* Node::Op() const
    {
        return m_op;
    }

    const std::vector<NodeArg>& Node::InputDefs() const
    {
        return m_inputDefs;
    }

    std::vector<NodeArg>& Node::Mutable_InputDefs()
    {
        m_graph->m_graphResolveNeeded = true;
        m_graph->m_graphProtoSyncNeeded = true;
        return m_inputDefs;
    }

    const std::vector<int>& Node::InputArgCount() const
    {
        return m_inputArgCount;
    }

    std::vector<int>& Node::Mutable_InputArgCount()
    {
        m_graph->m_graphResolveNeeded = true;
        m_graph->m_graphProtoSyncNeeded = true;
        return m_inputArgCount;
    }

    Node::NodeConstIterator Node::InputNodes_begin() const
    {
        return NodeConstIterator(m_inputNodes.begin());
    }

    Node::NodeConstIterator Node::InputNodes_end() const
    {
        return NodeConstIterator(m_inputNodes.end());
    }

    Node::NodeConstIterator Node::OutputNodes_begin() const
    {
        return NodeConstIterator(m_outputNodes.begin());
    }

    Node::NodeConstIterator Node::OutputNodes_end() const
    {
        return NodeConstIterator(m_outputNodes.end());
    }

    bool Node::InputEdgeSrcEnd(NodeArg* p_inputArg,
        /*out*/const EdgeEnd** p_inputEdgeSrcEnd) const
    {
        if (nullptr == p_inputArg
            || nullptr == p_inputEdgeSrcEnd)
        {
            return false;
        }

        auto edgeEndIter = m_inputs.find(p_inputArg);
        if (m_inputs.end() == edgeEndIter)
        {
            // There's no input edge for the specified input argument.
            return false;
        }

        *p_inputEdgeSrcEnd = &(edgeEndIter->second);
        return true;
    }

    const std::vector<NodeArg>& Node::OutputDefs() const
    {
        return m_outputDefs;
    }

    std::vector<NodeArg>& Node::Mutable_OutputDefs()
    {
        m_graph->m_graphResolveNeeded = true;
        m_graph->m_graphProtoSyncNeeded = true;
        return m_outputDefs;
    }

    const std::string& Node::Device() const
    {
        return m_device;
    }

    void Node::SetDevice(const std::string& p_device)
    {
        m_device = p_device;
    }

    void Node::ToProto(NodeProto& p_proto) const
    {
        // Set name.
        p_proto.set_name(m_name);
        // Set op type.
        p_proto.set_op_type(m_opType);
        // Set op domain;
        p_proto.set_domain(m_domain);
        // Set doc string.
        p_proto.set_doc_string(m_description);

        // Set attributes.
        p_proto.clear_attribute();
        for (auto attribute : m_attributes)
        {
            auto attr = p_proto.add_attribute();
            *attr = attribute.second;
        }

        // Set inputs' definitions.
        p_proto.clear_input();
        for (auto& inputDef : m_inputDefs)
        {
            auto input = p_proto.add_input();
            *input = inputDef.Name();
        }

        // Set outputs' definitions.
        p_proto.clear_output();
        for (auto& outputDef : m_outputDefs)
        {
            auto output = p_proto.add_output();
            *output = outputDef.Name();
        }
    }

    void Node::Init(const NodeProto& p_nodeProto,
        const ArgNameToTypeMap& p_nameToType)
    {
        m_name = p_nodeProto.name();
        m_opType = p_nodeProto.op_type();
        m_domain = p_nodeProto.domain();

        for (int i = 0; i < p_nodeProto.input().size(); ++i)
        {
            const TypeProto* type = nullptr;

            auto nameToTypeIter = p_nameToType.find(p_nodeProto.input(i));
            if (p_nameToType.end() != nameToTypeIter)
            {
                // This node input arg type/shape does exist in graph proto.
                // Assign type/shape information to node input arg.
                type = &(nameToTypeIter->second);
            }

            m_inputDefs.push_back(NodeArg(p_nodeProto.input(i), type));
        }

        // Set input arg count as 1:1 maping with input defs.
        // NOTE: it may be refined per operator definition.
        // There will be cases having arg count as, 1, 1, ..., 1, N.
        // It means that the last operator input is variadic.
        m_inputArgCount.assign(m_inputDefs.size(), 1);

        for (int i = 0; i < p_nodeProto.output().size(); ++i)
        {
            const TypeProto* type = nullptr;

            auto nameToTypeIter = p_nameToType.find(p_nodeProto.output(i));
            if (p_nameToType.end() != nameToTypeIter)
            {
                // This output arg type/shape does exist in graph proto.
                // Assign type/shape information to node output arg.
                type = &(nameToTypeIter->second);
            }

            m_outputDefs.push_back(NodeArg(p_nodeProto.output(i), type));
        }

        for (int i = 0; i < p_nodeProto.attribute_size(); ++i)
        {
            auto& attr = p_nodeProto.attribute(i);
            m_attributes[attr.name()] = attr;
        }
    }

    void Node::Init(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_inputArgs,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        Init(p_name, p_opType, p_description, p_outputArgs, p_domain);
        m_inputDefs = p_inputArgs;
        // Set each arg count as 1 by default.
        // It could be adjusted when resolving the node with its operator
        // information.
        m_inputArgCount.assign(m_inputDefs.size(), 1);
    }

    void Node::Init(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_inputArgs,
        const std::vector<int>& p_inputArgCount,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        Init(p_name, p_opType, p_description, p_outputArgs, p_domain);
        m_inputDefs = p_inputArgs;
        m_inputArgCount = p_inputArgCount;
    }

    void Node::Init(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        m_name = p_name;
        m_opType = p_opType;
        m_description = p_description;
        m_outputDefs = p_outputArgs;
        m_domain = p_domain;
    }

    bool Node::AddAttribute(const std::string& p_attrName, const AttributeProto& p_value)
    {
        auto it = m_attributes.find(p_attrName);
        if (it == m_attributes.end())
        {
            m_graph->m_graphResolveNeeded = true;
            m_graph->m_graphProtoSyncNeeded = true;
            m_attributes.emplace(p_attrName, p_value);
            return true;
        }
        else
        {
            return false;
        }
    }

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                               \
    bool Node::AddAttribute(const std::string& p_attrName, const type& p_value)  \
    {                                                                            \
        auto it = m_attributes.find(p_attrName);                                 \
        if (it == m_attributes.end())                                            \
        {                                                                        \
            m_graph->m_graphResolveNeeded = true;                                \
            m_graph->m_graphProtoSyncNeeded = true;                              \
            AttributeProto a;                                                    \
            a.set_name(p_attrName);                                              \
            a.set_type(enumType);                                               \
            a.set_##field(p_value);                                              \
            m_attributes.emplace(p_attrName, a);                                 \
            return true;                                                         \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
    };                                                                           \

#define ADD_ATTR_IMPL(type, enumType, field)                                     \
    bool Node::AddAttribute(const std::string& p_attrName, const type& p_value)  \
    {                                                                            \
        auto it = m_attributes.find(p_attrName);                                 \
        if (it == m_attributes.end())                                            \
        {                                                                        \
            m_graph->m_graphResolveNeeded = true;                                \
            m_graph->m_graphProtoSyncNeeded = true;                              \
            AttributeProto a;                                                    \
            a.set_name(p_attrName);                                              \
            a.set_type(enumType);                                               \
            *(a.mutable_##field()) = p_value;                                    \
            m_attributes.emplace(p_attrName, a);                                 \
            return true;                                                         \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
    };                                                                           \

#define ADD_LIST_ATTR_IMPL(type, enumType, field)                                \
    bool Node::AddAttribute(const std::string& p_attrName,                       \
                            const std::vector<type>& p_values)                   \
    {                                                                            \
        auto it = m_attributes.find(p_attrName);                                 \
        if (it == m_attributes.end())                                            \
        {                                                                        \
            m_graph->m_graphResolveNeeded = true;                                \
            m_graph->m_graphProtoSyncNeeded = true;                              \
            AttributeProto a;                                                    \
            a.set_name(p_attrName);                                              \
            a.set_type(enumType);                                               \
            for (const auto& val : p_values)                                     \
            {                                                                    \
                *(a.mutable_##field()->Add()) = val;                             \
            }                                                                    \
            m_attributes.emplace(p_attrName, a);                                 \
            return true;                                                         \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
    };                                                                           \

    ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
    ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
    ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
    ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR, t)
    ADD_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH, g)
    ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)
    ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)
    ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)
    ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS, tensors)
    ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS, graphs)

    bool Node::ClearAttribute(const std::string& p_attrName)
    {
        m_graph->m_graphResolveNeeded = true;
        m_graph->m_graphProtoSyncNeeded = true;
        return m_attributes.erase(p_attrName) > 0;
    }

    const NodeAttributes& Node::GetAttributes() const
    {
        return m_attributes;
    }

    bool Graph::NodeIterator::operator==(
        const Graph::NodeIterator& p_other) const
    {
        return (m_graph == p_other.m_graph &&
            m_currentNodeIndex == p_other.m_currentNodeIndex);
    }

    bool Graph::NodeIterator::operator!=(
        const Graph::NodeIterator& p_other) const
    {
        return !(*this == p_other);
    }

    void Graph::NodeIterator::operator++()
    {
        while (true)
        {
            m_currentNodeIndex++;
            if (m_currentNodeIndex >= m_graph->MaxNodeIndex()
                || nullptr != m_graph->GetNode(m_currentNodeIndex))
            {
                return;
            }
        }
    }

    Node* Graph::NodeIterator::operator*()
    {
        return m_graph->GetNode(m_currentNodeIndex);
    }

    Graph::Graph(GraphProto* p_graphProto,
        const std::unordered_map<std::string, int>& p_domainToVersion, bool p_isONNX)
        : m_graphProto(p_graphProto),
        m_graphProtoSyncNeeded(false),
        m_graphResolveNeeded(true),
        m_numOfNodes(0)
    {
        m_removedInitializerIndexes.clear();
        m_domainToVersion = &p_domainToVersion;
        // This is a main graph.
        m_graphType |= Type::Main;
        if (!p_isONNX)
        {
            m_graphType |= Type::Strict;
        }

        // Copy initial tensor indexes to a map.
        for (int i = 0; i < m_graphProto->initializer_size(); ++i)
        {
            m_nameToInitialTensorIndex[m_graphProto->initializer()[i].name()] = i;
            m_nameToInitialTensorPtr[m_graphProto->initializer()[i].name()] = m_graphProto->mutable_initializer(i);
        }

        // Collect all node arg name, type, shape information in the graph.
        // type/shape information will be assigned to each node arg when going
        // thru all nodes later.
        ArgNameToTypeMap nameToTypeMap;
        for (auto& graphInput : m_graphProto->input())
        {
            if (graphInput.has_name() && graphInput.has_type())
            {
                nameToTypeMap[graphInput.name()] = graphInput.type();
            }
        }
        for (auto& graphOutput : m_graphProto->output())
        {
            if (graphOutput.has_name() && graphOutput.has_type())
            {
                nameToTypeMap[graphOutput.name()] = graphOutput.type();
            }
        }
        for (auto& nodeArg : m_graphProto->value_info())
        {
            if (nodeArg.has_name() && nodeArg.has_type())
            {
                nameToTypeMap[nodeArg.name()] = nodeArg.type();
            }
        }

        // Add nodes.
        AddSourceSinkNodes();
        for (auto& nodeProto : p_graphProto->node())
        {
            AddNode(nodeProto, nameToTypeMap);
        }
    }

    Status Graph::VerifyNoDuplicateName(
        /*out*/ std::unordered_map<std::string, Node::EdgeEnd>& p_outputArgs,
        /*out*/ std::unordered_map<std::string, NODEINDEX>& p_nodeNameToIndex)
    {
        p_outputArgs.clear();
        p_nodeNameToIndex.clear();

        for (auto nodeIter = Nodes_begin();
            nodeIter != Nodes_end();
            ++nodeIter)
        {
            // Verify node name should be unique.
            auto& nodeName = (*nodeIter)->Name();

            if (!nodeName.empty()
                && p_nodeNameToIndex.end() != p_nodeNameToIndex.find(nodeName))
            {
                // The node has name and its name was used by another node.
                Status status(ONNX,
                    FAIL,
                    "Error: two nodes with same node name (" + nodeName + ").");
                return status;
            }
            p_nodeNameToIndex[nodeName] = (*nodeIter)->Index();

            // Verify node outputs' name should be unique.
            for (auto& outputDef : (*nodeIter)->OutputDefs())
            {
                std::string outputArgname = outputDef.Name();
                if (p_outputArgs.end() != p_outputArgs.find(outputArgname))
                {
                    // Two outputs with same name.
                    Status status(ONNX,
                        FAIL,
                        "Error: two output args with same name ("
                        + outputArgname + ").");
                    return status;
                }
                p_outputArgs.insert(
                { outputArgname, Node::EdgeEnd(*(*nodeIter), outputDef) });
            }
        }
        return Status::OK();
    }

    Status Graph::BuildConnections(
        const std::unordered_map<std::string, Node::EdgeEnd>& p_outputArgs,
        const std::unordered_map<std::string, NODEINDEX>& p_nodeNameToIndex)
    {
        std::unordered_set<Node*> innerNodes;
        for (auto nodeIter = Nodes_begin();
            nodeIter != Nodes_end();
            ++nodeIter)
        {
            if (IsSourceNode((*nodeIter)->Index())
                || IsSinkNode((*nodeIter)->Index()))
            {
                continue;
            }

            for (auto& controlInput : (*nodeIter)->m_controlInputs)
            {
                auto nameToIndexIter = p_nodeNameToIndex.find(controlInput);
                if (p_nodeNameToIndex.end() == nameToIndexIter)
                {
                    Status status(ONNX, FAIL,
                        "The control input (" + controlInput + ") of Node ("
                        + (*nodeIter)->Name() + ") does not exist in the graph.");
                    return status;
                }

                NODEINDEX srcNodeIndex = nameToIndexIter->second;
                NODEINDEX dstNodeIndex = (*nodeIter)->Index();
                m_nodes[srcNodeIndex]->
                    m_outputNodes.insert(m_nodes[dstNodeIndex].get());
                m_nodes[dstNodeIndex]->
                    m_inputNodes.insert(m_nodes[srcNodeIndex].get());
            }

            auto& inputArgs = (*nodeIter)->InputDefs();
            if (inputArgs.size() > 0)
            {
                // This node needs inputs.

                for (auto& inputArg : inputArgs)
                {
                    if (!inputArg.Exist())
                    {
                        // This input could be optional and it does not exist in this case.
                        continue;
                    }

                    auto outputArgIter = p_outputArgs.find(inputArg.Name());
                    if (p_outputArgs.end()
                        == outputArgIter)
                    {
                        // No such outputArg matching this inputArg.
                        // This input arg should be fed when running evaluation.

                        // Add a control edge between <souce> node and this node.
                        AddControlEdge(m_sourceNodeIndex, (*nodeIter)->Index());
                        continue;
                    }

                    // Setup input/output relationship between <*nodeIter>
                    // and <outputArgIter>.
                    (*nodeIter)->m_inputNodes.insert(
                        outputArgIter->second.GetNode());
                    (*nodeIter)->m_inputs.insert({ &inputArg , outputArgIter->second });

                    NODEINDEX outputNodeIndex =
                        outputArgIter->second.GetNode()->Index();
                    m_nodes[outputNodeIndex]->m_outputNodes.insert((*nodeIter));

                    innerNodes.insert(m_nodes[outputNodeIndex].get());
                }
            }
            else
            {
                if ((*nodeIter)->OutputDefs().size() <= 0)
                {
                    // This is a useless node.
                    // It has no input/output.
                    RemoveNode((*nodeIter)->Index());
                }

                // This is a starting node.
                // Add a control edge between <souce> node and this node.
                AddControlEdge(m_sourceNodeIndex, (*nodeIter)->Index());
            }
        }

        for (auto nodeIter = Nodes_begin();
            nodeIter != Nodes_end();
            ++nodeIter)
        {
            if (IsSourceNode((*nodeIter)->Index())
                || IsSinkNode((*nodeIter)->Index()))
            {
                continue;
            }

            if (innerNodes.size() <= 0
                || innerNodes.end() == innerNodes.find((*nodeIter)))
            {
                // This is an ending node.
                // Add a control edge from this node to sink node.
                AddControlEdge((*nodeIter)->Index(), m_sinkNodeIndex);
            }
        }

        return Status::OK();
    }

    Status Graph::CheckIsAcyclic(
        std::vector<NODEINDEX>& p_nodesInTopologicalOrder)
    {
        p_nodesInTopologicalOrder.clear();
        // nodes that have been processed and added to p_nodesInTopologicalOrder.
        std::unordered_set<NODEINDEX> visitedNodes;
        std::unordered_set<NODEINDEX> ancestorNodes;
        // tracks nodes whose child nodes have been processed.
        std::unordered_set<NODEINDEX> childrenVisitedNodes;
        std::stack<NODEINDEX> stack;
        stack.push(m_sinkNodeIndex);

        while (!stack.empty())
        {
            NODEINDEX current = stack.top();
            stack.pop();

            if (visitedNodes.end() != visitedNodes.find(current))
            {
                // The node has been visited before
                continue;
            }

            if (childrenVisitedNodes.end() != childrenVisitedNodes.find(current))
            {
                // children are done so we mark this one complete.
                visitedNodes.insert(current);
                p_nodesInTopologicalOrder.push_back(current);
                ancestorNodes.erase(current);
                continue;
            }

            if (m_nodes[current]->InputNodes_begin() ==
                m_nodes[current]->InputNodes_end())
            {
                // no children
                childrenVisitedNodes.insert(current);
                visitedNodes.insert(current);
                p_nodesInTopologicalOrder.push_back(current);
                ancestorNodes.erase(current);
                continue;
            }

            stack.push(current);

            // mark as children done. by the time the node is popped off the stack again,
            // its children will have been processed
            childrenVisitedNodes.insert(current);

            ancestorNodes.insert(current);

            // check children
            for (auto iter = m_nodes[current]->InputNodes_begin();
                iter != m_nodes[current]->InputNodes_end();
                ++iter)
            {
                NODEINDEX idx = (*iter)->Index();
                if (ancestorNodes.end() != ancestorNodes.find(idx))
                {
                    Status status(ONNX, FAIL,
                        "Error: the graph is not acyclic.");
                    return status;
                }

                // avoid re-processing nodes
                if (childrenVisitedNodes.end() == childrenVisitedNodes.find(idx))
                {
                    stack.push(idx);
                }
            }
        }

        if (this->NumberOfNodes() == p_nodesInTopologicalOrder.size())
        {
            return Status::OK();
        }
        else
        {
            return Status(ONNX, FAIL, "Error: the graph is not acyclic.");
        }
    }

    Status Graph::InferAndVerifyTypeMatch(Node* p_node,
        const OpSignature* p_op,
        const std::unordered_map<std::string, Node::EdgeEnd>& p_outputArgs)
    {
        auto& nodeName = p_node->Name();

        // <k> index used to navigate node->InputDefs().
        int k = 0;
        std::unordered_map<std::string, PTYPE> typeParameterToTypeMap;

        for (size_t i = 0; i < p_node->InputArgCount().size(); ++i)
        {
            // Number of inputs matching to the i-th argument.
            int argCount = p_node->InputArgCount()[i];
            // The i-th argument definition.
            auto opFormalParameter = p_op->GetInputs()[i];

            // Infer and verify all <arguCount> inputs (k-th input)
            // matching operator definition (i-th argument).
            for (int j = 0; j < argCount; ++j, ++k)
            {
                auto& inputDef = p_node->Mutable_InputDefs()[k];

                // For each input arg.
                auto outputArgsIter = p_outputArgs.find(inputDef.Name());
                if (p_outputArgs.end() == outputArgsIter)
                {
                    // This input arg should either be fed by callers,
                    // or be in initializers.
                    // If it's fed by callers, it's needed to have type
                    // information defined well.
                    auto initialTensorIter
                        = m_nameToInitialTensorPtr.find(inputDef.Name());
                    if (m_nameToInitialTensorPtr.end()
                        != initialTensorIter)
                    {
                        // This input is fed with default value by initializer.
                        // Infer its type from initializer tensor.
                        TypeProto initialTensorType;
                        initialTensorType.mutable_tensor_type()->set_elem_type(
                            initialTensorIter->second->data_type());
                        inputDef.SetType(OpUtils::ToType(initialTensorType));

                        // Set shape accordingly.
                        TensorShapeProto shape;
                        for (auto dim : initialTensorIter->second->dims())
                        {
                            shape.add_dim()->set_dim_value(dim);
                        }
                        inputDef.SetShape(shape);
                    }
                    else if (!inputDef.m_nodeArgInfo.has_type())
                    {
                        // This input is fed by callers and its type has to be specified.

                        Status status(ONNX, FAIL,
                            "Node (" + nodeName + ") input arg ("
                            + inputDef.Name()
                            + ") does not have type information.");
                        return status;

                    }
                }
                else
                {
                    // Infer its type by copying from its corresponding
                    // input node's output arg.
                    auto outputArgOfInputNode
                        = outputArgsIter->second.GetNodeArg();

                    inputDef.SetType(outputArgOfInputNode->Type());
                }

                // Verify the input arg type complying with operator
                // definition.

                auto iter = opFormalParameter.GetTypes().find(inputDef.Type());
                if (opFormalParameter.GetTypes().end() == iter)
                {
                    Status status(ONNX, FAIL,
                        "Node (" + nodeName + ") input arg ("
                        + inputDef.Name() + ") type does not match operator ("
                        + p_op->GetName() + ") definition.");
                    return status;
                }

                auto paramToTypeIter = typeParameterToTypeMap.find(opFormalParameter.GetTypeStr());
                if (typeParameterToTypeMap.end() == paramToTypeIter)
                {
                    typeParameterToTypeMap[opFormalParameter.GetTypeStr()]
                        = inputDef.Type();

                }
                else if (paramToTypeIter->second != inputDef.Type() && argCount == 1)
                {
                    // This is the case.
                    // An operator's inputs' type is "T", and T"s allowed value set is "float, int32".
                    // However, one input is specified as "float", and another one is specified as "int".
                    // NOTE: for variadic arguments (argCount > 1), this verification rule is not applicable.
                    // Different types are allowed for variadic arguments although there's only one type "T"
                    // specified in op definition.
                    Status status(ONNX, FAIL,
                        "Node (" + nodeName + ") has different input"
                        " types (" + *(paramToTypeIter->second) + ","
                        + *(inputDef.Type()) + ") matching to same "
                        "type string (" + opFormalParameter.GetTypeStr()
                        + ").");
                    return status;
                }
            }
        }

        // Infer and verify node output arg type information.
        int i = 0;
        for (auto& outputDef : p_node->Mutable_OutputDefs())
        {
            // For each output arg.

            auto opFormalParameter = p_op->GetOutputs()[i++];

            // Infer output arg type per input arg type if they share
            // the same type string. For example, type string is "T" 
            // for both input arg and output arg.
            auto inputTypesIter
                = typeParameterToTypeMap.find(opFormalParameter.GetTypeStr());
            if (typeParameterToTypeMap.end() != inputTypesIter)
            {
                outputDef.SetType(inputTypesIter->second);
                continue;
            }

            if (typeParameterToTypeMap.empty())
            {
                // There's no input arg.
                // The output should be read from an attribute named c_constantValue.

                auto nodeAttributesIter
                    = p_node->GetAttributes().find(c_constantValue);
                if (p_node->GetAttributes().end() == nodeAttributesIter)
                {
                    Status status(ONNX, FAIL,
                        "Node (" + nodeName + ") output arg value should"
                        "be specified via node attribute '" + c_constantValue + "'.");
                    return status;
                }

                AttrType attrType;
                RETURN_IF_ERROR(TypeUtils::GetType(nodeAttributesIter->second, attrType));
                if (AttrType::AttributeProto_AttributeType_TENSOR == attrType)
                {
                    auto& tensor = nodeAttributesIter->second.t();
                    TypeProto typeProto;
                    typeProto.mutable_tensor_type()->set_elem_type(tensor.data_type());
                    outputDef.SetType(OpUtils::ToType(typeProto));
                }
                else
                {
                    Status status(ONNX, FAIL,
                        "For attribute " + c_constantValue + " , only Tensor type"
                        "is allowed. The attribute type in this model is "
                        + ONNXIR::c_attrTypeStr[(int)attrType] + ".");
                    return status;
                }

                continue;
            }

            // For case that input arg and output arg have different types.
            if (outputDef.m_nodeArgInfo.has_type())
            {
                // The output arg has already had type information.
                // Check whether it matches operator definition.
                auto iter = opFormalParameter.GetTypes().find(outputDef.Type());
                if (opFormalParameter.GetTypes().end() == iter)
                {
                    Status status(ONNX, FAIL,
                        "Node (" + nodeName + ") output arg ("
                        + outputDef.Name() + ") type does not match operator ("
                        + p_op->GetName() + ") definition.");
                    return status;
                }
                continue;
            }

            // Output arg has no type information.
            if (1 == opFormalParameter.GetTypes().size())
            {
                // Infer output arg type as the only one type defined
                // in operator definition.
                outputDef.SetType(*(opFormalParameter.GetTypes().begin()));
                continue;
            }

            // Output arg has no type information, and there're
            // multiple allowed types defined in operator definition.
            // Type inference fails in this case.
            Status status(ONNX, FAIL,
                "Node (" + nodeName + ") output arg ("
                + outputDef.Name() + ") type inference failed");
            return status;
        }

        return Status::OK();
    }

    Status Graph::VerifyNodeAndOpMatch(
        const std::vector<NODEINDEX>& p_nodesInTopologicalOrder,
        std::unordered_map<std::string, Node::EdgeEnd>& p_outputArgs)
    {
        for (auto nodeIndex : p_nodesInTopologicalOrder)
        {
            if (IsSourceNode(nodeIndex)
                || IsSinkNode(nodeIndex))
            {
                continue;
            }

            auto node = GetNode(nodeIndex);
            auto& nodeName = node->Name();
            auto& op_type = node->OpType();
            auto& domain = node->Domain();
            auto versionIter = m_domainToVersion->find(domain);
            if (m_domainToVersion->end() == versionIter)
            {
                // The domain referred by this node does not exist either
                // in <OpSetIdProto> in the <ModelProto> loaded (in the case of model loaded from file) or
                // in global DomainToVersionRange map (in the case of model constructed from scratch).
                return Status(ONNX, FAIL, "The op domain (" + domain + ") used by node ("
                    + nodeName + ") is not supported for this model.");
            }

            // Get op schema given op name, max inclusive version and domain.
            node->m_op = OpSchemaRegistry::Schema(op_type, versionIter->second, domain);
            if (nullptr == node->m_op)
            {
                // A op_type refers to nothing.
                Status status(ONNX, FAIL,
                    "Error: the operator or function (" + op_type
                    + ") refered by node (" + nodeName
                    + ") does not exist.");
                return status;
            }

            auto& op = node->Op()->GetOpSignature();

            // The node refers to a primitive operator.
            // Infer and verify node input arg type information.
            auto totalArgCount = std::accumulate(node->InputArgCount().begin(),
                node->InputArgCount().end(), 0);
            if (totalArgCount != node->InputDefs().size())
            {
                Status status(ONNX, FAIL,
                    "The sum of input arg count is not equal to size of"
                    "input defs in node (" + nodeName + ").");
                return status;
            }

            // Verify size of node arg count is same as input number in
            // operator definition.
            if (op.GetInputs().size() != node->InputArgCount().size())
            {
                if (0 == (m_graphType & Type::Strict))
                {
                    // It's ONNX case.
                    // Adjust input arg count array with op definition
                    // The adjustment will work as below,
                    // In total, there're <totalArgCount> inputs, which
                    // will be split as <1, 1, 1, 1, ... 1, x> or
                    // <1, 1, 1, 1, ...1, 0, 0, ...0>. The final input 
                    // arg count array's element number will be the same
                    // as op definition, and the sum of all elements will
                    // be equal to <totalArgCount>.
                    auto& inputArgCount = node->Mutable_InputArgCount();
                    inputArgCount.clear();
                    size_t m = 0;
                    auto argCountLeft = totalArgCount;
                    if (0 < op.GetInputs().size())
                    {
                        for (; m < op.GetInputs().size() - 1; ++m)
                        {
                            if (argCountLeft > 0)
                            {
                                inputArgCount.push_back(1);
                                argCountLeft--;
                            }
                            else
                            {
                                inputArgCount.push_back(0);
                            }
                        }
                    }

                    // Set the arg count for the last input formal parameter.
                    // NOTE: in the case that there's no .input(...) defined
                    // in op schema, all input args will be fed as one input
                    // of the operator.
                    inputArgCount.push_back(argCountLeft);
                }
                else
                {
                    // Number of inputs do not match.
                    Status status(ONNX, FAIL, "Error: node (" + nodeName
                        + ")'s number of inputs do not match its operator ("
                        + op_type + ") specification.");
                    return status;
                }
            }

            // Verify node outputs have same size with operator definition.
            if (op.GetOutputs().size() != node->OutputDefs().size())
            {
                if (0 != (m_graphType & Type::Strict))
                {
                    // Number of outputs do not match.
                    Status status(ONNX, FAIL, "Error: node (" + nodeName
                        + ")'s number of outputs does not match its operator ("
                        + op_type + ") specification.");
                    return status;
                }
            }

            if (0 != (m_graphType & Type::Strict))
            {
                // Strict type checking needed.
                NO_CHANGE_ON_SYNC_FLAG(RETURN_IF_ERROR(InferAndVerifyTypeMatch(node, &op, p_outputArgs)));
            }

            // Attribute verification and fill node attribute with
            // default value defined in operator definition if needed.
            auto attrParser = node->Op()->GetAttributeParser();
            if (nullptr != attrParser)
            {
                // Attribute parser registered.
                // Verifying attribute match by running attribute parser.
                RETURN_IF_ERROR(attrParser(node->GetAttributes()));
            }
            else
            {
                // No attribute parser registered.
                auto nodeAttributes = node->GetAttributes();
                for (auto attrDef : op.GetAttributes())
                {
                    auto nodeAttrIter = nodeAttributes.find(attrDef.GetName());
                    if (nodeAttributes.end() == nodeAttrIter)
                    {
                        const AttributeProto* defaultValue = nullptr;
                        bool hasDefaultValue
                            = attrDef.HasDefaultValue(&defaultValue);
                        if (hasDefaultValue)
                        {
                            // Set default value to the node attributes.
                            node->AddAttribute(attrDef.GetName(), *defaultValue);
                        }
                    }
                    else
                    {
                        // Verify node attribute type matching type of
                        // attribute defined in operator definition.
                        AttrType nodeAttrType;
                        RETURN_IF_ERROR(TypeUtils::GetType(nodeAttrIter->second, nodeAttrType));
                        if (nodeAttrType != attrDef.GetType())
                        {
                            Status status(ONNX, FAIL,
                                "Node (" + nodeName + ") attribute ("
                                + nodeAttrIter->first + ") type does not match operator definition.");
                            return status;
                        }
                    }
                }
            }
        }

        return Status::OK();
    }

    Status Graph::Resolve()
    {
        if (!m_graphResolveNeeded)
        {
            return Status::OK();
        }

        std::unordered_map<std::string, Node::EdgeEnd> outputArgs;
        std::unordered_map<std::string, NODEINDEX> nodeNameToIndex;
        RETURN_IF_ERROR(VerifyNoDuplicateName(outputArgs, nodeNameToIndex));
        RETURN_IF_ERROR(BuildConnections(outputArgs, nodeNameToIndex));
        RETURN_IF_ERROR(CheckIsAcyclic(m_nodesInTopologicalOrder));
        RETURN_IF_ERROR(VerifyNodeAndOpMatch(m_nodesInTopologicalOrder, outputArgs));
        RETURN_IF_ERROR(SetGraphInputsOutputs());

        m_graphResolveNeeded = false;
        return Status::OK();
    }

    Status Graph::GetNodesInTopologicalOrder(std::vector<NODEINDEX>** nodes)
    {
        RETURN_IF_ERROR(Resolve());

        *nodes = &m_nodesInTopologicalOrder;
        return Status::OK();
    }

    void Graph::AddSourceSinkNodes()
    {
        std::vector<NodeArg> emptyArgs;
        m_sourceNodeIndex = AddNode("_Graph_Source",
            c_noOp,
            "Source node internally in a graph.",
            emptyArgs,
            emptyArgs)->Index();
        m_sinkNodeIndex = AddNode("_Graph_Sink",
            c_noOp,
            "Sink node internally in a graph.",
            emptyArgs,
            emptyArgs)->Index();
        AddControlEdge(m_sourceNodeIndex, m_sinkNodeIndex);
    }

    const std::string& Graph::Name() const
    {
        return m_graphProto->name();
    }

    void Graph::SetName(const std::string& p_name)
    {
        m_graphProto->set_name(p_name);
    }

    const std::string& Graph::Description() const
    {
        return m_graphProto->doc_string();
    }

    void Graph::SetDescription(const std::string& p_desription)
    {
        m_graphProto->set_doc_string(p_desription);
    }

    void Graph::AddInitialTensor(const TensorProto& p_tensor)
    {
        if (m_nameToInitialTensorPtr.end() != m_nameToInitialTensorPtr.find(p_tensor.name()))
        {
            return;
        }

        auto tensorAdded = m_graphProto->add_initializer();
        *(tensorAdded) = p_tensor;
        m_nameToInitialTensorIndex[p_tensor.name()] = m_graphProto->initializer_size() - 1;
        m_nameToInitialTensorPtr[p_tensor.name()] = tensorAdded;
        m_graphProtoSyncNeeded = true;
        m_graphResolveNeeded = true;
    }

    void Graph::RemoveInitialTensor(const std::string& p_tensorName)
    {
        auto iter = m_nameToInitialTensorIndex.find(p_tensorName);
        if (m_nameToInitialTensorIndex.end() != iter)
        {
            m_removedInitializerIndexes.push_back(iter->second);
            m_nameToInitialTensorIndex.erase(p_tensorName);
            m_nameToInitialTensorPtr.erase(p_tensorName);
            m_graphProtoSyncNeeded = true;
            m_graphResolveNeeded = true;
        }
    }

    bool Graph::GetInitialTensor(const std::string& p_tensorName,
        const TensorProto** p_value) const
    {
        auto iter = m_nameToInitialTensorPtr.find(p_tensorName);
        if (m_nameToInitialTensorPtr.end() == iter)
        {
            return false;
        }
        *p_value  = iter->second;
        return true;
    }

    const InitialTensorSet& Graph::GetAllInitialTensors() const
    {
        return m_nameToInitialTensorPtr;
    }

    const std::vector<const NodeArg*>& Graph::GetInputs() const
    {
        return m_graphInputs;
    }

    const std::vector<const NodeArg*>& Graph::GetOutputs() const
    {
        return m_graphOutputs;
    }

    const std::vector<const NodeArg*>& Graph::GetValueInfo() const
    {
        return m_valueInfo;
    }

    Node* Graph::GetNode(NODEINDEX p_nodeIndex)
    {
        if (MaxNodeIndex() <= p_nodeIndex)
        {
            return nullptr;
        }

        return m_nodes[p_nodeIndex].get();
    }

    Graph::NodeIterator Graph::Nodes_begin()
    {
        return Graph::NodeIterator(0, this);
    }

    Graph::NodeIterator Graph::Nodes_end()
    {
        return Graph::NodeIterator(MaxNodeIndex(), this);
    }

    NODEINDEX Graph::MaxNodeIndex() const
    {
        return m_nodes.size();
    }

    int Graph::NumberOfNodes() const
    {
        return m_numOfNodes;
    }

    Node* Graph::AddNode(const NodeProto& p_nodeProto,
        const ArgNameToTypeMap& p_nameToType)
    {
        auto node = AllocateNode();
        node->Init(p_nodeProto, p_nameToType);
        return node;
    }

    Node* Graph::AddNode(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_inputArgs,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        auto node = AllocateNode();
        node->Init(p_name, p_opType, p_description, p_inputArgs, p_outputArgs, p_domain);
        if (0 != p_opType.compare(c_noOp))
        {
            m_graphProtoSyncNeeded = true;
        }
        return node;
    }

    Node* Graph::AddNode(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_inputArgs,
        const std::vector<int>& p_inputArgCount,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        auto node = AllocateNode();
        node->Init(p_name,
            p_opType,
            p_description,
            p_inputArgs,
            p_inputArgCount,
            p_outputArgs,
            p_domain);
        m_graphProtoSyncNeeded = true;
        return node;
    }

    Node* Graph::AddNode(const std::string& p_name,
        const std::string& p_opType,
        const std::string& p_description,
        const std::vector<NodeArg>& p_outputArgs,
        const std::string& p_domain)
    {
        auto node = AllocateNode();
        node->Init(p_name,
            p_opType,
            p_description,
            p_outputArgs,
            p_domain);
        m_graphProtoSyncNeeded = true;
        return node;
    }

    Node* Graph::AddNode(const Node& p_other)
    {
        auto node = AllocateNode();
        *node = p_other;
        m_graphProtoSyncNeeded = true;
        return node;
    }

    bool Graph::RemoveNode(NODEINDEX p_index)
    {
        if (MaxNodeIndex() <= p_index || nullptr == m_nodes[p_index])
        {
            return false;
        }

        ReleaseNode(p_index);
        return true;
    }

    Node* Graph::AddConstantNode(const std::string& p_name,
        const std::string& p_description,
        const std::vector<NodeArg>& p_outputArgs,
        const TensorProto& p_tensor)
    {
        Node* node = AddNode(p_name, c_constantOp, p_description, p_outputArgs);
        node->AddAttribute(c_constantValue, p_tensor);
        return node;
    }

    bool Graph::AddControlEdge(NODEINDEX p_srcNodeIndex,
        NODEINDEX p_dstNodeIndex)
    {
        if (MaxNodeIndex() <= p_srcNodeIndex
            || MaxNodeIndex() <= p_dstNodeIndex
            || nullptr == m_nodes[p_srcNodeIndex]
            || nullptr == m_nodes[p_dstNodeIndex])
        {
            // Invalid node indexes specified.
            return false;
        }
        m_nodes[p_srcNodeIndex]->
            m_outputNodes.insert(m_nodes[p_dstNodeIndex].get());
        m_nodes[p_dstNodeIndex]->
            m_inputNodes.insert(m_nodes[p_srcNodeIndex].get());
        m_nodes[p_dstNodeIndex]->
            m_controlInputs.insert(m_nodes[p_srcNodeIndex]->Name());

        if (!IsSourceNode(p_srcNodeIndex)
            && !IsSinkNode(p_dstNodeIndex))
        {
            m_graphProtoSyncNeeded = true;
            m_graphResolveNeeded = true;
        }

        return true;
    }

    const GraphProto& Graph::ToGraphProto()
    {
        if (!m_graphProtoSyncNeeded)
        {
            return *m_graphProto;
        }

        // Nodes.
        m_graphProto->clear_node();

        // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
        for (auto& nodeIdx : m_nodesInTopologicalOrder)
        {
            if (IsSourceNode(nodeIdx)
                || IsSinkNode(nodeIdx))
            {
                continue;
            }
            auto nodeProto = m_graphProto->add_node();
            m_nodes[nodeIdx]->ToProto(*nodeProto);
        }

        if (m_removedInitializerIndexes.size() > 0)
        {
            // Move initializers.
            std::sort(m_removedInitializerIndexes.begin(), m_removedInitializerIndexes.end());
            int lastInUseInitializerIndex = m_graphProto->initializer_size() - 1;
            int start = 0, end = static_cast<int>(m_removedInitializerIndexes.size()) - 1;
            int lastRemovedInitializerIndex = m_removedInitializerIndexes[end];

            for (; start <= end; start++)
            {
                // Find a lastInUseInitializer.
                while (start <= end && lastInUseInitializerIndex == lastRemovedInitializerIndex)
                {
                    m_graphProto->mutable_initializer()->RemoveLast();
                    lastInUseInitializerIndex--;
                    end--;
                    if (start <= end)
                    {
                        lastRemovedInitializerIndex = m_removedInitializerIndexes[end];
                    }
                }

                if (start <= end)
                {
                    // Copy the <lastInUseInitializerIndex> initializer in use to the <start> slot which is removed.
                    *m_graphProto->mutable_initializer(m_removedInitializerIndexes[start]) = m_graphProto->initializer(lastInUseInitializerIndex);
                    m_graphProto->mutable_initializer()->RemoveLast();
                    lastInUseInitializerIndex--;
                }
            }
            m_removedInitializerIndexes.clear();
        }

        // Sync graph inputs/outputs/valueInfo.
        SyncGraphInputsOutputs();

        m_graphProtoSyncNeeded = false;

        return *m_graphProto;
    }

    void Graph::SyncGraphInputsOutputs()
    {
        m_graphProto->clear_input();
        m_graphProto->clear_output();
        m_graphProto->clear_value_info();

        for (auto inputArg : m_graphInputs)
        {
            *(m_graphProto->mutable_input()->Add()) = inputArg->ToProto();
        }

        for (auto outputArg : m_graphOutputs)
        {
            *(m_graphProto->mutable_output()->Add()) = outputArg->ToProto();
        }

        for (auto valueInfo : m_valueInfo)
        {
            *(m_graphProto->mutable_value_info()->Add()) = valueInfo->ToProto();
        }
    }

    Status Graph::SetGraphInputsOutputs()
    {
        // Reset graphInputs/graphOutputs/valueInfo state.
        m_graphInputs.clear();
        m_graphOutputs.clear();
        m_valueInfo.clear();

        // Flag indicates that this graph is loaded from model file.
        // If it's true, then graph inputs and outputs will keep the same
        // as what are specified in the model, otherwise, graph inputs
        // and outputs will be inferred.
        bool loadedFromModelFile = m_graphProto->input_size() != 0
            || m_graphProto->output_size() != 0
            || m_graphProto->value_info_size() != 0;

        std::unordered_set<std::string> addedInputNames{};
        if (loadedFromModelFile)
        {
            // Collect all graph inputs/outputs specified in original graph proto
            std::unordered_set<std::string> specifiedGraphInputs;
            std::unordered_set<std::string> specifiedGraphOutputs;
            std::unordered_set<std::string> specifiedGraphValueInfo;
            std::unordered_set<std::string> specifiedInitializers;
            for (auto& graphInput : m_graphProto->input())
            {
                specifiedGraphInputs.insert(graphInput.name());
            }
            for (auto& graphOutput : m_graphProto->output())
            {
                specifiedGraphOutputs.insert(graphOutput.name());
            }
            for (auto& graphValueInfo : m_graphProto->value_info())
            {
                specifiedGraphValueInfo.insert(graphValueInfo.name());
            }
            for (auto& initializer : m_graphProto->initializer())
            {
                specifiedInitializers.insert(initializer.name());
            }

            std::unordered_map<std::string, const NodeArg*> outputNameToNodeArg;
            for (auto nodeIter = Nodes_begin();
                nodeIter != Nodes_end();
                ++nodeIter)
            {
                for (auto& outputDef : (*nodeIter)->OutputDefs())
                {
                    if (specifiedGraphOutputs.erase(outputDef.Name()) >= 1)
                    {
                        m_graphOutputs.push_back(&outputDef);
                    }
                    outputNameToNodeArg.insert({ outputDef.Name(), &outputDef });
                }
            }
            if (specifiedGraphOutputs.size() != 0)
            {
                return Status(ONNX, FAIL, "Some graph outputs which don't exist in the graph.");
            }

            for (auto nodeIter = Nodes_begin();
                nodeIter != Nodes_end();
                ++nodeIter)
            {
                // Go thru all node's inputs.
                for (auto& inputArg : (*nodeIter)->InputDefs())
                {
                    if (!inputArg.Exist())
                    {
                        // It's an optional input and does not exist in this case.
                        continue;
                    }

                    if (specifiedGraphInputs.end() != specifiedGraphInputs.find(inputArg.Name()))
                    {
                        if (addedInputNames.end() == addedInputNames.find(inputArg.Name()))
                        {
                            // The node input is specified as graph input.
                            m_graphInputs.push_back(&inputArg);
                            addedInputNames.insert(inputArg.Name());
                        }
                        continue;
                    }

                    auto outputArgIter = outputNameToNodeArg.find(inputArg.Name());
                    if (outputNameToNodeArg.end() == outputArgIter
                        && specifiedInitializers.end() == specifiedInitializers.find(inputArg.Name()))
                    {
                        // The node input is not specified as graph input,
                        // and it's not fed by another node neither.
                        return Status(ONNX, FAIL, "Node input (" + inputArg.Name() + ") should be a graph input.");
                    }

                    if (specifiedGraphValueInfo.erase(inputArg.Name()) >= 1)
                    {
                        m_valueInfo.push_back(&inputArg);
                    }
                }
            }
        }
        else
        {
            std::unordered_map<std::string, const NodeArg*> outputNameToNodeArg;
            for (auto nodeIter = Nodes_begin();
                nodeIter != Nodes_end();
                ++nodeIter)
            {
                for (auto& outputDef : (*nodeIter)->OutputDefs())
                {
                    outputNameToNodeArg.insert({ outputDef.Name(), &outputDef });
                }
            }
            // Init graph output args with all node output args.
            auto graphOutputArgs = outputNameToNodeArg;

            std::unordered_set<Node*> innerNodes;
            for (auto nodeIter = Nodes_begin();
                nodeIter != Nodes_end();
                ++nodeIter)
            {
                // Go thru all node's inputs.
                for (auto& inputArg : (*nodeIter)->InputDefs())
                {
                    if (!inputArg.Exist())
                    {
                        // It's an optional input and does not exist in this case.
                        continue;
                    }

                    auto outputArgIter = outputNameToNodeArg.find(inputArg.Name());
                    if (outputNameToNodeArg.end() == outputArgIter)
                    {
                        // This input arg should be fed when running evaluation.
                        // it should be a graph input.
                        if (addedInputNames.end() == addedInputNames.find(inputArg.Name()))
                        {
                            // This graph input has not been added into <m_graphInputs>.
                            m_graphInputs.push_back(&inputArg);
                            addedInputNames.insert(inputArg.Name());
                        }
                    }
                    else if (graphOutputArgs.erase(outputArgIter->first) >= 1)
                    {
                        // Remove the output arg name from graph outputs since it's
                        // the input of another node, which we call it intermediate result
                        // and store it in <m_valueinfo>.
                        m_valueInfo.push_back(&inputArg);
                    }
                }
            }

            // Set graph outputs.
            for (auto& outputArg : graphOutputArgs)
            {
                m_graphOutputs.push_back(outputArg.second);
            }
        }

        return Status::OK();
    }

    bool Graph::IsSourceNode(NODEINDEX p_index) const
    {
        return m_sourceNodeIndex == p_index;
    }

    bool Graph::IsSinkNode(NODEINDEX p_index) const
    {
        return m_sinkNodeIndex == p_index;
    }

    const Node* Graph::SourceNode() const
    {
        return m_nodes[m_sourceNodeIndex].get();
    }

    const Node* Graph::SinkNode() const
    {
        return m_nodes[m_sinkNodeIndex].get();
    }

    Node* Graph::AllocateNode()
    {
        std::unique_ptr<Node> node(new Node(MaxNodeIndex(), this));
        m_nodes.push_back(std::move(node));
        m_numOfNodes++;
        m_graphResolveNeeded = true;
        return m_nodes.back().get();
    }

    void Graph::ReleaseNode(NODEINDEX p_nodeIndex)
    {
        m_nodes[p_nodeIndex] = nullptr;
        m_numOfNodes--;
        m_graphProtoSyncNeeded = true;
        m_graphResolveNeeded = true;
    }
}
