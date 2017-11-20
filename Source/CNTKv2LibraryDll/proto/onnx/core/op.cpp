#include "op.h"
#include "opsignature.h"
#include "utils.h"
#include <cstring>

namespace ONNXIR
{
    const std::string& OperatorSchema::GetName() const
    {
        return m_opSignature.GetName();
    }

    const OpSignature& OperatorSchema::GetOpSignature() const
    {
        return m_opSignature;
    }

    ShapeInferenceFunc OperatorSchema::GetShapeInferenceFn() const
    {
        return m_shapeInferenceFunc;
    }

    AttributeParser OperatorSchema::GetAttributeParser() const
    {
        return m_attrParser;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Name(const std::string& p_opName)
    {
        m_opSchema.m_opSignature.m_name = p_opName;
        return *this;
    }



    OperatorSchemaSetter&
        OperatorSchemaSetter::Description(const std::string& p_description)
    {
        m_opSchema.m_opSignature.m_description = p_description;
        return *this;
    }

    #pragma warning(disable : 4100) // unused p_optional
    OperatorSchemaSetter&
        OperatorSchemaSetter::Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type,
            bool p_optional) /* TODO: add logic for this */
    {
        m_inputs.push_back(std::make_tuple(p_inputName, p_description, p_type));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type)
    {
        m_outputs.push_back(std::make_tuple(p_outputName, p_description, p_type));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool /*required*/)
    {
        m_opSchema.m_opSignature.m_attributes.push_back(
            OpSignature::Attribute(p_attrName, p_attrType, p_description));
        return *this;
    }

#define ATTR_SETTER_BASIC_IMPL(type, field)                                               \
    OperatorSchemaSetter&                                                                 \
        OperatorSchemaSetter::Attr(const std::string& p_attrName,                         \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const type& p_defaultValue)                                                   \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        a.set_##field(p_defaultValue);                                                    \
                                                                                          \
        m_opSchema.m_opSignature.m_attributes.push_back(                                  \
            OpSignature::Attribute(p_attrName,                                            \
                                        p_attrType,                                       \
                                        p_description,                                    \
                                        a));                                              \
                                                                                          \
        return *this;                                                                     \
    }                                                                                     \

#define ATTR_SETTER_LIST_IMPL(type, field)                                                \
    OperatorSchemaSetter&                                                                 \
        OperatorSchemaSetter::Attr(const std::string& p_attrName,                         \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const std::vector<type>& p_defaultValue)                                      \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        for (const auto& v : p_defaultValue)                                              \
        {                                                                                 \
            a.add_##field(v);                                                             \
        }                                                                                 \
                                                                                          \
        m_opSchema.m_opSignature.m_attributes.push_back(                                  \
        OpSignature::Attribute(p_attrName,                                                \
            p_attrType,                                                                   \
            p_description,                                                                \
            a));                                                                          \
        return *this;                                                                     \
    }                                                                                     \

    ATTR_SETTER_BASIC_IMPL(int64_t, i)
        ATTR_SETTER_BASIC_IMPL(float, f)
        ATTR_SETTER_BASIC_IMPL(std::string, s)
        ATTR_SETTER_LIST_IMPL(int64_t, ints)
        ATTR_SETTER_LIST_IMPL(float, floats)
        ATTR_SETTER_LIST_IMPL(std::string, strings)

        OperatorSchemaSetter&
        OperatorSchemaSetter::TypeConstraint(const std::string& p_typeName,
            const std::vector<std::string>& p_constraints,
            const std::string& p_description)
    {
        m_constraints.push_back(std::make_tuple(p_typeName, p_constraints, p_description));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc)
    {
        m_opSchema.m_shapeInferenceFunc = p_shapeInferFunc;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetAttributeParser(
            AttributeParser p_attrParser)
    {
        m_opSchema.m_attrParser = p_attrParser;
        return *this;
    }

    OperatorSchemaSetter& OperatorSchemaSetter::FillUsing(std::function<void(OperatorSchemaSetter&)> populator)
    {
        if (populator) {
            populator(*this);
        }
        return *this;
    }

    OperatorSchemaRegistry::RegisterOnce::RegisterOnce(
        OperatorSchemaSetter& p_opSchemaSetter)
    {
        auto& opSchema = p_opSchemaSetter.m_opSchema;
        // Process type constraints.
        for (const auto& constraint : p_opSchemaSetter.m_constraints)
        {
            std::string name;
            std::vector<std::string> types;
            std::string desc;
            std::tie(name, types, desc) = constraint;

            auto it = opSchema.m_opSignature.m_typeConstraintMap.find(name);
            if (it == opSchema.m_opSignature.m_typeConstraintMap.end())
            {
                DataTypeSet d;
                for (const auto& t : types)
                {
                    d.insert(Utils::OpUtils::ToType(t));
                }
                opSchema.m_opSignature.m_typeConstraintMap.insert(std::make_pair(name, std::make_pair(d, desc)));
            }
            else
            {
                // already a constraint with the same name. error.
            }
        }

        opSchema.m_opSignature.m_inputs.reserve(p_opSchemaSetter.m_inputs.size());
        for (const auto& input : p_opSchemaSetter.m_inputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = input;
            opSchema.m_opSignature.m_inputs.push_back(
                OpSignature::FormalParameter(name, type, desc, opSchema.m_opSignature.m_typeConstraintMap));
        }

        opSchema.m_opSignature.m_outputs.reserve(p_opSchemaSetter.m_outputs.size());
        for (const auto& output : p_opSchemaSetter.m_outputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = output;
            opSchema.m_opSignature.m_outputs.push_back(
                OpSignature::FormalParameter(name, type, desc,
                    opSchema.m_opSignature.m_typeConstraintMap));
        }

        OperatorSchemaRegistry::Get()->Register(p_opSchemaSetter.m_opSchema);
    }

    bool OperatorSchemaRegistry::TryGetOp(const std::string& p_name,
        const OperatorSchema** p_opSchema) const
    {
        if (nullptr == p_opSchema)
        {
            return false;
        }

        auto iter = m_opNameToOpSchemaMap.find(p_name);
        if (m_opNameToOpSchemaMap.end() == iter)
        {
            return false;
        }
        *p_opSchema = &(iter->second);
        return true;
    }

    Status OperatorSchemaRegistry::Register(
        const OperatorSchema& p_opSchema)
    {
        auto iter = m_opNameToOpSchemaMap.find(p_opSchema.GetName());
        if (m_opNameToOpSchemaMap.end() != iter)
        {
            Status status(ONNX, FAIL,
                "Error: operator schema with same name ("
                + p_opSchema.GetName() + ") exists.");
            assert(false);
            return status;
        }
        else
        {
            m_opNameToOpSchemaMap[p_opSchema.GetName()] = p_opSchema;
            return Status::OK();
        }
    }

    OperatorSchemaRegistry* OperatorSchemaRegistry::Get()
    {
        static OperatorSchemaRegistry* s_registry
            = new OperatorSchemaRegistry();
        return s_registry;
    }

    Status TypeUtils::GetType(const AttributeProto& p_attr, AttrType& p_type)
    {
        if (!OpSignature::IsValidAttribute(p_attr))
        {
            return Status(ONNX, FAIL, "Invalid AttributeProto.");
        }

        p_type = p_attr.type();
        if (AttrType::AttributeProto_AttributeType_UNDEFINED == p_type)
        {
            if (p_attr.has_f())
            {
                p_type = AttrType::AttributeProto_AttributeType_FLOAT;
            }
            else if (p_attr.has_i())
            {
                p_type = AttrType::AttributeProto_AttributeType_INT;
            }
            else if (p_attr.has_s())
            {
                p_type = AttrType::AttributeProto_AttributeType_STRING;
            }
            else if (p_attr.has_t())
            {
                p_type = AttrType::AttributeProto_AttributeType_TENSOR;
            }
            else if (p_attr.has_g())
            {
                p_type = AttrType::AttributeProto_AttributeType_GRAPH;
            }
            else if (p_attr.floats_size())
            {
                p_type = AttrType::AttributeProto_AttributeType_FLOATS;
            }
            else if (p_attr.ints_size())
            {
                p_type = AttrType::AttributeProto_AttributeType_INTS;
            }
            else if (p_attr.strings_size())
            {
                p_type = AttrType::AttributeProto_AttributeType_STRINGS;
            }
            else if (p_attr.tensors_size())
            {
                p_type = AttrType::AttributeProto_AttributeType_TENSORS;
            }
            else if (p_attr.graphs_size())
            {
                p_type = AttrType::AttributeProto_AttributeType_GRAPHS;
            }
            else
            {
                return Status(ONNX, FAIL, "Invalid AttributeProto.");
            }
        }
        return Status::OK();
    }
}
