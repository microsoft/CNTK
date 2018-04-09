#pragma warning(disable : 4503)

#include "constants.h"
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

    int OperatorSchema::SinceVersion() const
    {
        return m_opSignature.SinceVersion();
    }

    const std::string& OperatorSchema::Domain() const
    {
        return m_opSignature.Domain();
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
        OperatorSchemaSetter::SinceVersion(int p_opSetVersion)
    {
        m_opSchema.m_opSignature.m_sinceVersion = p_opSetVersion;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetDomain(const std::string& p_domain)
    {
        m_opSchema.m_opSignature.m_domain = p_domain;
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

    OpSchemaRegistry::DomainToVersionRange::DomainToVersionRange()
    {
        // Increase the highest version when you make BC-breaking changes to the
        // operator schema on specific domain. Update the lowest version when it's
        // determined to remove too old version history.
        m_map[c_onnxDomain] = std::make_pair(1, 2);
        m_map[c_mlDomain] = std::make_pair(1, 1);
        m_map[c_msDomain] = std::make_pair(1, 1);
    }

    const std::unordered_map<std::string, std::pair<int, int>>&
        OpSchemaRegistry::DomainToVersionRange::Map() const
    {
        return m_map;
    }

    OpSchemaRegistry::DomainToVersionRange& OpSchemaRegistry::DomainToVersionRange::Instance()
    {
        static DomainToVersionRange domain_to_version_range;
        return domain_to_version_range;
    }

    OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterOnce(OperatorSchemaSetter& p_opSchemaSetter)
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
            assert(it == opSchema.m_opSignature.m_typeConstraintMap.end());
            DataTypeSet d;
            for (const auto& t : types)
            {
                d.insert(Utils::OpUtils::ToType(t));
            }
            opSchema.m_opSignature.m_typeConstraintMap.insert(std::make_pair(name, std::make_pair(d, desc)));
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

        auto& m = map();
        auto& op_name = p_opSchemaSetter.m_opSchema.GetName();
        auto& op_domain = p_opSchemaSetter.m_opSchema.Domain();
        auto ver = p_opSchemaSetter.m_opSchema.SinceVersion();
        assert(m[op_name][op_domain].count(ver) == 0);
        m[op_name][op_domain].emplace(std::make_pair(ver, p_opSchemaSetter.m_opSchema));
    }

    const OperatorSchema* OpSchemaRegistry::Schema(
        const std::string& p_key,
        const std::string& p_domain)
    {
        auto& m = map();
        if (m.count(p_key) && m[p_key].count(p_domain))
        {
            return &m[p_key][p_domain].rbegin()->second;
        }
        else {
            return nullptr;
        }
    }

    const OperatorSchema* OpSchemaRegistry::Schema(
        const std::string& p_key,
        const int p_maxInclusiveVersion,
        const std::string& p_domain)
    {
        auto& m = map();
        if (m.count(p_key) && m[p_key].count(p_domain))
        {
            auto pos = m[p_key][p_domain].lower_bound(p_maxInclusiveVersion);
            if (m[p_key][p_domain].begin() == pos && pos->first > p_maxInclusiveVersion)
            {
                // All versions are greater than specified version.
                return nullptr;
            }

            if (m[p_key][p_domain].end() == pos || pos->first > p_maxInclusiveVersion)
            {
                // All versions are less than specified version, or,
                // The <pos> version is greater than specified version.
                pos--;
                return &(pos->second);
            }
            // Schema with exact version as specified one exists.
            return &(pos->second);
        }
        else {
            return nullptr;
        }
    }

    OpSchemaMap& OpSchemaRegistry::map()
    {
        static OpSchemaMap map;
        return map;
    }
}
