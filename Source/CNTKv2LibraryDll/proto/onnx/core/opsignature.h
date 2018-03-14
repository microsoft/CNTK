#ifndef CORE_GRAPH_OPSCHEMA_H
#define CORE_GRAPH_OPSCHEMA_H

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "utils.h"
#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456 4189 4996 4503)
#include "proto/onnx/protobuf/onnx-ml.pb.h"
#pragma warning(pop)

using namespace onnx;

namespace ONNXIR
{
    typedef AttributeProto_AttributeType AttrType;

    // This string array should exactly match the AttrType defined above.
    static const std::string c_attrTypeStr[14] =
    {
        "FLOAT",
        "INT",
        "STRING",
        "GRAPH",
        "TENSOR",
        "FLOATS",
        "INTS",
        "STRINGS",
        "GRAPHS",
        "TENSORS"
    };

    typedef std::unordered_set<PTYPE> DataTypeSet;
    typedef std::unordered_map<std::string, std::pair<DataTypeSet, std::string>> TypeConstraintMap;

    // Operator signature declaration.
    // It defines input formal parameter, output formal parameters and
    // attributes.
    // Once an operator signature created, it's "Read-Only".
    class OpSignature
    {
    public:

        // Formal parameter represenation, including parameter name, type.
        class FormalParameter
        {
        public:

            // Constructor.
            explicit FormalParameter(const std::string& p_name,
                const std::string& p_type,
                const std::string& p_description,
                const TypeConstraintMap& p_constraintMap = TypeConstraintMap());

            // Get formal parameter name.
            const std::string& GetName() const;

            // Get supported data types.
            const DataTypeSet& GetTypes() const;

            // Get formal parameter type string.
            const std::string& GetTypeStr() const;

            // Get formal parameter description.
            const std::string& GetDescription() const;

        private:

            FormalParameter() {}

            // Formal parameter name.
            std::string m_name;

            // A set of data types supported for <*this> formal parameter.
            // It should contain at least one element if this formal parameter
            // is good.
            DataTypeSet m_types;

            // The <parameter type> string specified when registring an op.
            // It could be a supported data type or a type constraint key, which
            // maps to a set of supported data types.
            std::string m_typeStr;

            // Formal parameter description
            std::string m_description;

        };

        // Attribute representation, including name, type, and allowed values.
        // The first element of allowed values (if specified) is the default
        // value.
        class Attribute
        {
        public:

            // Constructor.
            explicit Attribute(const std::string& p_attrName,
                AttrType p_type,
                const std::string& p_description);

            // Constructor with default value.
            explicit Attribute(const std::string& p_attrName,
                AttrType p_type,
                const std::string& p_description,
                const AttributeProto& p_defaultVal);

            // Get attribute name.
            const std::string& GetName() const;

            // Get attribute type.
            AttrType GetType() const;

            // Get to know whether this attribute has default value,
            // if yes, <p_value> will be assigned to be the default value.
            bool HasDefaultValue(const AttributeProto** p_value) const;

        private:

            Attribute() {}

            // Attribute name.
            std::string m_name;

            // Attribute type.
            AttrType m_type;

            // Attribute description.
            std::string m_description;

            // Flag indicates whether a default value specified.
            // It it's true, the first element of <m_allowedValues> is the
            // default value.
            bool m_hasDefaultValue;

            // Allowed attribute values.
            std::vector<AttributeProto> m_allowedValues;
        };

        static bool IsValidAttribute(const AttributeProto& p_attribute);

        // Constructor.
        OpSignature() = default;

        // Get operator name.
        const std::string& GetName() const;
        int SinceVersion() const;
        const std::string& Domain() const;

        // Get operator description.
        const std::string& GetDescription() const;

        // Get input formal parameters.
        const std::vector<FormalParameter>& GetInputs() const;

        // Get output formal parameters.
        const std::vector<FormalParameter>& GetOutputs() const;

        // Get attributes.
        const std::vector<Attribute>& GetAttributes() const;

        // Get type constraint map.
        const TypeConstraintMap& GetTypeConstraintMap() const;

    private:

        friend class OperatorSchemaSetter;
        friend class OpSchemaRegistry;

        // Operator name.
        std::string m_name;
        int m_sinceVersion = 1;
        std::string m_domain = "";

        // Operator description.
        std::string m_description;

        // Operator input formal parameters.
        std::vector<FormalParameter> m_inputs;

        // Operator output formal parameters.
        std::vector<FormalParameter> m_outputs;

        // Operator attributes' definitions.
        std::vector<Attribute> m_attributes;

        // Map from constraint name to DataTypeSet
        TypeConstraintMap m_typeConstraintMap;
    };
}
#endif
