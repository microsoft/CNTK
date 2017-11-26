#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "opsignature.h"
#include "shape_inference.h"

namespace ONNXIR
{
    class OpSignature;

    class TypeUtils
    {
    public:

        // Get attribute type given attribute proto data.
        static Status GetType(const AttributeProto& p_attr, AttrType& p_type);

    };

    // An attribute parser - it's specified when registering an operator.
    // The parser is designed and used in two ways.
    // 1) It will be used to verify whether a Node's attributes match the
    //    operator's definition.
    // 2) It will be used to parse a Node's attributes into a <T> object,
    //    which makes it be easier to access node attributes.
    // TODO: to implement the 2nd point above, NodeAttributes should be changed
    // to contain a <T> field, which is structured attributes.
    typedef std::function<Status(const NodeAttributes&)> AttributeParser;

    class OperatorSchema
    {
    public:

        const std::string& GetName() const;
        const OpSignature& GetOpSignature() const;
        ShapeInferenceFunc GetShapeInferenceFn() const;
        AttributeParser GetAttributeParser() const;

    private:

        friend class OperatorSchemaSetter;
        friend class OperatorSchemaRegistry;

        OpSignature m_opSignature;
        ShapeInferenceFunc m_shapeInferenceFunc;
        AttributeParser m_attrParser;
    };

    typedef std::tuple<std::string, std::string, std::string> InputOutputParam;
    typedef std::tuple<std::string, std::string, AttrType, AttributeProto> AttrParam;
    typedef std::tuple<std::string, std::vector<std::string>, std::string> TypeConstraintParam;

#define ATTR_SETTER_INTERFACE(TypeName) \
    OperatorSchemaSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const TypeName& p_defaultValue); \
    OperatorSchemaSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const std::vector<TypeName>& p_defaultValues); \

    // Operator registry setter helper.
    // This is used in "OPERATOR_DEFINITION" macro, to separate setters from getters
    // in OpSignature.
    class OperatorSchemaSetter
    {
    public:

        OperatorSchemaSetter() = default;

        OperatorSchemaSetter& Name(const std::string& p_opName);

        OperatorSchemaSetter& Description(const std::string& p_description);

        // Grammar for type strings used in Input(), Output(), AttrWithRichType(), and TypeConstraint() api's
        // <type> ::= <data_type> |
        //            tensor(<data_type>) |
        //            sparse(<data_type>) |
        //            seq(<type>) |
        //            map(<data_type>, <type>) |
        //            record(<name_type_list>) |
        //            union(<name_type_list>)
        // <name_type_list> :: = <name>:<type>{ ,<name_type_list> }
        // <data_type> :: = float | uint8 | ...   (see data_type strings defined in constants.h)
        OperatorSchemaSetter& Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type = "",
            bool p_optional = false);

        OperatorSchemaSetter& Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type = ""); // see grammar above.

        OperatorSchemaSetter& TypeConstraint(const std::string& p_typeName,
            const std::vector<std::string>& p_constraints, // see grammar above.
            const std::string& p_description);

        OperatorSchemaSetter& Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool required = false);

        ATTR_SETTER_INTERFACE(int64_t)
        ATTR_SETTER_INTERFACE(float)
        ATTR_SETTER_INTERFACE(std::string)
        ATTR_SETTER_INTERFACE(TensorProto)
        ATTR_SETTER_INTERFACE(GraphProto)
        ATTR_SETTER_INTERFACE(TypeProto)
        ATTR_SETTER_INTERFACE(TypeProto::TensorShapeProto)

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorSchemaSetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorSchemaSetter& SetAttributeParser(
            AttributeParser p_attrParser);

        // adding docs for temlated/macro ops.
        OperatorSchemaSetter& FillUsing(std::function<void(OperatorSchemaSetter&)> populator);

    private:

        //friend class OpSignature;
        friend class OperatorSchemaRegistry;

        OperatorSchema m_opSchema;

        // Operator input formal parameters.
        std::vector<InputOutputParam> m_inputs;

        // Operator output formal parameters.
        std::vector<InputOutputParam> m_outputs;

        // Operator type constraints.
        std::vector<TypeConstraintParam> m_constraints;
    };

    // Operator schema registry. A singleton registry to manage all operator
    // schemas.
    class OperatorSchemaRegistry
    {
    public:

        // Helper function providing a way to call
        // OpSignatureFactory::Register().
        class RegisterOnce
        {
        public:

            RegisterOnce(OperatorSchemaSetter& p_opRegistry);
        };

        // Try to get operator with specified operator name.
        bool TryGetOp(const std::string& p_name,
            const OperatorSchema** p_opRegistry) const;

        // Register an operator.
        Status Register(const OperatorSchema& p_opSchema);

        // Get the global operator registry factory instance.
        static OperatorSchemaRegistry* Get();

    private:

        OperatorSchemaRegistry() = default;

        // An operator name to operator definition data map.
        std::unordered_map<std::string, OperatorSchema> m_opNameToOpSchemaMap;
    };

#define REGISTER_OPERATOR_SCHEMA(OpName) OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, OpName)
#define OPERATOR_SCHEMA_UNIQ_HELPER(Counter, OpName) OPERATOR_SCHEMA_UNIQ(Counter, OpName)
#define OPERATOR_SCHEMA_UNIQ(Counter, OpName)                     \
    static OperatorSchemaRegistry::RegisterOnce op_##Counter  \
    = OperatorSchemaSetter().Name(#OpName)

    // Operator registration example.
    // REGISTER_OPERATOR_SCHEMA(Add).Description("An operator to sum two float numbers.")
    //   .Input("input_1", "docstr for input_1.", "T")
    //   .Input("input_2", "docstr for input_2.", "T")
    //   .Output("output_1", "docstr for output_1.", "T")
    //   .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.");
}

#endif
