#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "opsignature.h"
#include "shape_inference.h"

namespace ONNXIR
{
    class OpSignature;
    typedef std::unordered_map<std::string, AttributeProto> NodeAttributes;

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
        int SinceVersion() const;
        const std::string& Domain() const;

        const OpSignature& GetOpSignature() const;
        ShapeInferenceFunc GetShapeInferenceFn() const;
        AttributeParser GetAttributeParser() const;

    private:

        friend class OperatorSchemaSetter;
        friend class OpSchemaRegistry;

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

        OperatorSchemaSetter& SinceVersion(int p_opSetVersion);

        OperatorSchemaSetter& SetDomain(const std::string& p_domain);

        OperatorSchemaSetter& Description(const std::string& p_description);

        // Grammar for type strings used in Input(), Output(), AttrWithRichType(), and TypeConstraint() api's
        // <type> ::= <data_type> |
        //            tensor(<data_type>) |
        //            seq(<type>) |
        //            map(<data_type>, <type>)
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
        friend class OpSchemaRegistry;

        OperatorSchema m_opSchema;

        // Operator input formal parameters.
        std::vector<InputOutputParam> m_inputs;

        // Operator output formal parameters.
        std::vector<InputOutputParam> m_outputs;

        // Operator type constraints.
        std::vector<TypeConstraintParam> m_constraints;
    };

    // Map type to store operator schemas. The format is,
    // <OpName, <Domain, <OperatorSetVersion, OpSchema>>>.
    typedef std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::map<int, OperatorSchema>>>
        OpSchemaMap;

    class OpSchemaRegistry
    {
    public:
        class DomainToVersionRange
        {
        public:
            DomainToVersionRange();

            const std::unordered_map<std::string, std::pair<int, int>>& Map() const;

            static DomainToVersionRange& Instance();

        private:

            // Key: domain. Value: <lowest version, highest version> pair.
            std::unordered_map<std::string, std::pair<int, int>> m_map;
        };

        class OpSchemaRegisterOnce
        {
        public:

            OpSchemaRegisterOnce(OperatorSchemaSetter& p_opSchemaSetter);
        };

        // Return the latest schema for an operator in specified domain.
        // Domain with default value "" means ONNX.
        static const OperatorSchema* Schema(
            const std::string& p_key,
            const std::string& p_domain = "");

        // Return the schema with biggest version, which is not greater than specified
        // <maxInclusiveVersion> in specified domain. Domain with default value "" means ONNX.
        static const OperatorSchema* Schema(
            const std::string& p_key,
            const int p_maxInclusiveVersion,
            const std::string& p_domain = "");

    private:

        // OpSchemaRegistry should not need to be instantiated.
        OpSchemaRegistry() = delete;

        /**
        * @brief Returns the underlying string to OpSchema map.
        *
        * You should not manually manipulate the map object returned. Instead, use
        * the macros defined such as OPERATOR_SCHEMA to register your operator
        * schema.
        *
        * We wrap it inside a function to avoid the statia initialization order
        * fiasco.
        */
        static OpSchemaMap& map();
    };




#define REGISTER_OPERATOR_SCHEMA(OpName) OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, OpName)
#define OPERATOR_SCHEMA_UNIQ_HELPER(Counter, OpName) OPERATOR_SCHEMA_UNIQ(Counter, OpName)
#define OPERATOR_SCHEMA_UNIQ(Counter, OpName)                     \
    static OpSchemaRegistry::OpSchemaRegisterOnce op_##Counter  \
    = OperatorSchemaSetter().Name(#OpName)

    // Operator registration example.
    // REGISTER_OPERATOR_SCHEMA(Add).Description("An operator to sum two float numbers.")
    //   .Input("input_1", "docstr for input_1.", "T")
    //   .Input("input_2", "docstr for input_2.", "T")
    //   .Output("output_1", "docstr for output_1.", "T")
    //   .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.");
}
#endif
