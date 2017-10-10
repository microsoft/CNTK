#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "opsignature.h"
#include "shape_inference.h"

namespace ONNXIR
{
    class OpSignature;
    class OperatorSchemaSetter;
    typedef OperatorSchemaSetter OpSchema;

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

        OperatorSchemaSetter& Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type = "");

        OperatorSchemaSetter& Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type = "");

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

        OperatorSchemaSetter& TypeConstraint(const std::string& p_typeName,
            const std::vector<std::string>& p_constraints,
            const std::string& p_description);

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorSchemaSetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorSchemaSetter& SetAttributeParser(
            AttributeParser p_attrParser);

        enum class SupportType {
            COMMON,
            EXPERIMENTAL,
        };
        // Methods added for compatibility with ONNX OpSchema registration API
        OpSchema& NumInputs(int n)
        {
            return NumInputs(n, n);
        }
        OpSchema& NumInputs(int min, int max)
        {
            m_opSchema.m_opSignature.m_onnxMinInput = min;
            m_opSchema.m_opSignature.m_onnxMaxInput = max;
            return *this;
        }
        OpSchema& NumInputs(std::set<int> allowed_input_nums)
        {
            return NumInputs([allowed_input_nums](int n)-> bool {
                return allowed_input_nums.count(n) > 0;
            });
        }
        OpSchema& NumInputs(std::function<bool(int)> func)
        {
            m_opSchema.m_opSignature.m_onnxNumInputsAllowed = func;
            return *this;
        }
        OpSchema& NumOutputs(int n) {
            return NumOutputs(n, n);
        }
        OpSchema& NumOutputs(int min, int max)
        {
            m_opSchema.m_opSignature.m_onnxMinOutput = min;
            m_opSchema.m_opSignature.m_onnxMaxOutput = max;
            return *this;
        }
        OpSchema& NumOutputs(std::set<int> allowed_output_nums)
        {
            return NumOutputs([allowed_output_nums](int n)-> bool {
                return allowed_output_nums.count(n) > 0;
            });
        }
        OpSchema& NumOutputs(std::function<bool(int)> func)
        {
            m_opSchema.m_opSignature.m_onnxNumOutputsAllowed = func;
            return *this;
        }
        OpSchema& NumInputsOutputs(std::function<bool(int, int)> func)
        {
            m_opSchema.m_opSignature.m_onnxNumInputsOutputsAllowed = func;
            return *this;
        }
        OpSchema& OutputCalculator(std::function<int(int)> calc) { return *this; }
        OpSchema& SameNumberOfOutput() { return *this; }
        OpSchema& AllowConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& AllowConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& AllowOneToOneConsumed() { return *this; }
        OpSchema& EnforceConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& EnforceConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& EnforceOneToOneConsumed() { return *this; }
        OpSchema& SetSupportLevel(SupportType) { return *this; }
        OpSchema& AllowUncheckedAttributes() { return *this; }
        OpSchema& FillUsing(std::function<void(OpSchema&)> populator)
        {
            if (populator)
            {
                populator(*this);
            }
            return *this;
        }
        OpSchema& Input(const int, const char* name, const char* description)
        {
            return Input(name, description);
        }
        OpSchema& Output(const int, const char* name, const char* description)
        {
            return Output(name, description);
        }
        OpSchema& SetDoc(const std::string& doc)
        {
            return Description(doc);
        }

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

    // utility function used by ONNX v1 op registration defs.
    size_t ReplaceAll(std::string& s, const char* from, const char* to);

#define REGISTER_OPERATOR_SCHEMA(OpName) OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, OpName)
#define OPERATOR_SCHEMA_UNIQ_HELPER(Counter, OpName) OPERATOR_SCHEMA_UNIQ(Counter, OpName)
#define OPERATOR_SCHEMA_UNIQ(Counter, OpName)                     \
    static OperatorSchemaRegistry::RegisterOnce op_##Counter  \
    = OperatorSchemaSetter().Name(#OpName)

    // Operator registration example.
    // OPERATOR_DEFINITION(Add).Description("An operator to sum two float numbers.")
    //   .Input("input_1", "docstr for input_1.", "T")
    //   .Input("input_2", "docstr for input_2.", "T")
    //   .Output("output_1", "docstr for output_1.", "T")
    //   .TypeConstraint("T", { "float16", "float32", "float64" }, "Constrain input and output types to floats.");
}

#endif