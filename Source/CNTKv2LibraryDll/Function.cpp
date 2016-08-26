//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Function.h"
#include "ComputationNetworkBuilder.h"
#include "Utils.h"
#include "ComputationNode.h"
#include "ReshapingNodes.h"
#include "EvaluationNodes.h"
#include "TrainingNodes.h"
#include "LinearAlgebraNodes.h"
#include "InputAndParamNodes.h"

using namespace Microsoft::MSR::CNTK;

bool g_shareNodeValueMatrices = true;

namespace CNTK
{
    std::shared_ptr<std::vector<Variable>> Function::InputsImpl() const
    {
        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        std::vector<Variable> inputs;
        if (compositeFunction == nullptr)
            inputs = m_inputs;
        else
            inputs = compositeFunction->DetermineInputs();

        return std::shared_ptr<std::vector<Variable>>(new std::vector<Variable>(std::move(inputs)), [](std::vector<Variable>* ptr) { delete ptr; });
    }

    FunctionPtr Function::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements)
    {
        // Cannot be called on primitive functions
        if (RootFunction() == nullptr)
            InvalidArgument("ReplacePlaceholders should never be called on primitive functions");

        std::unordered_set<const Function*> visitedFunctions;
        std::unordered_set<Placeholder> replacedPlaceholders;
        ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        for (auto replacementPair : placeholderReplacements)
        {
            if (replacedPlaceholders.find(replacementPair.first) == replacedPlaceholders.end())
                InvalidArgument("At least one of the placeholders specified for replacement was not found in the function");
        }

        return this->shared_from_this();
    }

    // Placeholders can be replaced incrementally - i.e. not all placeholders need to replaced in one go.
    // The only requirement is that they must all be replaced before making any 'Forward' calls on the Function instance.
    /*virtual*/ void Function::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                                   std::unordered_set<const Function*>& visitedFunctions,
                                                   std::unordered_set<Placeholder>& replacedPlaceholders)
    {
        visitedFunctions.insert(this);

        for (auto& inputVar : m_inputs)
        {
            if (inputVar.IsPlaceholder())
            {
                Placeholder placeholder(inputVar);
                if (placeholderReplacements.find(placeholder) != placeholderReplacements.end())
                {
                    inputVar = placeholderReplacements.at(placeholder);
                    replacedPlaceholders.insert(placeholder);
                }
            }
            else if (inputVar.IsOutput() && (visitedFunctions.find(inputVar.Owner().get()) == visitedFunctions.end()))
                inputVar.Owner()->ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);
        }
    }

    // Names for the reduction operations as used by the CNTK ReduceElementsNode
    /*static*/ const std::wstring PrimitiveFunction::InternalSumReductionOpName = L"Sum";
    /*static*/ const std::wstring PrimitiveFunction::InternalLogSumReductionOpName = L"LogSum";
    /*static*/ const std::wstring PrimitiveFunction::InternalMeanReductionOpName = L"Mean";
    /*static*/ const std::wstring PrimitiveFunction::InternalMaxReductionOpName = L"Max";
    /*static*/ const std::wstring PrimitiveFunction::InternalMinReductionOpName = L"Min";
    /*static*/ const std::wstring PrimitiveFunction::InternalAllReductionOpName = L"All";
    /*static*/ const std::wstring PrimitiveFunction::InternalAnyReductionOpName = L"Any";

    // Names of the various attributes of CNTK primitive Functions
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis = L"axis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis1 = L"axis1";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis2 = L"axis2";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDropoutRate = L"dropoutRate";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewShape = L"newShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOutputRank = L"outputRank";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOffset = L"offset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameStrides = L"strides";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSharing = L"sharing";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAutoPadding = L"autoPadding";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameLowerPad = L"lowerPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUpperPad = L"upperPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTranspose = L"transpose";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples = L"maxTempMemSizeInSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingType = L"poolingType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingWindowShape = L"poolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSpatial = L"spatial";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNormalizationTimeConstant = L"normalizationTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlendTimeConstant = L"blendTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEpsilon = L"epsilon";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUseCuDNNEngine = L"useCuDNNEngine";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewDynamicAxes = L"newDynamicAxes";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginIndex = L"beginIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndIndex = L"endIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameReductionOpName = L"reductionOpName";

    /*static*/ std::vector<Variable> PrimitiveFunction::GetOutputVariables(PrimitiveOpType op, const std::vector<Variable>& inputs, Function* owner, const Dictionary& functionConfig)
    {
        std::vector<Variable> outputs;

        // TODO: We are just using the input[0]'s DataType as output node's DataType. This is not always correct
        DataType outputDataType = DataType::Unknown;
        size_t i = 0;
        while ((outputDataType == DataType::Unknown) && (i < inputs.size()))
            outputDataType = inputs[i++].GetDataType();

        if (outputDataType == DataType::Unknown)
            InvalidArgument("The DataType of all the input operands of primitive function named %S with op type %s are unknown", owner->Name().c_str(), PrimitiveOpTypeName(op));

        // We currently require that the inputs' dynamic axes if any match
        std::vector<Axis> outputDynamicAxes;
        if (op == PrimitiveOpType::Where)
            ;
        else if ((op == PrimitiveOpType::PackedIndex) || (op == PrimitiveOpType::GatherPacked))
        {
            outputDynamicAxes = inputs[1].DynamicAxes();
        }
        else
        {
            outputDynamicAxes = inputs[0].DynamicAxes();
            for (auto inputVar : inputs)
            {
                auto currentInputDynamicAxes = inputVar.DynamicAxes();
                if (outputDynamicAxes.empty())
                    outputDynamicAxes = currentInputDynamicAxes;
                else
                {
                    if (!currentInputDynamicAxes.empty() && (currentInputDynamicAxes != outputDynamicAxes))
                        LogicError("Currently if an operand of a elementwise operation has any dynamic axes, those must match the dynamic axes of the other operands");
                }
            }
        }

        switch (op)
        {
        case PrimitiveOpType::Negate:
        case PrimitiveOpType::Sigmoid:
        case PrimitiveOpType::Tanh:
        case PrimitiveOpType::ReLU:
        case PrimitiveOpType::Exp:
        case PrimitiveOpType::Log:
        case PrimitiveOpType::Sqrt:
        case PrimitiveOpType::Floor:
        case PrimitiveOpType::Abs:
        case PrimitiveOpType::Reciprocal:
        case PrimitiveOpType::Softmax:
        case PrimitiveOpType::Hardmax:
            assert(inputs.size() == 1);
            if (((op == PrimitiveOpType::Softmax) || (op == PrimitiveOpType::Hardmax)) && (inputs[0].Shape().NumAxes() > 1))
                LogicError("Softmax/Hardmax operation can currently only be applied to a 1D input");

            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::TransposeAxes:
        {
            assert(inputs.size() == 1);
            auto axis1 = functionConfig[PrimitiveFunction::AttributeNameAxis1].Value<Axis>();
            auto axis2 = functionConfig[PrimitiveFunction::AttributeNameAxis2].Value<Axis>();

            if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                LogicError("TransposeAxes operation currently does not support transposing dynamic axes");

            auto transposedTensorShape = AsTensorShape(inputs[0].Shape(), true);
            transposedTensorShape.SwapDimsInPlace(axis1.StaticAxisIndex(), axis2.StaticAxisIndex());
            outputs.push_back(Variable(AsNDShape(transposedTensorShape), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::Where:
        {
            assert(inputs.size() == 1);
            std::vector<Axis> newDynamicAxes;
            auto newDynamicAxesNames = AsBasicElementTypeVector<std::wstring>(functionConfig[PrimitiveFunction::AttributeNameNewDynamicAxes].Value<std::vector<DictionaryValue>>());
            for (auto axisName : newDynamicAxesNames)
                newDynamicAxes.push_back(Axis(axisName));

            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, newDynamicAxes));
            break;
        }
        case PrimitiveOpType::Slice:
        {
            auto axis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            int beginIndex = functionConfig[PrimitiveFunction::AttributeNameBeginIndex].Value<size_t>();
            int endIndex = functionConfig[PrimitiveFunction::AttributeNameEndIndex].Value<size_t>();
            if (!axis.IsStaticAxis())
                LogicError("Built-in Slice operation currently does not support slicing along dynamic axis");

            if (axis.StaticAxisIndex() >= inputs[0].Shape().NumAxes())
                InvalidArgument("The specified axis index (%d) for the Slice operation is outside the bounds of the available axes of the input", (int)axis.StaticAxisIndex());

            size_t sliceAxisDim = inputs[0].Shape()[axis.StaticAxisIndex()];
            int realBeginIndex = (beginIndex >= 0) ? beginIndex : beginIndex + sliceAxisDim;
            int realEndIndex = (endIndex > 0) ? endIndex : endIndex + sliceAxisDim;
            if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                RuntimeError("Slice operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input ([%S]).",
                beginIndex,
                endIndex,
                realBeginIndex,
                realEndIndex,
                inputs[0].Shape().AsString().c_str());

            auto outputTensorShape = AsTensorShape(inputs[0].Shape(), true);

            // propagate as much as we can
            if ((axis.StaticAxisIndex() < outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim))
                outputTensorShape.NarrowTo(axis.StaticAxisIndex(), realBeginIndex, realEndIndex);

            outputs.push_back(Variable(AsNDShape(outputTensorShape), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::Dropout:
            assert(inputs.size() == 1);
            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::Reshape:
        {
            auto newShape = functionConfig[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();
            outputs.push_back(Variable(ReshapeOutputShape(inputs[0].Shape(), newShape), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::Pooling:
        {
            assert(inputs.size() == 1);
            auto poolingWindowsShape = functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            outputs.push_back(Variable(ConvolutionOpOutputShape(inputs[0].Shape(), poolingWindowsShape, { 1 }, strides, { true }, autoPadding, lowerPad, upperPad, false), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::SumAll:
            assert(inputs.size() == 1);
            outputs.push_back(Variable({}, outputDataType, owner, std::vector<Axis>({})));
            break;
        case PrimitiveOpType::Plus:
        case PrimitiveOpType::Minus:
        case PrimitiveOpType::ElementTimes:
        case PrimitiveOpType::Equal:
        case PrimitiveOpType::NotEqual:
        case PrimitiveOpType::Less:
        case PrimitiveOpType::LessEqual:
        case PrimitiveOpType::Greater:
        case PrimitiveOpType::GreaterEqual:
            assert(inputs.size() == 2);
            outputs.push_back(Variable(BinaryElementwiseOpOutputShape(op, inputs[0].Shape(), inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::Times:
        {
            assert(inputs.size() == 2);
            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            outputs.push_back(Variable(TimesOpOutputShape(inputs[0].Shape(), inputs[1].Shape(), outputRank), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::TransposeTimes:
        {
            assert(inputs.size() == 2);

            auto numLeftOperandAxes = inputs[0].Shape().NumAxes();
            if (numLeftOperandAxes > 2)
                LogicError("TransposeTimes operation currently only supports left operands of rank 1 or 2");

            NDShape transposedLeftOperandShape(2, 1);
            for (size_t i = 0; i < numLeftOperandAxes; ++i)
                transposedLeftOperandShape[transposedLeftOperandShape.NumAxes() - i - 1] = inputs[0].Shape()[i];

            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            outputs.push_back(Variable(TimesOpOutputShape(transposedLeftOperandShape, inputs[1].Shape(), outputRank), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::Convolution:
        {
            assert(inputs.size() == 2);
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto sharing = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
            auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            bool transpose = functionConfig[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
            if (inputs[0].Shape().NumAxes() < inputs[1].Shape().NumAxes())
                InvalidArgument("The convolution map should have at least as many axes as the shape of the input it operates on!");

            NDShape outputMapCount, kernelShape;
            std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(inputs[0].Shape(), inputs[1].Shape());
            outputs.push_back(Variable(ConvolutionOpOutputShape(inputs[1].Shape(), kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::SquaredError:
        case PrimitiveOpType::CrossEntropyWithSoftmax:
        case PrimitiveOpType::ClassificationError:
        {
            assert(inputs.size() == 2);

            if ((inputs[0].Shape().NumAxes() > 2) || ((inputs[0].Shape().NumAxes() > 1) && (inputs[0].Shape()[1] != 1)))
                InvalidArgument("The shape of input operands for the %s operation should have at most one axis", PrimitiveOpTypeName(op));

            auto predictionShape = inputs[0].Shape();
            auto labelsShape = inputs[1].Shape();
            if (predictionShape != labelsShape)
                RuntimeError("Prediction output operand's shape %s is incompatible with label operand's shape %s for the %s operation", AsString(predictionShape).c_str(), AsString(labelsShape).c_str(), PrimitiveOpTypeName(op));

            std::vector<size_t> reductionAxes;
            for (size_t i = 0; i < inputs[0].Shape().NumAxes(); ++i)
                reductionAxes.push_back(i);

            outputs.push_back(Variable(ReductionOpOutputShape(op, predictionShape, reductionAxes), outputDataType, owner, std::vector<Axis>({})));
            break;
        }
        case PrimitiveOpType::PastValue:
        case PrimitiveOpType::FutureValue:
        {
            assert(inputs.size() == 2);
            Variable inputOperandVar = inputs[0];
            Variable initialStateVar = inputs[1];
            // TODO: Current we only support a scalar initial state
            if (!initialStateVar.IsConstant() || (initialStateVar.Shape().NumAxes() > 0))
                LogicError("Currently PastValue/FutureValue Function only supports scalar initial state");

            // TODO: We currently only support input operand with 1 static axis for PastValue/FutureValue
            if (inputOperandVar.Shape().NumAxes() > 1)
                LogicError("Currently PastValue/FutureValue Function only supports input operand with <= 1 static axis");

            // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
            if (inputOperandVar.DynamicAxes().size() != 2)
                LogicError("Currently PastValue/FutureValue Function only supports input operand with with 2 dynamic axis (1 sequence-axis and 1 batch-axis)");

            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::ReduceElements:
        {
            assert(inputs.size() == 1);
            auto reductionAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            std::vector<size_t> reductionAxes = { reductionAxis.StaticAxisIndex() };

            outputs.push_back(Variable(ReductionOpOutputShape(op, inputs[0].Shape(), reductionAxes), outputDataType, owner, inputs[0].DynamicAxes()));
            break;
        }
        case PrimitiveOpType::BatchNormalization:
            assert(inputs.size() == 5);
            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::Combine:
            outputs = inputs;
            break;
        case PrimitiveOpType::PackedIndex:
            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::GatherPacked:
        {
            bool sourceHasDynamicAxis = !inputs[0].DynamicAxes().empty();
            NDShape outputShape;

            // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
            if (sourceHasDynamicAxis)
                outputShape = inputs[0].Shape();
            else
            {
                if (inputs[0].Shape().NumAxes() > 1)
                    outputShape = outputShape.SubShape(0, outputShape.NumAxes() - 1);
                else
                    outputShape = {};
            }

            outputs.push_back(Variable(outputShape, outputDataType, owner, outputDynamicAxes));
            break;
        }
        case PrimitiveOpType::Clip:
            assert(inputs.size() == 3);
            outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
            break;
        case PrimitiveOpType::Splice:
        {
            assert(inputs.size() >= 2);
            Axis spliceAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            outputs.push_back(Variable(SpliceOutputShape(inputs, spliceAxis.StaticAxisIndex()), outputDataType, owner, outputDynamicAxes));
            break;
        }
        default:
            LogicError("Specified op %s not yet supported", PrimitiveOpTypeName(op));
            break;
        }

        return outputs;
    }

    // Names of the dynamic axes in the CNTK engine for some special sets of dynamic axes values
    // Note: The no sequence axis corresponds to a special case where there is no sequence axis (i.e. has been reduced over)
    // and the special name is used to identify this when loading back a model saved in CNTK v1 format. This will not really be needed
    // when the new CNTK v2 model serialization format is ready.
    /*static*/ const std::wstring CompositeFunction::InternalDefaultDynamicAxisName = L"";
    /*static*/ const std::wstring CompositeFunction::InternalNoSequenceAxisName = L"noSequenceAxis";

    // Replace any PlaceHolder Variables in the graph of Functions underlying 'this' CompositeFunction. All PlaceHolder variables
    // should have been replaced before performing any Forward compute of 'this' Function.
    /*virtual*/ void CompositeFunction::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                                            std::unordered_set<const Function*>& visitedFunctions,
                                                            std::unordered_set<Placeholder>& replacedPlaceholders)
    {
        RootFunction()->ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        // If any of the placeholders were replaced with Output variables, let's add the graph of function underneath each of those to 'm_allPrimitiveFunctions' set
        for (auto replacedPlaceholder : replacedPlaceholders)
        {
            auto replacingVariable = placeholderReplacements.at(replacedPlaceholder);
            if (replacingVariable.IsOutput())
            {
                auto ownerFunc = replacingVariable.Owner();
                std::unordered_set<FunctionPtr> visitedFunctions;
                DetermineInputs(ownerFunc, visitedFunctions);

                // Add the newly visited functions to 'm_allPrimitiveFunctions' set
                m_allPrimitiveFunctions.insert(visitedFunctions.begin(), visitedFunctions.end());
            }
        }
    }

    // Recursively create a sub-network of ComputationNode instances corresponding to the graph of Functions 
    // underlying the specified 'variable' and return the ComputationNode instance that corresponds to the 
    // top level 'variable'
    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetNode(const Variable& variable,
                                                                 Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                 ComputationNetworkBuilder<ElementType>& builder,
                                                                 std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap,
                                                                 std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        auto iter = variableToNodeMap.find(variable);
        if (iter != variableToNodeMap.end())
            return iter->second;

        // Lets add a null entry in the map for this variable, to break infinite recursion when processing recurrent graphs
        variableToNodeMap[variable] = nullptr;

        std::shared_ptr<ComputationNode<ElementType>> computationNodePtr;
        if (variable.IsParameter() || variable.IsConstant())
        {
            computationNodePtr = builder.CreateLearnableParameter(variable.Name(), AsTensorShape(variable.Shape()));
            network->InitLearnableParameters(computationNodePtr, L"fixedValue", 0); // must call this to follow protocol; can overwrite later
            if (!variable.NeedsGradient())
                computationNodePtr->SetLearningRateMultiplier(0.0);

            NDArrayViewPtr value = variable.IsConstant() ? Constant(variable).Value() : Parameter(variable).Value();
            std::shared_ptr<const Matrix<ElementType>> valueMatrix = variable.IsConstant() ? value->GetMatrix<ElementType>() : value->GetWritableMatrix<ElementType>();
            if (variable.IsParameter() || (valueMatrix->GetDeviceId() == network->GetDeviceId()))
                computationNodePtr->Value() = valueMatrix->AsReference();
            else
            {
                Matrix<ElementType> clonedMatrix(valueMatrix->GetNumRows(), valueMatrix->GetNumCols(), network->GetDeviceId(), valueMatrix->GetMatrixType(), valueMatrix->GetFormat());
                clonedMatrix.AssignValuesOf(*valueMatrix);
                computationNodePtr->Value() = std::move(clonedMatrix);
            }
        }
        else if (variable.IsInput())
        {
            // TODO: Input variables currently are required to have the default batch axis
            auto dynamicAxes = variable.DynamicAxes();
            auto foundDefaultBatchAxis = std::find(dynamicAxes.begin(), dynamicAxes.end(), Axis::DefaultBatchAxis());
            if (foundDefaultBatchAxis == dynamicAxes.end())
                LogicError("Currently Input Variables are required to have the DefaultBatchAxis as one of their dynamic axes");

            if (dynamicAxes.back() != Axis::DefaultBatchAxis())
                LogicError("Currently Input Variables are required to have the DefaultBatchAxis as their last dynamic axes");

            // TODO: Support inputs with > 1 dynamic axes
            if ((dynamicAxes.size() < 1) || (dynamicAxes.size() > 2))
                LogicError("Currently only Input variables with 1 or 2 dynamic axis are supported");

            // Construct the dynamic axis name to be used internally for the CNTK InputNodes
            std::wstring internalDynamicAxisName = InternalDynamicAxisNameFromDynamicAxes(dynamicAxes);

            if (!internalDynamicAxisName.empty())
                network->AddNodeToNetAndAttachInputs(New<DynamicAxisNode<ElementType>>(network->GetDeviceId(), internalDynamicAxisName), {});

            if (IsSparseInput(variable))
                computationNodePtr = builder.CreateSparseInputNode(variable.Name(), AsTensorShape(variable.Shape()), internalDynamicAxisName);
            else
                computationNodePtr = builder.CreateInputNode(variable.Name(), AsTensorShape(variable.Shape()), internalDynamicAxisName);

            if (variable.NeedsGradient())
            {
                // Set a dummy learning rate multiplier to force gradient computation for the input computation node since by default
                // gradients are not computed for Input nodes
                computationNodePtr->SetLearningRateMultiplier(0.00001f);
            }
        }
        else
        {
            assert(variable.IsOutput());
            computationNodePtr = GetOutputVariableNode(variable, network, builder, variableToNodeMap, isVariableRootMap)->template As<ComputationNode<ElementType>>()->shared_from_this();
        }

        variableToNodeMap[variable] = computationNodePtr;
        isVariableRootMap[variable] = variable.IsOutput();
        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::CreateComputationNode(const Variable& variable,
                                                                               PrimitiveFunction* primitiveFunction,
                                                                               const std::vector<std::shared_ptr<ComputationNode<ElementType>>>& inputNodes,
                                                                               Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                               ComputationNetworkBuilder<ElementType>& builder,
                                                                               std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap)
    {
        ComputationNodeBasePtr computationNodePtr;

        auto functionName = primitiveFunction->Name();
        auto& functionConfig = primitiveFunction->FunctionConfig();
            auto functionInputs = primitiveFunction->Inputs();
        PrimitiveOpType op = primitiveFunction->OpType();

            switch (op)
            {
            case PrimitiveOpType::Negate:
            computationNodePtr = builder.Negate(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Sigmoid:
            computationNodePtr = builder.Sigmoid(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Tanh:
            computationNodePtr = builder.Tanh(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::ReLU:
            computationNodePtr = builder.RectifiedLinear(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Exp:
            computationNodePtr = builder.Exp(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Log:
            computationNodePtr = builder.Log(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Sqrt:
            computationNodePtr = builder.Sqrt(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Floor:
            computationNodePtr = builder.Floor(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Abs:
            computationNodePtr = builder.Abs(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Reciprocal:
            computationNodePtr = builder.Reciprocal(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Softmax:
            computationNodePtr = builder.Softmax(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Hardmax:
            computationNodePtr = builder.Hardmax(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::TransposeAxes:
            {
            auto axis1 = functionConfig[PrimitiveFunction::AttributeNameAxis1].Value<Axis>();
            auto axis2 = functionConfig[PrimitiveFunction::AttributeNameAxis2].Value<Axis>();

                // The axis ids passed to the internal CNTK TransposeDimensionsNode are 1 based instead of 0 based
            computationNodePtr = New<TransposeDimensionsNode<ElementType>>(network->GetDeviceId(), functionName, AsCNTKInternalAxisIdx(axis1), AsCNTKInternalAxisIdx(axis2));
            network->AddNodeToNetAndAttachInputs(computationNodePtr, { inputNodes[0] });
                break;
            }
            case PrimitiveOpType::Where:
            {
                auto dynamicAxes = variable.DynamicAxes();
            auto internalCNTKWhereNodeDynamicAxisName = InternalDynamicAxisNameFromDynamicAxes(dynamicAxes);
            computationNodePtr = New<WhereNode<ElementType>>(network->GetDeviceId(), functionName, internalCNTKWhereNodeDynamicAxisName);
            network->AddNodeToNetAndAttachInputs(computationNodePtr, { inputNodes[0] });
            break;
        }
        case PrimitiveOpType::Slice:
        {
            auto axis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            int beginIndex = functionConfig[PrimitiveFunction::AttributeNameBeginIndex].Value<size_t>();
            int endIndex = functionConfig[PrimitiveFunction::AttributeNameEndIndex].Value<size_t>();

            // Internal CNTK SliceNode takes 1 based axis indices instead of 0 based
            computationNodePtr = New<SliceNode<ElementType>>(network->GetDeviceId(), functionName, beginIndex, endIndex, AsCNTKInternalAxisIdx(axis));
            network->AddNodeToNetAndAttachInputs(computationNodePtr, { inputNodes[0] });
            break;
        }
        case PrimitiveOpType::Dropout:
        {
            auto dropoutRate = functionConfig[PrimitiveFunction::AttributeNameDropoutRate].Value<double>();
            computationNodePtr = builder.Dropout(inputNodes[0], functionName);
            computationNodePtr->As<DropoutNode<ElementType>>()->SetDropoutRate(dropoutRate);
            break;
        }
        case PrimitiveOpType::Reshape:
        {
            auto newShape = functionConfig[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();
            computationNodePtr = builder.Reshape(inputNodes[0], AsTensorShape(newShape, true /*preserveRank*/), functionName);
                break;
            }
            case PrimitiveOpType::Pooling:
            {
            PoolingType poolingType = (PoolingType)(functionConfig[PrimitiveFunction::AttributeNamePoolingType].Value<size_t>());
            auto poolingWindowsShape = functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            computationNodePtr = builder.Pooling(inputNodes[0], AsCNTKPoolKind(poolingType), AsTensorShape(poolingWindowsShape, true), AsTensorShape(strides, true), autoPadding, AsTensorShape(lowerPad, true), AsTensorShape(upperPad, true), ImageLayoutKind::CHW, functionName);
                break;
            }
            case PrimitiveOpType::SumAll:
            computationNodePtr = builder.Sum(inputNodes[0], functionName);
                break;
            case PrimitiveOpType::Plus:
            computationNodePtr = builder.Plus(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::Minus:
            computationNodePtr = builder.Minus(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::ElementTimes:
            computationNodePtr = builder.ElementTimes(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::Equal:
            computationNodePtr = builder.Equal(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::NotEqual:
            computationNodePtr = builder.NotEqual(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::Less:
            computationNodePtr = builder.Less(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::LessEqual:
            computationNodePtr = builder.LessEqual(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::Greater:
            computationNodePtr = builder.Greater(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::GreaterEqual:
            computationNodePtr = builder.GreaterEqual(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::Times:
            {
            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            computationNodePtr = builder.Times(inputNodes[0], inputNodes[1], outputRank, functionName);
                break;
            }
            case PrimitiveOpType::TransposeTimes:
            {
            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            computationNodePtr = network->AddNodeToNetAndAttachInputs(New<TransposeTimesNode<ElementType>>(network->GetDeviceId(), functionName, outputRank), { inputNodes[0], inputNodes[1] });
                break;
            }
            case PrimitiveOpType::Convolution:
            {
                NDShape outputMapCount, kernelShape;
                std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(functionInputs[0].Shape(), functionInputs[1].Shape());
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto sharing = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
            auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            auto transpose = functionConfig[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
            auto maxTempMemSizeInSamples = functionConfig[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples].Value<size_t>();
            computationNodePtr = builder.Convolution(inputNodes[0], inputNodes[1], AsTensorShape(kernelShape, true), AsTensorShape(outputMapCount, true), AsTensorShape(strides, true), sharing, autoPadding, AsTensorShape(lowerPad, true), AsTensorShape(upperPad, true), transpose, ImageLayoutKind::CHW, maxTempMemSizeInSamples, functionName);
                break;
            }
            case PrimitiveOpType::SquaredError:
            computationNodePtr = builder.SquareError(inputNodes[0], inputNodes[1], functionName);
                break;
            case PrimitiveOpType::CrossEntropyWithSoftmax:
            computationNodePtr = builder.CrossEntropyWithSoftmax(inputNodes[1], inputNodes[0], functionName);
                break;
            case PrimitiveOpType::ClassificationError:
            computationNodePtr = builder.ClassificationError(inputNodes[1], inputNodes[0], functionName);
                break;
            case PrimitiveOpType::PastValue:
            case PrimitiveOpType::FutureValue:
            {
            Variable inputOperandVar = functionInputs[0];
            Variable initialStateVar = functionInputs[1];

                // Get the intial state of the PastValue/FutureValue operation
                ElementType initStateValue;
                NDArrayView tempView({}, &initStateValue, 1, DeviceDescriptor::CPUDevice());
                tempView.CopyFrom(*Constant(initialStateVar).Value());

            size_t offset = primitiveFunction->FunctionConfig()[PrimitiveFunction::AttributeNameOffset].Value<size_t>();
                if (op == PrimitiveOpType::PastValue)
                computationNodePtr = builder.PastValue(inputNodes[0], (float)initStateValue, inputOperandVar.Shape().TotalSize(), offset, functionName);
                else
                computationNodePtr = builder.FutureValue(inputNodes[0], (float)initStateValue, inputOperandVar.Shape().TotalSize(), offset, functionName);

                break;
            }
            case PrimitiveOpType::ReduceElements:
            {
            auto reductionAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            auto reductionOpName = functionConfig[PrimitiveFunction::AttributeNameReductionOpName].Value<std::wstring>();
            computationNodePtr = network->AddNodeToNetAndAttachInputs(New<ReduceElementsNode<ElementType>>(network->GetDeviceId(), functionName, reductionOpName, AsCNTKInternalAxisIdx(reductionAxis)), { inputNodes[0] });
                break;
            }
            case PrimitiveOpType::BatchNormalization:
            {
            auto spatial = functionConfig[PrimitiveFunction::AttributeNameSpatial].Value<bool>();
            auto normalizationTimeConstant = functionConfig[PrimitiveFunction::AttributeNameNormalizationTimeConstant].Value<double>();
            auto blendTimeConstant = functionConfig[PrimitiveFunction::AttributeNameBlendTimeConstant].Value<double>();
            auto epsilon = functionConfig[PrimitiveFunction::AttributeNameEpsilon].Value<double>();
            auto useCuDNNEngine = functionConfig[PrimitiveFunction::AttributeNameUseCuDNNEngine].Value<bool>();
            computationNodePtr = builder.BatchNormalization(inputNodes[0], inputNodes[1], inputNodes[2], inputNodes[3], inputNodes[4], spatial, normalizationTimeConstant, blendTimeConstant, epsilon, !useCuDNNEngine, ImageLayoutKind::CHW, functionName);
                break;
            }
            case PrimitiveOpType::Combine:
                // This operation is just a no-op and is a means to combine multiple functions to create a single Function
            // whose outputs are a union of the outputs of the Functions being combined.

                computationNodePtr = variableToNodeMap[variable];

                break;
            case PrimitiveOpType::PackedIndex:
            computationNodePtr = New<PackedIndexNode<ElementType>>(network->GetDeviceId(), functionName);
            network->AddNodeToNetAndAttachInputs(computationNodePtr, { inputNodes[0], inputNodes[1] });
                break;
            case PrimitiveOpType::GatherPacked:
            computationNodePtr = New<GatherPackedNode<ElementType>>(network->GetDeviceId(), functionName);
            network->AddNodeToNetAndAttachInputs(computationNodePtr, { inputNodes[1], inputNodes[0] });
                break;
        case PrimitiveOpType::Clip:
            {
            computationNodePtr = builder.Clip(inputNodes[1], inputNodes[2], inputNodes[0], functionName);
            break;
        }
        case PrimitiveOpType::Splice:
        {
            Axis spliceAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();

            // The CNTK internal spliceAxis indices are 1 based instead of 0 based
            computationNodePtr = New<RowStackNode<ElementType>>(network->GetDeviceId(), functionName, AsCNTKInternalAxisIdx(spliceAxis));
            std::vector<ComputationNodeBasePtr> inputNodesBasePtrs;
            for (auto inputNode : inputNodes)
                inputNodesBasePtrs.push_back(inputNode);

            network->AddNodeToNetAndAttachInputs(computationNodePtr, inputNodesBasePtrs);
                break;
            }
            default:
                LogicError("Specified op %s not yet supported", PrimitiveOpTypeName(op));
                break;
            }

        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetOutputVariableNode(const Variable& variable,
                                                                               Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                               ComputationNetworkBuilder<ElementType>& builder,
                                                                               std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap,
                                                                               std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        assert(variable.IsOutput());

        Function* function = variable.Owner().get();
        ComputationNodeBasePtr computationNodePtr;
        if (dynamic_cast<PrimitiveFunction*>(function))
        {
            PrimitiveFunction* primitiveFunction = dynamic_cast<PrimitiveFunction*>(function);
            PrimitiveOpType op = primitiveFunction->OpType();
            auto functionInputs = primitiveFunction->Inputs();

            // Create the nodes corresponding to the inputs

            std::vector<std::shared_ptr<ComputationNode<ElementType>>> inputNodes;
            for (auto inputVar : functionInputs)
            {
                auto baseNodePtr = GetNode(inputVar, network, builder, variableToNodeMap, isVariableRootMap);
                inputNodes.push_back((baseNodePtr != nullptr) ? baseNodePtr->template As<ComputationNode<ElementType>>()->shared_from_this() : nullptr);
            }

            computationNodePtr = CreateComputationNode(variable, primitiveFunction, inputNodes, network, builder, variableToNodeMap);
            if (op != PrimitiveOpType::Combine)
            {
                for (auto inputVar : functionInputs)
                    isVariableRootMap[inputVar] = false;
            }
        }
        else
            LogicError("User defined Functions are currently unsupported!");

        return computationNodePtr;
    }

    template <typename ElementType>
    ComputationNetworkPtr CompositeFunction::GetComputationNetwork(const DeviceDescriptor& device, const std::unordered_set<Variable>& backpropRoots)
    {
        if (m_computationNetwork != nullptr)
        {
            // TODO: We should either invalidate and readapt the network if he backpropRoots change compared to what was specified when the network
            // was last constructed, to just recreate a new network.
            // For now just disallow changing the backpropRoots after the network is created
            if (m_currentBackpropRoots != backpropRoots)
                LogicError("Changing backprop roots across different Forward calls on a CNTK composite Function is currently unsupported");

            // TODO: Support changing the device across different invocations of the forward method on a Function instance
            if (AsDeviceDescriptor(m_computationNetwork->GetDeviceId()) != device)
                LogicError("Changing device across different Forward calls on a CNTK composite Function is currently unsupported");
        }

        if (m_computationNetwork == nullptr)
        {
            m_computationNetwork = std::make_shared<ComputationNetwork>(AsCNTKImplDeviceId(device));

            ComputationNetworkBuilder<ElementType> builder(*m_computationNetwork);

            // TODO: We current only support one backprop root
            if (backpropRoots.size() > 1)
                LogicError("More than one backprop roots is currently unsupported");

            ComputationNodeBasePtr backpropRootNode;

            // Now recursively create the network in a top-down fashion
            auto rootFunction = RootFunction();
            auto rootFunctionOutputs = rootFunction->Outputs();
            std::vector<ComputationNodeBasePtr> forwardRootNodes;
            for (auto rootOutput : rootFunctionOutputs)
            {
                auto currentRootNode = GetNode(rootOutput, m_computationNetwork, builder, m_variableToNodeMap, m_isVariableRootMap);
                forwardRootNodes.push_back(currentRootNode);

                if (backpropRoots.find(rootOutput) != backpropRoots.end())
                    backpropRootNode = m_variableToNodeMap[rootOutput];
            }

            // If any of the function outputs is not a root node, we need to explicitly add it to the 'output' group of the ComputationNetwork
            for (auto rootOutput : rootFunctionOutputs)
            {
                if (!m_isVariableRootMap[rootOutput])
                    m_computationNetwork->AddToNodeGroup(L"output", m_variableToNodeMap[rootOutput]);
            }

            m_currentBackpropRoots = backpropRoots;

            // In case of recurrence, the inputs of some of the ComputationNodes are not attached due to cycles.
            // Now attach those after we have created all ComputationNodes in the network
            for (auto varNodePair : m_variableToNodeMap)
            {
                auto currentComputationNodeInputs = varNodePair.second->GetInputs();

                // TODO: Can any node other than a non PastValue/FutureValue Function have a null input attached after the first pass is finished?
                if (std::find(currentComputationNodeInputs.begin(), currentComputationNodeInputs.end(), nullptr) != currentComputationNodeInputs.end())
                {
                    // We found a null input; this variable must correspond to a PastValue or FutureValue function
                    const PrimitiveFunction* primitiveFunc = dynamic_cast<const PrimitiveFunction*>(varNodePair.first.Owner().get());
                    if ((primitiveFunc == nullptr) || ((primitiveFunc->OpType() != PrimitiveOpType::PastValue) && (primitiveFunc->OpType() != PrimitiveOpType::FutureValue)))
                        InvalidArgument("Invalid Function graph detected; recurrence found at a Function that is not a PastValue/FutureValue function");

                    // The 1st input of the PastValue/FutureValue function denotes the recurrent input
                    auto actualInput = m_variableToNodeMap[primitiveFunc->Inputs()[0]];
                    varNodePair.second->AttachInputs({ actualInput });
                }
            }

            m_computationNetwork->CompileNetwork();

            // Verify that the shapes of the output Variables that we computed match the corresponding nodes in the ComputationNetwork
            for (auto varNodePair : m_variableToNodeMap)
            {
                if (varNodePair.first.IsOutput())
                {
                    auto outputVar = varNodePair.first;
                    auto computationNodePtr = m_variableToNodeMap[outputVar];
                    auto outputShape = outputVar.Shape();
                    auto computationNodeSampleLayout = computationNodePtr->GetSampleLayout();
                    if (((outputShape.NumAxes() == 0) && (computationNodeSampleLayout[0] != 1)) ||
                        ((outputShape.NumAxes() != 0) && (computationNodeSampleLayout != AsTensorShape(outputShape)) && (computationNodeSampleLayout != AsTensorShape(outputShape, true))))
                    {
                        LogicError("The output Variable shape %s does not match the SampleLayout shape %s of the corresponding ComputationNode in the network", AsString(outputShape).c_str(), ((std::string)computationNodeSampleLayout).c_str());
                    }
                }
            }

            m_computationNetwork->AllocateAllMatrices(forwardRootNodes, {}, backpropRootNode);
        }

        return m_computationNetwork;
    }

    template <typename ElementType>
    /*static*/ std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CompositeFunction::GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value)
    {
        if (var.GetDataType() != value->Data()->GetDataType())
            LogicError("The Variable's DataType %s does not match the corresponding Value's DataType %s", DataTypeName(var.GetDataType()), DataTypeName(value->Data()->GetDataType()));

        if (AsDataType<ElementType>() != value->Data()->GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(value->Data()->GetDataType()));

        // TODO: Is supplying dense data for an Input variable tagged as sparse, a fatal error?
        if (IsSparseInput(var) && !value->Data()->IsSparse())
            InvalidArgument("Dense input data supplied for a sparse input Variable");

        if (IsSparseInput(var) && (value->Data()->GetStorageFormat() != StorageFormat::SparseCSC))
            InvalidArgument("Sparse Input data must be in SparseCSC format");

        if (value->Data()->Shape().NumAxes() == var.Shape().NumAxes())
            return{ value->Data()->GetMatrix<ElementType>(), nullptr };

        if (value->Data()->Shape().NumAxes() < (var.Shape().NumAxes() + var.DynamicAxes().size()))
            InvalidArgument("Value's number of axes should be larger than the Variable's number of axes by number of dynamic axes");

        if (var.DynamicAxes().size() > 2)
            LogicError("More than 2 dynamic axis for a variable is currently unsupported");

        size_t maxNumTimeSteps = value->Data()->Shape()[var.Shape().NumAxes()];
        size_t numSequences = value->Data()->Shape()[var.Shape().NumAxes() + 1];

        auto mask = value->Mask();
        if ((mask != nullptr) && ((var.Shape().NumAxes() + mask->Shape().NumAxes()) != value->Data()->Shape().NumAxes()))
            InvalidArgument("Invalid Value object; the sum of the #axes of the mask and data does not equal the Variable's number of axes by 1 + number of dynamic axes");

        if ((numSequences == 1) || (maxNumTimeSteps == 1))
        {
            // The data need not be shuffled
            std::shared_ptr<const Matrix<ElementType>> matrixData = value->Data()->GetMatrix<ElementType>(var.Shape().NumAxes());
            auto layout = std::make_shared<MBLayout>();
            if (maxNumTimeSteps == 1)
                layout->InitAsFrameMode(numSequences);
            else
            {
                layout->Init(1, maxNumTimeSteps);
                layout->AddSequence(0, 0, 0, maxNumTimeSteps);
            }

            return{ matrixData , layout};
        }
        else
        {
            std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
            if (mask != nullptr)
            {
                // Determine the sequence lengths from the mask
                std::unique_ptr<char[]> maskData(mask->GetMatrix()->CopyToArray());
                for (size_t i = 0; i < numSequences; ++i)
                {
                    size_t currentSequenceLength = 0;
                    bool currentSequenceEndAlreadyFound = false;
                    for (size_t j = 0; j < maxNumTimeSteps; ++j)
                    {
                        if (maskData[(i * maxNumTimeSteps) + j] == 1)
                        {
                            if (currentSequenceEndAlreadyFound)
                                InvalidArgument("Invalid Value object; only trailing steps of a sequence can be masked");

                            currentSequenceLength++;
                        }
                        else
                        {
                            currentSequenceEndAlreadyFound = true;
                        }
                    }

                    sequenceLengths[i] = currentSequenceLength;
                }
            }

            // The data needs to be rearranged since CNTK requires sequences to be interleaved across timesteps
            std::vector<MBLayout::SequenceInfo> sequences;
            for (size_t i = 0; i < numSequences; ++i)
                sequences.push_back({ i, SIZE_MAX, 0, sequenceLengths[i]});

            auto layout = std::make_shared<MBLayout>();
            std::vector<std::pair<size_t, size_t>> placement;
            std::vector<size_t> rowAllocations;
            layout->InitAsPackedSequences(sequences, placement, rowAllocations);
            if (maxNumTimeSteps != layout->GetNumTimeSteps())
                LogicError("The number of time steps in the packed MBLayout does not match the longest sequence's length in the Value object");

            if (numSequences != layout->GetNumSequences())
                LogicError("The number of sequences in the packed MBLayout does not match the sequence count in the Value object");

            // Now generate the gather indices
            auto matrixData = std::make_shared<Matrix<ElementType>>(var.Shape().TotalSize(),
                                                                    layout->GetNumCols(),
                                                                    AsCNTKImplDeviceId(value->Data()->Device()),
                                                                    value->Data()->IsSparse() ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                    AsCNTKImplMatrixFormat(value->Data()->GetStorageFormat()));

            std::vector<size_t> sequencesShorterThanLongestSequence;
            for (size_t i = 0; i < numSequences; ++i)
                if (sequenceLengths[i] != maxNumTimeSteps)
                    sequencesShorterThanLongestSequence.push_back(i);

            // Set the source location for all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
            size_t sourceColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
            std::vector<ElementType> gatherIndicesVector(layout->GetNumCols(), (ElementType)sourceColIdxForInvalidColumns);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t targetParallelStreamIdx = placement[i].first;
                size_t targetStartIdxInParallelStream = placement[i].second;
                for (size_t j = 0; j < sequenceLengths[i]; ++j)
                    gatherIndicesVector[((targetStartIdxInParallelStream + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = (ElementType)((i * maxNumTimeSteps) + j);
            }

            auto gatherIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), gatherIndicesVector.data(), AsCNTKImplDeviceId(value->Data()->Device()));
            matrixData->DoGatherColumnsOf(0, *gatherIdxMatrix, *(value->Data()->GetMatrix<ElementType>(var.Shape().NumAxes())), 1);
            return{ matrixData, layout };
        }
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        NDShape valueDataShape = sampleShape;
        if (layout != nullptr)
            valueDataShape = valueDataShape.AppendShape({ layout->GetNumTimeSteps(), layout->GetNumSequences() });

        // No data shuffling needed if no layout or the layout has just one time-step or just one sequence
        if ((layout == nullptr) || (layout->GetNumTimeSteps() == 1) || (layout->GetNumSequences() == 1))
        {
            // Just create a view over the existing matrix itself
            auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorShape(valueDataShape));
            auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), valueDataShape, readOnly, tensorView);
            return MakeSharedObject<Value>(data);
        }

        if (layout->GetNumCols() != matrix.GetNumCols())
            LogicError("Bad MBLayout: The number of columns in the MBLayout does not match the number of columns in the data matrix!");

        size_t maxNumTimeSteps = layout->GetNumTimeSteps();
        size_t numSequences = layout->GetNumSequences();

        std::vector<size_t> sequenceLengths;
        auto& layoutSequences = layout->GetAllSequences();
        for (auto sequenceInfo : layoutSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
                sequenceLengths.push_back(sequenceInfo.GetNumTimeSteps());
        }

        // Reshuffle to data to unpack and uninterleave the CNTK form packed data
        // Now generate the scatter indices
        auto shuffledMatrixData = std::make_shared<Matrix<ElementType>>(matrix.GetNumRows(), maxNumTimeSteps * numSequences, matrix.GetDeviceId(), matrix.GetMatrixType(), matrix.GetFormat());

        std::vector<size_t> sequencesShorterThanLongestSequence;
        for (size_t i = 0; i < numSequences; ++i)
            if (sequenceLengths[i] != maxNumTimeSteps)
                sequencesShorterThanLongestSequence.push_back(i);

        // Set the target location of all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
        size_t targetColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
        std::vector<ElementType> scatterIndicesVector(layout->GetNumCols(), (ElementType)targetColIdxForInvalidColumns);
        size_t i = 0;
        for (auto sequenceInfo : layoutSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
            {
                size_t targetParallelStreamIdx = sequenceInfo.s;
                size_t targetStartIdxInParallelStream = sequenceInfo.tBegin;
                for (size_t j = 0; j < sequenceInfo.GetNumTimeSteps(); ++j)
                    scatterIndicesVector[((targetStartIdxInParallelStream + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = (ElementType)((i * maxNumTimeSteps) + j);

                i++;
            }
        }

        auto scatterIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), scatterIndicesVector.data(), matrix.GetDeviceId());
        shuffledMatrixData->DoScatterColumnsOf(0, *scatterIdxMatrix, matrix, 1);

        // Create the mask if needed
        NDMaskPtr mask;
        if (!sequencesShorterThanLongestSequence.empty())
        {
            mask = MakeSharedObject<NDMask>(NDShape({ maxNumTimeSteps, numSequences }), AsDeviceDescriptor(matrix.GetDeviceId()));
            for (auto shortSequenceIdx : sequencesShorterThanLongestSequence)
            {
                mask->MaskSection({ sequenceLengths[shortSequenceIdx], shortSequenceIdx }, { NDShape::InferredDimension, 1 });
            }
        }

        auto tensorView = new TensorView<ElementType>(shuffledMatrixData, AsTensorShape(valueDataShape));
        auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(shuffledMatrixData->GetFormat()), valueDataShape, readOnly, tensorView);
        return MakeSharedObject<Value>(data, mask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        if (var.DynamicAxes().size() > 2)
            LogicError("More than 2 dynamic axis for a variable is currently unsupported");

        if (AsDataType<ElementType>() != var.GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(var.GetDataType()));

        if ((layout != nullptr) && (matrix.GetNumRows() != var.Shape().TotalSize()))
            LogicError("Unexpected matrix layout: The number of rows in the matrix does not match the sample size of the Variable");

        return GetValueObjectFromCNTKImplMatrixAndMBLayout(var.Shape(), matrix, layout, readOnly);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeValue(const std::pair<Variable, ValuePtr>& variableValue, ComputationNodeBasePtr& computationNode)
    {
        auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableValue.first, variableValue.second);
        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;

        auto& nodeData = computationNode->As<ComputationNode<ElementType>>()->Value();

        // Switch the node matrix to the right matrix type
        nodeData.SwitchToMatrixType(CNTKMatrixAndMBLayout.first->GetMatrixType(), CNTKMatrixAndMBLayout.first->GetFormat(), false);
        nodeData.AssignValuesOf(*CNTKMatrixAndMBLayout.first);
        computationNode->GetMBLayout()->CopyFrom(layout);
    }

    void CompositeFunction::PopulateNetworkInputs(const std::unordered_map<Variable, ValuePtr>& arguments)
    {
        auto functionArguments = this->Arguments();
        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto argument : functionArguments)
        {
            // Ensure we have values for all arguments of the function
            if (arguments.find(argument) == arguments.end())
                InvalidArgument("Value not specified for required Function Argument");

            auto argumentComputationNode = m_variableToNodeMap[argument];
            inputNodes.push_back(argumentComputationNode);

            ValuePtr argumentValue = arguments.at(argument);

            MBLayoutPtr layout;
            switch (argumentValue->Data()->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeValue<float>({ argument, argumentValue }, argumentComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeValue<double>({ argument, argumentValue }, argumentComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(argumentValue->Data()->GetDataType()));
                break;
            }
        }

        m_computationNetwork->BumpEvalTimeStamp(inputNodes);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeGradient(const std::pair<Variable, ValuePtr>& variableGradient, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode)
    {
        auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableGradient.first, variableGradient.second);
        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;
        auto nodeLayout = computationNode->GetMBLayout();
        if (((layout == nullptr) != (nodeLayout == nullptr)) || ((layout != nullptr) && (*layout != *nodeLayout)))
            InvalidArgument("The layout of the specified gradient Value is incompatible with the layout of the corresponding Variable computed during Forward call");
        computationNode->As<ComputationNode<ElementType>>()->AssignGradient(*CNTKMatrixAndMBLayout.first);
    }

    // Assign the supplied gradients corresponding to the root(s) of the network to be backpropagated through the graph
    void CompositeFunction::PopulateNetworkGradients(const std::unordered_map<Variable, ValuePtr>& gradients)
    {
        auto functionOutputs = this->Outputs();
        for (auto gradientVarValuePair : gradients)
        {
            // Only gradients for roots of the function can be specified
            if (std::find(functionOutputs.begin(), functionOutputs.end(), gradientVarValuePair.first) == functionOutputs.end())
                InvalidArgument("Gradients cannot be specified for a Variable that is not an Output of the Function");

            auto outputComputationNode = m_variableToNodeMap[gradientVarValuePair.first];
            ValuePtr gradientValue = gradientVarValuePair.second;

            switch (gradientValue->Data()->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeGradient<float>(gradientVarValuePair, outputComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeGradient<double>(gradientVarValuePair, outputComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(gradientValue->Data()->GetDataType()));
                break;
            }
        }
    }

    static NDShape GetValueShape(const Variable& var, const ComputationNodeBasePtr& computationNodePtr)
    {
        size_t outputValueNumAxes = var.Shape().NumAxes();

        // Add the batch and dynamic axes if needed
        if (computationNodePtr->GetMBLayout() != nullptr)
            outputValueNumAxes += 2;

        std::vector<size_t> outputShapeDims(outputValueNumAxes);
        for (size_t i = 0; i < var.Shape().NumAxes(); ++i)
            outputShapeDims[i] = computationNodePtr->GetSampleLayout().GetDim(i);

        if (computationNodePtr->GetMBLayout() != nullptr)
        {
            outputShapeDims[var.Shape().NumAxes()] = computationNodePtr->GetMBLayout()->GetNumTimeSteps();
            outputShapeDims[var.Shape().NumAxes() + 1] = computationNodePtr->GetMBLayout()->GetNumSequences();
        }

        return NDShape(outputShapeDims);
    }

    /*static*/ void CompositeFunction::GetNodeOutputOrGradient(Variable var, ValuePtr& varValue, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode, bool getGradient)
    {
        auto valueShape = GetValueShape(var, computationNode);
        if (varValue != nullptr)
        {
            // TODO: The shape of the specified output Value object must match the actual output shape
            if (varValue->Data()->Shape() != valueShape)
                InvalidArgument("The shape %s of the specified Value object for %s does not match the actual shape %s", AsString(varValue->Data()->Shape()).c_str(), getGradient ? "gradient" : "output", AsString(valueShape).c_str());
        }

        ValuePtr nodeValue;
        switch (var.GetDataType())
        {
        case DataType::Float:
            nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(var,
                                                                           getGradient ? computationNode->As<ComputationNode<float>>()->Gradient() : computationNode->As<ComputationNode<float>>()->Value(),
                                                                           computationNode->GetMBLayout());
            break;
        case DataType::Double:
            nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(var,
                                                                            getGradient ? computationNode->As<ComputationNode<double>>()->Gradient() : computationNode->As<ComputationNode<double>>()->Value(),
                                                                            computationNode->GetMBLayout());
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(var.GetDataType()));
            break;
        }

        if (varValue == nullptr)
            varValue = nodeValue->DeepClone();
        else
            varValue->CopyFrom(*nodeValue);
    }

    void CompositeFunction::GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs)
    {
        // Now copy the Forward values of output nodes from the network to outputs' Value objects
        for (auto outputVarValuePair : outputs)
            GetNodeOutputOrGradient(outputVarValuePair.first, outputs[outputVarValuePair.first], m_variableToNodeMap[outputVarValuePair.first], false /*getGradient*/);
    }

    void CompositeFunction::GetNetworkGradients(std::unordered_map<Variable, ValuePtr>& gradients)
    {
        auto networkInputs = this->Inputs();
        // Now copy the gradient values of input nodes of the network to gradients' Value objects
        for (auto gradientVarValuePair : gradients)
        {
            // Only gradients corresponding to inputs of the network can be obtained
            if (std::find(networkInputs.begin(), networkInputs.end(), gradientVarValuePair.first) == networkInputs.end())
                InvalidArgument("Backpropagated gradient values can only be obtained for inputs of a Function");

            // Gradients can only be obtained for parameter variables or input variables that NeedsGradient
            if (!gradientVarValuePair.first.NeedsGradient())
                InvalidArgument("Gradient value incorrectly requested for an Output or Constant Variable, or an Input Variable with NeedsGradient setting of false");

            auto computationNodePtr = m_variableToNodeMap[gradientVarValuePair.first];

            if (!computationNodePtr->NeedsGradient())
                LogicError("Backpropagated gradient value cannot be read from a ComputationNode that has NeedsGradient set to false");

            GetNodeOutputOrGradient(gradientVarValuePair.first, gradients[gradientVarValuePair.first], computationNodePtr, true /*getGradient*/);
        }
    }

    /*virtual*/ BackPropStatePtr CompositeFunction::Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                                            std::unordered_map<Variable, ValuePtr>& outputs,
                                                            const DeviceDescriptor& computeDevice,
                                                            const std::unordered_set<Variable>& outputsToRetainBackwardStateFor)
    {
        // TODO: How about zero argument functions?
        // TODO: We need a better way to determine the ElementType for the network
        auto dataType = arguments.begin()->second->Data()->GetDataType();
        if (dataType == DataType::Float)
            GetComputationNetwork<float>(computeDevice, outputsToRetainBackwardStateFor);
        else
            GetComputationNetwork<double>(computeDevice, outputsToRetainBackwardStateFor);

        // TODO: Avoid copying the data when possible

        // Feed data into the arguments of the network
        PopulateNetworkInputs(arguments);

        // Dropout nodes have an implicit input in the form of the random mask that is applied to its explicit input
        // This mask is regerated every minibatch and hence dropout nodes with a non-zero dropout rate must me marked outdated
        // w.r.t. inputs to force evaluation in each minibatch
        list<ComputationNodeBasePtr> dropoutNodes = m_computationNetwork->GetNodesWithType(OperationNameOf(DropoutNode));
        for (auto& nodeIter : dropoutNodes)
            nodeIter->SetEvalTimeStampOutdatedWrtAll();
        std::unordered_set<Variable> functionOutputs(this->Outputs().begin(), this->Outputs().end());
        std::vector<ComputationNodeBasePtr> outputsToEvaluate;

        for (auto outputVarValuePair : outputs)
        {
            // Ensure that only a subset of this function's outputs are being asked to be evaluated
            if (functionOutputs.find(outputVarValuePair.first) == functionOutputs.end())
                InvalidArgument("Requested output is not an Ouptut of the Function");

            auto outputComputationNode = m_variableToNodeMap[outputVarValuePair.first];
            outputsToEvaluate.push_back(outputComputationNode);
        }

        // The 'outputsToRetainBackwardStateFor' nodes also need to be evaluated if not already specified in 'outputs'
        for (auto rootVarForBackprop : outputsToRetainBackwardStateFor)
        {
            if (outputs.find(rootVarForBackprop) == outputs.end())
                outputsToEvaluate.push_back(m_variableToNodeMap[rootVarForBackprop]);
        }

        ScopedNetworkOperationMode modeGuard(m_computationNetwork, outputsToRetainBackwardStateFor.empty() ? NetworkOperationMode::inferring : NetworkOperationMode::training);

        m_computationNetwork->ForwardProp(outputsToEvaluate);

        GetNetworkOutputs(outputs);

        // TODO: How to deal with the specified 'computeDevice'

        return (outputsToRetainBackwardStateFor.size() > 0) ? MakeSharedObject<CNTKBackPropState>(this->shared_from_this(), std::make_pair(arguments.begin()->first, m_variableToNodeMap[arguments.begin()->first]->GetEvalTimeStamp())) : nullptr;
    }

    /*virtual*/ void CompositeFunction::Backward(const BackPropStatePtr& state,
                                                 const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                                                 std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs)
    {
        auto backpropState = dynamic_cast<const CNTKBackPropState*>(state.get());
        if (backpropState == nullptr)
            InvalidArgument("Invalid backprop state specified");

        // TODO: Support multiple concurrent backprop states
        if (backpropState->EvalTimeStamp().second != m_variableToNodeMap[backpropState->EvalTimeStamp().first]->GetEvalTimeStamp())
            LogicError("The specified backprop state specified cannot be used for backpropagation as the Function's internal state was modified by subsequent Forward calls to the function."
                       "This is not a user error but a shortcoming of the current implementation where multiple independent backprop states are not simultaneously supported");

        if (rootGradientValues.size() > 1)
            LogicError("Currently gradient backprop from only one of the Function Outputs is supported");

        // TODO: Avoid copying the data when possible

        // Zero all gradients of nodes below the root nodes
        for (auto rootGradientVarValuePair : rootGradientValues)
            m_computationNetwork->ZeroInputGradients(m_variableToNodeMap[rootGradientVarValuePair.first]);

        // Feed data into the arguments of the network
        PopulateNetworkGradients(rootGradientValues);

        // Backpropagate through the network
        ScopedNetworkOperationMode modeGuard(m_computationNetwork, NetworkOperationMode::training);

        auto rootComputationNodePtr = m_variableToNodeMap[rootGradientValues.begin()->first];
        m_computationNetwork->GetNestedNetwork(rootComputationNodePtr)->Backprop(FrameRange(nullptr), true, true);

        GetNetworkGradients(backPropagatedGradientValuesForInputs);

        // TODO: How to deal with the specified 'computeDevice'
    }

    FunctionPtr UnaryOp(PrimitiveOpType op, const Variable& operand, Dictionary&& opConfig, const std::wstring& name)
    {
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, std::vector<Variable>({ operand }), std::move(opConfig), name), name);
    }

    FunctionPtr Negate(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Negate, operand, Dictionary(), name);
    }

    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Sigmoid, operand, Dictionary(), name);
    }

    FunctionPtr Tanh(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Tanh, operand, Dictionary(), name);
    }

    FunctionPtr ReLU(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::ReLU, operand, Dictionary(), name);
    }

    FunctionPtr Exp(const Variable& operand, const std::wstring& name/* = L""*/)
        {
        return UnaryOp(PrimitiveOpType::Exp, operand, Dictionary(), name);
    }

    FunctionPtr Log(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Log, operand, Dictionary(), name);
        }

    FunctionPtr Square(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return ElementTimes(operand, operand, name);
    }

    FunctionPtr Sqrt(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Sqrt, operand, Dictionary(), name);
    }

    FunctionPtr Round(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return Floor(Plus(operand, ScalarConstant(operand.GetDataType(), 0.5f)), name);
    }

    FunctionPtr Floor(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Floor, operand, Dictionary(), name);
    }

    FunctionPtr Ceil(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return Negate(Floor(Negate(operand)), name);
    }

    FunctionPtr Abs(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Abs, operand, Dictionary(), name);
    }

    FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Reciprocal, operand, Dictionary(), name);
    }

    FunctionPtr Softmax(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Softmax, operand, Dictionary(), name);
    }

    FunctionPtr Hardmax(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Hardmax, operand, Dictionary(), name);
    }

    FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name /*= L""*/)
    {
        if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
            LogicError("TransposeAxes currently does not support transposing dynamic axes");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis1] = axis1;
        additionalProperties[PrimitiveFunction::AttributeNameAxis2] = axis2;
        return UnaryOp(PrimitiveOpType::TransposeAxes, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Transpose(const Variable& operand, const std::wstring& name /*= L""*/)
    {
        if (operand.Shape().NumAxes() <= 2)
            InvalidArgument("Transpose can already be called for 1D or 2D operands");

        return TransposeAxes(operand, Axis(0), Axis(1), name);
    }
    FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name /*= L""*/)
    {
        if ((endIndex - beginIndex) <= 0)
            InvalidArgument("CNTK::Slice: endIndex (%d) - beginIndex (%d) must be a positive number", endIndex, beginIndex);

        if (axis == Axis::DefaultBatchAxis())
            LogicError("Slice is currently unsupported along the batch axis");

        if (axis.IsStaticAxis())
            return Internal::Slice(operand, axis, beginIndex, endIndex, name);

        auto operandAxes = operand.DynamicAxes();
        auto findAxis = std::find(operandAxes.begin(), operandAxes.end(), axis);
        if (findAxis == operandAxes.end())
            InvalidArgument("The specified dynamic axis named %S does not match any of the dynamic axes of the operand", axis.Name().c_str());

        auto beginFlagsLambda = [beginIndex, operand]() {
            return (beginIndex > 0) ? Minus(ScalarConstant(operand.GetDataType(), 1.0f), Internal::IsWithin(operand, beginIndex)) : Internal::IsWithin(operand, beginIndex);
        };

        auto endFlagsLambda = [endIndex, operand]() {
            return (endIndex > 0) ? Internal::IsWithin(operand, endIndex) : Minus(ScalarConstant(operand.GetDataType(), 1.0f), Internal::IsWithin(operand, endIndex));
        };

        FunctionPtr flags;
        if (beginIndex == 0)
            flags = endFlagsLambda();
        else if (endIndex == 0)
            flags = beginFlagsLambda();
        else
            flags = ElementTimes(beginFlagsLambda(), endFlagsLambda());

        // Since we are slicing along a dynamic axis, the output variable's dynamic axes will be different than the operand
        std::vector<Axis> newDynamicAxes;
        for (auto operandAxis : operandAxes)
        {
            if (operandAxis == axis)
            {
                // If we are selecting just one frame from the dynamic axis, we can remove that axis
                if ((endIndex - beginIndex) > 1)
                    newDynamicAxes.push_back(CompositeFunction::NextAutoGeneratedDynamicAxis());
            }
            else
                newDynamicAxes.push_back(operandAxis);
        }

        return Internal::Gather(operand, flags, newDynamicAxes);
    }

    FunctionPtr Dropout(const Variable& operand, double dropoutRate, const std::wstring& name /*= L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameDropoutRate] = dropoutRate;

        return UnaryOp(PrimitiveOpType::Dropout, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name /*= L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNewShape] = newShape;

        return UnaryOp(PrimitiveOpType::Reshape, operand, std::move(additionalProperties), name);
    }
    FunctionPtr BinaryOp(PrimitiveOpType op, const Variable& leftOperand, const Variable& rightOperand, Dictionary&& opConfig, const std::wstring& name)
    {
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, std::vector<Variable>({ leftOperand, rightOperand }), std::move(opConfig), name), name);
    }

    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Plus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Minus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::ElementTimes, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return ElementTimes(leftOperand, Reciprocal(rightOperand), name);
    }

    FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Equal, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::NotEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Less, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::LessEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Greater, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::GreaterEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank /*= 1*/, const std::wstring& name/* = L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOutputRank] = outputRank;
        return BinaryOp(PrimitiveOpType::Times, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank /*= 1*/, const std::wstring& name/* = L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOutputRank] = outputRank;
        return BinaryOp(PrimitiveOpType::TransposeTimes, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::SquaredError, prediction, targets, Dictionary(), name);
    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return ReduceSum(Minus(ReduceLogSum(prediction, Axis(0)), TransposeTimes(labels, prediction)), name);
        //return BinaryOp(PrimitiveOpType::CrossEntropyWithSoftmax, prediction, labels, Dictionary(), name);
    }

    FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return ReduceSum(Minus(ScalarConstant(prediction.GetDataType(), 1.0f), TransposeTimes(labels, Hardmax(prediction))), name);
        //return BinaryOp(PrimitiveOpType::ClassificationError, prediction, labels, Dictionary(), name);
    }

    FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 2)
            InvalidArgument("PastValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic sequence-axis");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::PastValue, operand, initialState, std::move(additionalProperties), name);
    }

    FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 2)
            InvalidArgument("FutureValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic sequence-axis");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::FutureValue, operand, initialState, std::move(additionalProperties), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::SumAll, operand, Dictionary(), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name/* = L""*/)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name/* = L""*/)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalLogSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name/* = L""*/)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMeanReductionOpName, axis, name);
    }

    FunctionPtr ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name/* = L""*/)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMaxReductionOpName, axis, name);
    }

    FunctionPtr ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name/* = L""*/)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMinReductionOpName, axis, name);
    }
    FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name /*= L""*/)
    {
        Constant meanVar(mean);
        Constant invStdDevVar(invStdDev);

        return ElementTimes(Minus(operand, meanVar), invStdDevVar);
    }

    FunctionPtr Convolution(const Variable& convolutionMap,
                            const Variable& operand,
                            const NDShape& strides,
                            const std::vector<bool>& sharing,
                            const std::vector<bool>& autoPadding,
                            const NDShape& lowerPad,
                            const NDShape& upperPad,
                            bool transpose,
                            size_t maxTempMemSizeInSamples,
                            const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = lowerPad;
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = upperPad;
        additionalProperties[PrimitiveFunction::AttributeNameTranspose] = transpose;
        additionalProperties[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples] = maxTempMemSizeInSamples;

        return BinaryOp(PrimitiveOpType::Convolution, convolutionMap, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Pooling(const Variable& operand,
                        PoolingType poolingType,
                        const NDShape& poolingWindowShape,
                        const NDShape& strides,
                        const std::vector<bool>& autoPadding,
                        const NDShape& lowerPad,
                        const NDShape& upperPad,
                        const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePoolingType] = (size_t)poolingType;
        additionalProperties[PrimitiveFunction::AttributeNamePoolingWindowShape] = poolingWindowShape;
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = lowerPad;
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = upperPad;

        return UnaryOp(PrimitiveOpType::Pooling, operand, std::move(additionalProperties), name);
    }

    FunctionPtr BatchNormalization(const Variable& operand,
                                   const Variable& scale,
                                   const Variable& bias,
                                   const Variable& runningMean,
                                   const Variable& runningInvStd,
                                   bool spatial,
                                   double normalizationTimeConstant,
                                   double blendTimeConstant,
                                   double epsilon,
                                   bool useCuDNNEngine,
                                   const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[L"spatial"] = spatial;
        additionalProperties[L"normalizationTimeConstant"] = normalizationTimeConstant;
        additionalProperties[L"blendTimeConstant"] = blendTimeConstant;
        additionalProperties[L"epsilon"] = epsilon;
        additionalProperties[L"useCuDNNEngine"] = useCuDNNEngine;

        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::BatchNormalization,
                                                                             std::vector<Variable>({ operand, scale, bias, runningMean, runningInvStd }),
                                                                             std::move(additionalProperties),
                                                                             name),
                                         name);
    }

    FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name /*= L""*/)
    {
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Clip, std::vector<Variable>({ operand, min, max }), Dictionary(), name), name);
    }

    FunctionPtr Splice(const std::vector<Variable>& operands, size_t axis, const std::wstring& name /*= L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = Axis(axis);

        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Splice, operands, std::move(additionalProperties), name), name);
    }
    FunctionPtr Combine(const std::vector<FunctionPtr>& operands, const std::wstring& name/* = L""*/)
    {
        std::unordered_set<FunctionPtr> uniqueOperands;
        std::vector<Variable> inputs;
        for (auto operand : operands)
        {
            if (uniqueOperands.find(operand) != uniqueOperands.end())
                LogicError("All function operands specified to Combine must be unique");

            uniqueOperands.insert(operand);
            auto currentFunctionOutputs = operand->Outputs();
            std::copy(currentFunctionOutputs.begin(), currentFunctionOutputs.end(), std::back_inserter(inputs));
        }

        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Combine, inputs, Dictionary(), name), name);
    }

    namespace Internal
    {
        FunctionPtr PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name /*= L""*/)
        {
            return BinaryOp(PrimitiveOpType::PackedIndex, operand, index, Dictionary(), name);
        }

        FunctionPtr GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name /*= L""*/)
        {
            return BinaryOp(PrimitiveOpType::GatherPacked, operand, packedIndex, Dictionary(), name);
        }

        FunctionPtr ZeroesLike(const Variable& operand)
        {
            if (operand.Shape().NumAxes() > 1)
                LogicError("ZerosLike currently does not support operands with more than 1 static axes");

            auto rowSliceFunc = Internal::Slice(operand, Axis(0), 0, 1);
            return Minus(rowSliceFunc, rowSliceFunc);
        }

        FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name /*= L""*/)
        {
            if (offset == 0)
                InvalidArgument("Internal::CNTK::IsWithin: The offset must be positive");

            if (offset > 0)
                return PastValue(ZeroesLike(operand), ScalarConstant(operand.GetDataType(), 1.0f), offset, name);
            else
                return FutureValue(ZeroesLike(operand), ScalarConstant(operand.GetDataType(), 1.0f), -offset, name);
        }

        FunctionPtr Where(const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name /*= L""*/)
        {
            auto additionalProperties = Dictionary();
            std::vector<std::wstring> newDynamicAxesNames;
            for (auto axis : newDynamicAxes)
                newDynamicAxesNames.push_back(axis.Name());

            additionalProperties[PrimitiveFunction::AttributeNameNewDynamicAxes] = AsDictionaryValueVector(newDynamicAxesNames);
            return UnaryOp(PrimitiveOpType::Where, condition, std::move(additionalProperties), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name /*= L""*/)
        {
            return Internal::GatherPacked(operand, Internal::PackedIndex(operand, Where(condition, newDynamicAxes)));
        }

        FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name /*= L""*/)
        {
            auto additionalProperties = Dictionary();
            additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
            additionalProperties[PrimitiveFunction::AttributeNameBeginIndex] = (size_t)beginIndex;
            additionalProperties[PrimitiveFunction::AttributeNameEndIndex] = (size_t)endIndex;

            return UnaryOp(PrimitiveOpType::Slice, operand, std::move(additionalProperties), name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name /*= L""*/)
        {
            using namespace std::placeholders;

            if (axis.IsStaticAxis())
            {
                auto additionalProperties = Dictionary();
                additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
                additionalProperties[PrimitiveFunction::AttributeNameReductionOpName] = reductionOpName;
                return UnaryOp(PrimitiveOpType::ReduceElements, operand, std::move(additionalProperties), name);
            }

            if (axis == Axis::DefaultBatchAxis())
                LogicError("Reduction is currently unsupported along the batch axis");

            if (reductionOpName != PrimitiveFunction::InternalSumReductionOpName)
                LogicError("%S reduction along dynamic axis is currently unsupported", reductionOpName.c_str());

            std::function<FunctionPtr(const Variable& leftOperand, const Variable& rightOperand)> reductionFunctor;
            if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                reductionFunctor = std::bind(Plus, _1, _2, L"");

            // We are reducing over a dynamic axis which is currently implemented using recurrence
            auto cumulativeSumFunctionPlaceholder = Placeholder(operand.Shape());
            auto prevAccumulatedValuesFunction = PastValue(cumulativeSumFunctionPlaceholder, ScalarConstant(operand.GetDataType(), 0.0f), 1);
            auto cumulativeSumFunction = reductionFunctor(prevAccumulatedValuesFunction, operand);
            cumulativeSumFunction->ReplacePlaceholders({ { cumulativeSumFunctionPlaceholder, cumulativeSumFunction } });

            return CNTK::Slice(cumulativeSumFunction, axis, -1, 0);
        }
   }
}
