%module(directors="1") CNTKLib
//%feature("autodoc", "1");

%include <stl.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_pair.i>
%include <std_shared_ptr.i>
%include <windows.i>
%include <attribute.i>
%include <arrays_csharp.i>

// include the unordered_map.i.
%include "std_unordered_map.i"

%{
    #include "CNTKLibrary.h"
    #pragma warning(disable : 4100)
%}

%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(std::vector<float>);

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;
%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;

%template() std::vector<bool>;
%template() std::pair<size_t, double>;
%template() std::vector<std::pair<size_t, double>>;

// Ignore things in CNTKLibrary.h that are not exposed for C# Eval.
%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);

%ignore CNTK::Internal::GenerateUid(std::wstring&& prefix);
%ignore CNTK::Internal::GenerateUid(VariableKind varKind);
%ignore CNTK::Internal::GenerateUid(const std::wstring& prefix);

%ignore CNTK::PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes());
%ignore CNTK::PlaceholderVariable(const NDShape& shape, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes());
%ignore CNTK::PlaceholderVariable(const std::wstring& name = L"");

%ignore CNTK::InputVariable(const NDShape& shape,
                            bool isSparse, 
                            ::CNTK::DataType dataType, 
                            bool needsGradient, 
                            const std::wstring& name, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            ::CNTK::DataType dataType, 
                            bool needsGradient, 
                            const std::wstring& name = L"", 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            DataType dataType, 
                            const std::wstring& name, 
                            const std::vector<Axis>& 
                            dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            DataType dataType, 
                            const wchar_t* name, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            DataType dataType, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            bool isSparse, 
                            ::CNTK::DataType dataType, 
                            const std::wstring& name, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            bool isSparse, 
                            ::CNTK::DataType dataType,
                            const wchar_t* name, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, 
                            bool isSparse, 
                            ::CNTK::DataType dataType, 
                            const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());

%ignore CNTK::OutputVariable(const NDShape& shape, 
                             ::CNTK::DataType dataType, 
                             Function* ownerFunction, 
                             const std::vector<Axis>& dynamicAxes, 
                             const std::wstring& name = L"");

%ignore CNTK::Variable::CompositeFunction;
%ignore CNTK::Variable::Trainer;
%ignore CNTK::Varaiable::PrimitiveFunction;

%ignore CNTK::ConstantInitializer(double value = 0.0);
%ignore CNTK::UniformInitializer(double scale, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::NormalInitializer(double scale, 
                                int outputRank = SentinelValueForInferParamInitRank, 
                                int filterRank = SentinelValueForInferParamInitRank, 
                                unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::XavierInitializer(double scale = DefaultParamInitScale, 
                                int outputRank = SentinelValueForInferParamInitRank, 
                                int filterRank = SentinelValueForInferParamInitRank, 
                                unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::GlorotUniformInitializer(double scale = DefaultParamInitScale, 
                                       int outputRank = SentinelValueForInferParamInitRank, 
                                       int filterRank = SentinelValueForInferParamInitRank, 
                                       unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::GlorotNormalInitializer(double scale = DefaultParamInitScale, 
                                      int outputRank = SentinelValueForInferParamInitRank, 
                                      int filterRank = SentinelValueForInferParamInitRank, 
                                      unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::HeUniformInitializer(double scale = DefaultParamInitScale, 
                                   int outputRank = SentinelValueForInferParamInitRank, 
                                   int filterRank = SentinelValueForInferParamInitRank, 
                                   unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::HeNormalInitializer(double scale = DefaultParamInitScale, 
                                  int outputRank = SentinelValueForInferParamInitRank, 
                                  int filterRank = SentinelValueForInferParamInitRank, 
                                  unsigned long seed = SentinelValueForAutoSelectRandomSeed);
%ignore CNTK::BilinearInitializer(size_t kernelWidth, size_t kernelHeight);
%ignore CNTK::RandomInitializerWithRank(const ParameterInitializer& initializer, int outputRank, int filterRank);

%ignore CNTK::IDictionarySerializable;
%ignore CNTK::DictionaryValue;
%ignore CNTK::Dictionary;
%ignore CNTK::ParameterInitializer;

%ignore std::hash<::CNTK::Parameter>;
%ignore CNTK::hash<::CNTK::Constant>;

%ignore CNTK::Function::CompositeFunction;
%ignore CNTK::Function::Trainer;
%ignore CNTK::Function::Backward(const BackPropStatePtr& state,
                                 const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                                 std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs);
%ignore CNTK::Function::Forward(const std::unordered_map<Variable, ValuePtr>& arguments, 
                                std::unordered_map<Variable, ValuePtr>& outputs, 
                                const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(), 
                                const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {});
%ignore CNTK::Function::Serialize() const;
%ignore CNTK::Function::Deserialize(const Dictionary& dictionary, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());
%ignore CNTK::Function::Parameters() const;
%ignore CNTK::Function::Constants() const;
%ignore CNTK::Function::Placeholders() const;
%ignore CNTK::Function::Attributes() const;
%ignore CNTK::Function::PrintGraph() const;
%ignore CNTK::Function::ReplacePlaceholders(const std::unordered_map<Variable, Variable>& placeholderReplacements);
%ignore CNTK::Function::ReplacePlaceholder(const Variable& placeholderReplacement);
%ignore CNTK::Function::Function(const std::vector<Variable>& inputs, 
                                 const std::vector<Variable>& outputs,
                                 Dictionary&& functionConfig,
                                 const std::wstring& name = L"",
                                 const std::wstring& uid = Internal::GenerateUid(L"UserDefinedFunction"));
%ignore CNTK::Function::RestoreFromCheckpoint(const Dictionary& dictionary);

%ignore CNTK::Parameter;
%ignore CNTK::Constant;
%ignore CNTK::BackPropState;
%ignore CNTK::PoolingType;

%ignore CNTK::Negate(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::operator-(const Variable& operand);
%ignore CNTK::Sigmoid(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Tanh(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Sin(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Cos(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::ReLU(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Exp(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Log(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Square(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Sqrt(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Round(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Floor(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Ceil(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Abs(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Reciprocal(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Softmax(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Hardmax(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name = L"");
%ignore CNTK::Transpose(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = L"");
%ignore CNTK::RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name);
%ignore CNTK::RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name);
%ignore CNTK::Dropout(const Variable& operand, double dropoutRate, const std::wstring& name = L"");
%ignore CNTK::Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name = L"");
%ignore CNTK::Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::operator+(const Variable& leftOperand, const Variable& rightOperand);
%ignore CNTK::Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::operator-(const Variable& leftOperand, const Variable& rightOperand);
%ignore CNTK::LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, int inferInputRankToMap, const std::wstring& name = L"");
%ignore CNTK::Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"");
%ignore CNTK::Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"");
%ignore CNTK::TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::CosineDistance(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
%ignore CNTK::WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name = L"");
%ignore CNTK::SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");
%ignore CNTK::CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
%ignore CNTK::ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const std::wstring& name = L"");
%ignore CNTK::ClassificationError(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
%ignore CNTK::PastValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");
%ignore CNTK::PastValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"");
%ignore CNTK::FutureValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");
%ignore CNTK::FutureValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"");
%ignore CNTK::ReduceSum(const Variable& operand, const std::wstring& name = L"");
%ignore CNTK::ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = L"");
%ignore CNTK::Convolution(const Variable& convolutionMap,
                          const Variable& operand,
                          const NDShape& strides = {1}, 
                          const std::vector<bool>& sharing = {true},
                          const std::vector<bool>& autoPadding = {true}, 
                          const NDShape& lowerPad = {0},
                          const NDShape& upperPad = {0}, 
                          bool transpose = false, 
                          size_t maxTempMemSizeInSamples = 0, 
                          const std::wstring& name = L"");
%ignore CNTK::ROIPooling(const Variable& convolutionMap, const Variable& rois, const NDShape& roiOutputShape, const std::wstring& name = L"");

%ignore CNTK::Pooling(const Variable& operand, 
                      PoolingType poolingType, 
                      const NDShape& poolingWindowShape, 
                      const NDShape& strides = {1}, 
                      const std::vector<bool>& autoPadding = {false},
                      const NDShape& lowerPad = {0}, 
                      const NDShape& upperPad = {0}, 
                      const std::wstring& name = L"");
%ignore CNTK::BatchNormalization(const Variable& operand, 
                                 const Variable& scale, 
                                 const Variable& bias, 
                                 const Variable& runningMean, 
                                 const Variable& runningInvStd, 
                                 bool spatial,
                                 double normalizationTimeConstant = 0, 
                                 double blendTimeConstant = 0, 
                                 double epsilon = 0.00001, 
                                 bool useCuDNNEngine = false, 
                                 const std::wstring& name = L"");
%ignore CNTK::OptimizedRNNStack(const Variable& operand, 
                                const Variable& weights, 
                                size_t hiddenSize, 
                                size_t numLayers, 
                                bool bidirectional = false, 
                                const std::wstring& recurrentOp = L"lstm", 
                                const std::wstring& name = L"");
%ignore CNTK::Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name = L"");
%ignore CNTK::ElementSelect(const Variable& condition, const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
%ignore CNTK::Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name = L"");
%ignore CNTK::AsBlock(FunctionPtr&& composite, const std::unordered_map<Variable, Variable>& argumentsMap, const std::wstring& blockOpName, const std::wstring& blockName = L"");

%ignore CNTK::Sequence;

%ignore CNTK::TrainingParameterSchedule;
%ignore CNTK::TrainingParameterPerUnitSchedule;
%ignore CNTK::TrainingParameterPerSampleSchedule;
%ignore CNTK::TrainingParameterPerMinibatchSchedule;
%ignore CNTK::LearningRateSchedule;
%ignore CNTK::LearningRatePerSampleSchedule;
%ignore CNTK::LearningRatePerMinibatchSchedule;
%ignore CNTK::MomentumAsTimeConstantSchedule;
%ignore CNTK::AdditionalLearningOptions;
%ignore CNTK::Learner;

%ignore CNTK::SGDLearner(const std::vector<Parameter>& parameters, 
                         const LearningRateSchedule& learningRateSchedule, 
                         AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
%ignore CNTK::MomentumSGDLearner(const std::vector<Parameter>& parameters, 
                                 const LearningRateSchedule& learningRateSchedule, 
                                 const MomentumSchedule& momentumSchedule, 
                                 AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
%ignore CNTK::NesterovLearner(const std::vector<Parameter>& parameters, 
                              const LearningRateSchedule& learningRateSchedule, 
                              const MomentumSchedule& momentumSchedule, 
                              AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
%ignore CNTK::DefaultVarianceMomentum;
%ignore CNTK::AdamLearner(const std::vector<Parameter>& parameters, 
                          const LearningRateSchedule& learningRateSchedule, 
                          const MomentumSchedule& momentumSchedule, 
                          const MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum, 
                          bool lowMemory = true, 
                          AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
%ignore CNTK::AdaGradLearner(const std::vector<Parameter>& parameters, 
                             const LearningRateSchedule& learningRateSchedule, 
                             bool needAveMultiplier = true, 
                             AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());
%ignore CNTK::RMSPropLearner(const std::vector<Parameter>& parameters,
                             const LearningRateSchedule& learningRateSchedule, 
                             double gamma, 
                             double inc, 
                             double dec, 
                             double max, 
                             double min, 
                             bool needAveMultiplier = true, 
                             AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

%ignore CNTK::DistributedLearner;
%ignore CNTK::CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, 
                                                   LearnerPtr learner, 
                                                   size_t distributeAfterSamples, 
                                                   bool useAsyncBufferedParameterUpdate = false);
%ignore CNTK::CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, 
                                                            LearnerPtr learner, 
                                                            size_t distributeAfterSamples, 
                                                            bool useAsyncBufferedParameterUpdate = false);
%ignore CNTK::CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum = true,
        bool resetSGDMomentumAfterAggregation = true,
        double blockLearningRate = 1.0);
%ignore CNTK::CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        bool useNestrovMomentum = true,
        bool resetSGDMomentumAfterAggregation = true,
        double blockLearningRate = 1.0);

%ignore CNTK::Trainer;
%ignore CNTK::StreamInformation;
%ignore std::hash<::CNTK::StreamInformation>;

%ignore CNTK::MinibatchData;
%ignore CNTK::MinibatchSource;
%ignore CNTK::CreateCompositeMinibatchSource(const Dictionary& configuration);
%ignore CNTK::StreamConfiguration;
%ignore CNTK::TextFormatMinibatchSource;
%ignore CNTK::ComputeInputPerDimMeansAndInvStdDevs;
%ignore CNTK::DistributedWorkerDescriptor;
%ignore CNTK::DistributedCommunicator;
%ignore CNTK::QuantizedDistributedCommunicator;
%ignore CNTK::MPICommunicator();
%ignore CNTK::QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits);
%ignore CNTK::MinibatchInfo;
%ignore CNTK::DistributedTrainer;
%ignore CNTK::CreateDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, size_t distributedAfterSampleCount = 0);
%ignore CNTK::CreateQuantizedDataParallelDistributedTrainer(QuantizedDistributedCommunicatorPtr communicator, 
                                                            bool useAsyncBufferedParameterUpdate, 
                                                            size_t distributedAfterSampleCount);
%ignore CNTK::CreateBlockMomentumDistributedTrainer;

%ignore std::hash<::CNTK::DistributedWorkerDescriptor>;

// Todo: add correct typemap as they might be useful for C# in future.
%ignore CNTK::NDMask::DataBuffer() const;

// Ignore things in CNTKLibraryInternals.h that are not exposed for C# Eval.
%ignore CNTK::Internal::PrimitiveFunction;
%ignore CNTK::Internal::CompositeFunction;
%ignore CNTK::Internal::MaxNumCPUThreadsSet;
%ignore CNTK::PrimitiveOpType;
%ignore CNTK::Internal::IsWithin(const Variable& operand, int offset, const std::wstring& name = L"");
%ignore CNTK::Internal::PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name = L"");
%ignore CNTK::Internal::GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name = L"");
%ignore CNTK::Internal::ScatterPacked(const Variable& operand, const Variable& packedIndex, const Variable& condition, const std::wstring& name = L"");
%ignore CNTK::Internal::ZeroesWithDynamicAxesLike(const Variable& operand);
%ignore CNTK::Internal::Where(const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
%ignore CNTK::Internal::Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
%ignore CNTK::Internal::Gather(const Variable& operand, 
                               const Variable& condition, 
                               const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, 
                               const std::wstring& name = L"");
%ignore CNTK::Internal::Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
%ignore CNTK::Internal::Scatter(const Variable& operand, 
                                const Variable& condition, 
                                const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, 
                                const std::wstring& name = L"");
%ignore CNTK::Internal::Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = L"");
%ignore CNTK::Internal::ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name = L"");

%ignore CNTK::Internal::EnableReversingTensorShapesInErrorMessages();
%ignore CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled();
%ignore CNTK::Internal::AlwaysAllowSettingDefaultDevice();
%ignore CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed();
%ignore CNTK::Internal::SetAutomaticUnpackingOfPackedValues(bool disable);
%ignore CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled();
%ignore CNTK::Internal::SetComputationNetworkTraceLevel(int traceLevel);
%ignore CNTK::Internal::GetComputationNetworkTraceLevel();
%ignore CNTK::Internal::SetGPUMemoryAllocationTraceLevel(int traceLevel);
%ignore CNTK::Internal::ForceSynchronousCUDAKernelExecutions();
%ignore CNTK::Internal::ForceDeterministicAlgorithms();
%ignore CNTK::Internal::SetFixedRandomSeed(unsigned long fixedRandomSeed);
%ignore CNTK::Internal::EnableForwardValuesSharing();
%ignore CNTK::Internal::EnableHyperMemoryCompress();
%ignore CNTK::Internal::AreEquivalent(const ::CNTK::FunctionPtr& f1, const ::CNTK::FunctionPtr& f2);
%ignore CNTK::Internal::AreEquivalent(const ::CNTK::Variable& v1, const ::CNTK::Variable& v2, bool allowParameterAndConstantsEquivalence = false);
%ignore CNTK::Internal::AreEqual(const ::CNTK::NDArrayView& view1, const ::CNTK::NDArrayView& view2, double relativeTolerance = 0.0, double absoluteTolerance = 0.0);

// map the pointer to array
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

%rename (GetAllDevices) CNTK::DeviceDescriptor::AllDevices;
%rename (GetBestDevice) CNTK::DeviceDescriptor::BestDevice;
%rename (GetDefaultDevice) CNTK::DeviceDescriptor::DefaultDevice;
%rename (GetCPUDevice) CNTK::DeviceDescriptor::CPUDevice;
%rename (GetDeviceType) CNTK::DeviceDescriptor::Type;
%rename (GetId) CNTK::DeviceDescriptor::Id;
%rename (AreEqualDeviceDescriptor) CNTK::operator==(const DeviceDescriptor& left, const DeviceDescriptor& right);

%typemap(cscode) CNTK::DeviceDescriptor %{

    public uint Id
    {
        get { return GetId(); }
    }

    public DeviceKind Type
    {
        get { return GetDeviceType(); }
    }

    public static DeviceDescriptor CPUDevice
    {
        get { return GetCPUDevice(); }
    }

    public static DeviceDescriptor DefaultDevice
    {
        get { return GetDefaultDevice(); }
    }

    public static DeviceDescriptor BestDevice
    {
        get { return GetBestDevice(); }
    }

    public static System.Collections.Generic.List<DeviceDescriptor> AllDevices
    {
        get {
            var devices = GetAllDevices();
            var ret = new System.Collections.Generic.List<DeviceDescriptor>(devices.Count);
            foreach (var d in devices)
            {
                ret.Add(d);
            }
            return ret;
        }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        DeviceDescriptor p = obj as DeviceDescriptor;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    public bool Equals(DeviceDescriptor p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    public static bool operator ==(DeviceDescriptor first, DeviceDescriptor second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(first, second);
    }

    public static bool operator !=(DeviceDescriptor first, DeviceDescriptor second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        return this.GetDeviceType().GetHashCode();
    }
%}

%rename (GetName) CNTK::Axis::Name;
%rename (IsOrderedAxis) CNTK::Axis::IsOrdered;
%rename (AreEqualAxis) CNTK::operator==(const Axis& first, const Axis& second);

%typemap(cscode) CNTK::Axis %{

    public string Name
    {
        get 
        {
            return GetName();
        }
    }

    public bool IsStatic
    {
        get 
        {
            return IsStaticAxis();
        }
    }

    public bool IsDynamic
    {
        get 
        {
            return IsDynamicAxis();
        }
    }

    public bool IsOrdered
    {
        get 
        {
            return IsOrderedAxis();
        }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        Axis p = obj as Axis;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(this, p);
    }

    public bool Equals(Axis p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(this, p);
    }

    public static bool operator ==(Axis first, Axis second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(first, second);
    }

    public static bool operator !=(Axis first, Axis second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        if (this.IsDynamicAxis())
        {
            return this.GetName().GetHashCode();
        }
        else
        {
            return this.StaticAxisIndex().GetHashCode();
        }
    }
%}

%rename (GetName) CNTK::Function::Name;
%rename (GetUid) CNTK::Function::Uid;
%rename (GetRootFunction) CNTK::Function::RootFunction;
%rename (GetInputs) CNTK::Function::Inputs;
%rename (GetOutput) CNTK::Function::Output;
%rename (GetOutputs) CNTK::Function::Outputs;
%rename (GetArguments) CNTK::Function::Arguments;

%typemap(cscode) CNTK::Function %{

    public string Name
    {
        get 
        {
            return GetName();
        }
    }

    public string Uid
    {
        get 
        {
            return GetUid();
        }
    }

    public Function RootFunction
    {
        get 
        {
            return GetRootFunction();
        }
    }

    public System.Collections.Generic.List<Variable> Outputs
    {
        get 
        {
            var vars = GetOutputs();
            var ret = new System.Collections.Generic.List<Variable>(vars.Count);
            foreach (var v in vars)
            {
                ret.Add(v);
            }
            return ret;
        }
    }

    public Variable Output
    {
        get { return GetOutput(); }
    }

    public System.Collections.Generic.List<Variable> Arguments
    {
        get 
        {
            var vars = GetArguments();
            var ret = new System.Collections.Generic.List<Variable>(vars.Count);
            foreach (var v in vars)
            {
                ret.Add(v);
            }
            return ret;
        }
    }

    // Todo: do we have a better place to put this function?
    public static Function Combine(System.Collections.Generic.IEnumerable<Variable> outputVariable)
    {
        var varVect = new VariableVector();
        foreach (var v in outputVariable)
        {
            varVect.Add(v);
        }
        return CNTKLib.Combine(varVect);
    }
%}

%rename (GetShape) CNTK::Variable::Shape;
%rename (GetName) CNTK::Variable::Name;
%rename (AreEqualVariable) CNTK::operator==(const Variable& first, const Variable& second);

%typemap(cscode) CNTK::Variable %{

    public NDShape Shape
    {
        get { return GetShape(); }
    }

    public string Name
    {
        get { return GetName(); }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        Variable p = obj as Variable;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(this, p);
    }

    public bool Equals(Variable p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(this, p);
    }

    public static bool operator ==(Variable first, Variable second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(first, second);
    }

    public static bool operator !=(Variable first, Variable second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        // Todo: the hash value in C++ is size_t, but only in in C#
        return (int)GetHashValue();
    }
%}

%rename (GetDimensions) CNTK::NDShape::Dimensions;
%rename (GetRank) CNTK::NDShape::Rank;
%rename (GetTotalSize) CNTK::NDShape::TotalSize;
%rename (AreEqualShape) CNTK::operator==(const NDShape& first, const NDShape& second);

%typemap(cscode) CNTK::NDShape %{
    public uint Rank
    {
        get { return GetRank(); }
    }

    public System.Collections.Generic.List<uint> Dimensions
    {
        get 
        { 
            var ret = new System.Collections.Generic.List<uint>((int)GetRank());
            foreach (var dim in GetDimensions())
            {
                ret.Add((uint)dim);
            }
            return ret;
        }
    }

    public uint TotalSize
    {
        get { return GetTotalSize(); }
    }

    public uint this[int key]
    {
        get { return GetDimensionSize((uint)key); }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        NDShape p = obj as NDShape;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(this, p);
    }

    public bool Equals(NDShape p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(this, p);
    }

    public static bool operator ==(NDShape first, NDShape second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(first, second);
    }

    public static bool operator !=(NDShape first, NDShape second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        //Todo: another hash function??
        return this.GetDimensions().GetHashCode();
    }

%}

%rename (GetDevice) CNTK::Value::Device;
%rename (GetShape) CNTK::Value::Shape;
%rename (_IsSparse) CNTK::Value::IsSparse;
%rename (_IsReadOnly) CNTK::Value::IsReadOnly;
%rename (_MaskedCount) CNTK::Value::MaskedCount;

%typemap(cscode) CNTK::Value %{

    public DeviceDescriptor Device
    {
        get
        {
            return GetDevice();
        }
    }

    public DataType DataType
    {
        get
        {
            return GetDataType();
        }
    }

    public StorageFormat StorgeFormat
    {
        get
        {
            return GetStorageFormat();
        }
    }

    public NDShape Shape
    {
        get
        {
            return GetShape();
        }
    }

    public bool IsSparse
    {
        get
        {
            return _IsSparse();
        }
    }

    public bool IsReadOnly
    {
        get
        {
            return _IsReadOnly();
        }
    }

    public int MaskedCount
    {
        get
        {
            return (int)_MaskedCount();
        }
    }

    // Create Value object from dense input: batch, sequence or batch of sequences.

    public static Value CreateBatch<T>(NDShape shape, System.Collections.Generic.List<T> batch, DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    public static Value CreateSequence<T>(NDShape shape, System.Collections.Generic.List<T> sequence,
                                          bool seqStartFalg,
                                          DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    public static Value CreateBatchOfSequences<T>(NDShape shape,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> batchOfSequences,
                                                  System.Collections.Generic.List<bool> seqStartFalg,
                                                  DeviceDescriptor computeDevice)
    {
        var dim = shape.TotalSize;

        // Todo: deal with seqStartFlag
        if (typeof(T).Equals(typeof(float)))
        {
            var inputSeqVector = new FloatVectorVector();
            foreach (var seq in batchOfSequences)
            {
                if (seq.Count % dim != 0)
                {
                    throw new System.ArgumentException("the number of data in sequences does not match the input dimension");
                }
                var samples = new FloatVector(seq);
                inputSeqVector.Add(samples);
            }
            return Value.CreateDenseFloat(shape, inputSeqVector, computeDevice);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputSeqVector = new DoubleVectorVector();
            foreach (var seq in batchOfSequences)
            {
                if (seq.Count % dim != 0)
                {
                    throw new System.ArgumentException("the number of data in sequences does not match the input dimension");
                }
                var samples = new DoubleVector(seq);
                inputSeqVector.Add(samples);
            }
            return Value.CreateDenseDouble(shape, inputSeqVector, computeDevice);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input: batch, sequence or batch of sequences
    public static Value CreateBatch<T>(uint vacabSize, System.Collections.Generic.List<uint> batch, DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    public static Value CreateSequence<T>(uint vacabSize,
                                          System.Collections.Generic.List<uint> sequence,
                                          bool seqStartFalg,
                                          DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    public static Value CreateBatchOfSequences<T>(uint vacabSize, 
                                                  System.Collections.Generic.List<System.Collections.Generic.List<uint>> batchOfSequences,
                                                  System.Collections.Generic.List<bool> seqStartFalg,
                                                  DeviceDescriptor computeDevice)
    {
        throw new System.NotImplementedException("Not implemented");
    }

    // Create Value object from sparse input: batch, sequence or batch of sequences.
    public static Value CreateBatch<T>(NDShape shape, 
                                       System.Collections.Generic.List<T> data,
                                       System.Collections.Generic.List<uint> indexes,
                                       System.Collections.Generic.List<uint> nnzCounts,
                                       DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    public static Value CreateSequence<T>(NDShape shape, 
                                          System.Collections.Generic.List<T> data,
                                          System.Collections.Generic.List<uint> indexes,
                                          System.Collections.Generic.List<uint> nnzCounts,
                                          bool SequenceStartFlag,
                                          DeviceDescriptor computeDevice)
    {
        throw new System.ArgumentException("Not implemented yet.");
    }

    // Create Value based on sparse input
    // Todo: could this be a extension to Value class??
    // Todo: use Variable instead of varName. VarName as extension method
    public static Value CreateBatchOfSequences<T>(NDShape shape,
                                                 System.Collections.Generic.List<System.Collections.Generic.List<T>> data,
                                                 System.Collections.Generic.List<System.Collections.Generic.List<uint>> indexes,
                                                 System.Collections.Generic.List<System.Collections.Generic.List<uint>> nnzCounts,
                                                 System.Collections.Generic.List<bool> SequenceStartFlag,
                                                 DeviceDescriptor computeDevice)
    {
        throw new System.NotImplementedException("Not implemented");
    }
%}


%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%include "CNTKValueExtend.i"

//
// NDArryView
//
%extend CNTK::NDArrayView {
    NDArrayView(const NDShape& viewShape, float *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Float, viewShape, dataBuffer, numBufferElements * sizeof(float), device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, double *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Double, viewShape, dataBuffer, numBufferElements * sizeof(double), device, readOnly);
    }
}

// 
// NDShape
//
%extend CNTK::NDShape {
    size_t GetDimensionSize(size_t axisId)
    {
        return (*self)[axisId];
    }
}

