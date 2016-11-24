%module(directors="1") CSharpBindings
//%feature("autodoc", "1");

%include <stl.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_pair.i>
%include <std_shared_ptr.i>
%include <windows.i>
%include <attribute.i>
%include "std_unordered_map.i"

%{
    #include "CNTKLibrary.h"
    #pragma warning(disable : 4100)
%}

%nspace CNTK;

%template(SizeTVector) std::vector<size_t>;
%template(BoolVector) std::vector<bool>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(VariableVector) std::vector<CNTK::Variable>;
//%template() std::vector<CNTK::Parameter>;
//%template() std::vector<CNTK::Constant>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;
//%template() std::vector<CNTK::StreamConfiguration>;

//%template() std::vector<CNTK::DictionaryValue>;

%template() std::vector<std::shared_ptr<CNTK::Function>>;
%template() std::vector<std::shared_ptr<CNTK::Learner>>;
%template() std::pair<size_t, double>;
%template() std::vector<std::pair<size_t, double>>;

%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(std::vector<float>);

// SWIG does not understand ValuePtr here.
%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;

%ignore CNTK::IDictionarySerializable;
%ignore CNTK::DictionaryValue;
%ignore CNTK::Dictionary;
%ignore CNTK::ParameterInitializer;

%ignore CNTK::Parameter;
%ignore CNTK::Constant;
%ignore CNTK::PoolingType;
%ignore CNTK::TrainingParameterSchedule;
%ignore CNTK::TrainingParameterPerUnitSchedule;
%ignore CNTK::MomentumAsTimeConstantSchedule;
%ignore CNTK::AdditionalLearningOptions;
%ignore CNTK::Learner;
%ignore CNTK::Trainer;
%ignore CNTK::StreamInformation;
%ignore CNTK::MinibatchData;
%ignore CNTK::MinibatchSource;
%ignore CNTK::StreamConfiguration;
%ignore CNTK::DistributedWorkerDescriptor;
%ignore CNTK::DistributedCommunicator;
%ignore CNTK::QuantizedDistributedCommunicator;
%ignore CNTK::MinibatchInfo;
%ignore CNTK::DistributedTrainer;



%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"





