%module(directors="1") cntk_py
//%feature("autodoc", "1");


%include "stl.i"
%include "std_wstring.i"
%include <std_vector.i>
%include <std_map.i>
%include <std_set.i>
%include <std_pair.i>
%include <stdint.i>
%include <windows.i>
%include <attribute.i>
%include <std_shared_ptr.i>

%implicitconv CNTK::Variable;

%rename(_forward) CNTK::Function::Forward;
%rename(_add_progress_writers) CNTK::Internal::AddProgressWriters;
%rename(_backward) CNTK::Function::Backward;
%rename(_infer_outputs) CNTK::Function::InferOutputs;
%rename(_update) CNTK::Learner::Update;
%rename(sgd_learner) CNTK::SGDLearner;
%rename(momentum_sgd_learner) CNTK::MomentumSGDLearner;
%rename(gpu_device) CNTK::DeviceDescriptor::GPUDevice;
%rename(cpu_device) CNTK::DeviceDescriptor::CPUDevice;
%rename(GPUProperties) CNTK::GPUProperties;
%rename(get_gpu_properties) CNTK::DeviceDescriptor::GetGPUProperties;
%rename(times_transpose) CNTK::TransposeTimes;
%rename(sequence_slice) CNTK::Sequence::Slice;
%rename(sequence_reduce_sum) CNTK::Sequence::ReduceSum;
%rename(momentum_as_time_constant_schedule) CNTK::MomentumAsTimeConstantSchedule;
%rename(ctf_deserializer) CNTK::CTFDeserializer;
%rename(htk_feature_deserializer) CNTK::HTKFeatureDeserializer;
%rename(htk_mlf_deserializer) CNTK::HTKMLFDeserializer;

%rename(_none) CNTK::DictionaryValue::Type::None;

// Disabling warning about constructor shadowing, learner tests check this.
%warnfilter(401, 509) CNTK::TrainingParameterPerUnitSchedule;
%warnfilter(509) CNTK::MomentumAsTimeConstantSchedule;
%warnfilter(509) CNTK::NDArrayView::NDArrayView;

%warnfilter(315) CNTK::TrainingParameterPerSampleSchedule;

// Disabling warning about movable constructor shadowing, io tests check this.
%warnfilter(509) CNTK::DictionaryValue::DictionaryValue;
%warnfilter(509) CNTK::Dictionary::Dictionary;

// Disabling warning about Trainer shadowing, trainer tests check this.
%warnfilter(509) TrainerImpl;

// Returning an immutable string by reference.
%warnfilter(473) CNTK::Function::OpName;

// Operator overloading is not supported by Python.
%rename(eq) operator==;
%ignore CNTK::Variable::operator FunctionPtr;
%ignore CNTK::AddConfigString;
%ignore CNTK::GetCorrespondingOutputVariableFromClone;

// Specialization of non-template function - hash,
// TODO: it is not clear how to limit this only to hash, but we do not use partial specialization in other places.
#pragma SWIG nowarn=-317

// Disabling enable_shared_from_this - we never use this class to actually access the object.
%warnfilter(401) CNTK::NDArrayView;
%warnfilter(401) CNTK::NDMask;
%warnfilter(401) CNTK::Function;
%warnfilter(401) CNTK::Trainer;
%warnfilter(401) CNTK::Evaluator;
%warnfilter(401) CNTK::Value;
%warnfilter(401) CNTK::BackPropState;
%warnfilter(401) CNTK::MinibatchSource;

%warnfilter(401, 509) CNTK::MomentumAsTimeConstantSchedule;

// The following operators are not supported in Python.
%ignore operator<<;
%ignore operator>>;
%ignore CNTK::DictionaryValue::operator=;
%ignore CNTK::DictionaryValue::Value;

%ignore CNTK::Dictionary::operator=;
%ignore CNTK::Dictionary::operator[];

%ignore CNTK::TrainingParameterSchedule::operator=;
%ignore CNTK::TrainingParameterSchedule::operator[];


// renaming overloads for TrainMinibatch and TestMinibatch that take a map 
// of Variables and MinibatchData as their first parameter. If this is not done, 
// the overloads that are legal in C++ will be shadowed and ignored by SWIG.
// The naming here is somewhat cumbersome, but it's only intended for internal
// consumption in proxy objects.
%rename(train_minibatch_overload_for_minibatchdata) CNTK::Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(train_minibatch_overload_for_minibatchdata) CNTK::Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>&, std::unordered_map<Variable, ValuePtr>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(test_minibatch_overload_for_minibatchdata) CNTK::Evaluator::TestMinibatch(const std::unordered_map<Variable, MinibatchData>&, const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());
%rename(test_minibatch_overload_for_minibatchdata) CNTK::Evaluator::TestMinibatch(const std::unordered_map<Variable, MinibatchData>&, std::unordered_map<Variable, ValuePtr>& , const DeviceDescriptor& = DeviceDescriptor::UseDefaultDevice());

%rename(l1_regularization_weight) CNTK::AdditionalLearningOptions::l1RegularizationWeight;
%rename(l2_regularization_weight) CNTK::AdditionalLearningOptions::l2RegularizationWeight;
%rename(ndcg_at_1) CNTK::NDCGAt1;

// if we don't except RandomUniform the corresponding template functions will not be generated
%rename("%(utitle)s", %$isfunction, notregexmatch$name="RandomUniform") "";
%rename("%(utitle)s", %$isvariable) "";

%template() std::vector<bool>;
%template() std::vector<int>;
%template() std::vector<size_t>;
%template() std::vector<float>;
%template() std::vector<double>;
%template() std::vector<std::vector<size_t>>;
%template() std::vector<std::vector<float>>;
%template() std::vector<std::vector<double>>;

%template() std::vector<CNTK::Variable>;
%template() std::vector<CNTK::OutputVariable>;
%template() std::vector<CNTK::Parameter>;
%template() std::vector<CNTK::Constant>;
%template() std::vector<CNTK::Axis>;
%template() std::vector<CNTK::DeviceDescriptor>;
%template() std::vector<CNTK::StreamConfiguration>;
%template() std::vector<CNTK::HTKFeatureConfiguration>;
%template() std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template() std::vector<std::shared_ptr<CNTK::Value>>;
%template() std::vector<std::shared_ptr<CNTK::Function>>;
%template() std::vector<std::shared_ptr<CNTK::Learner>>;
%template() std::vector<std::shared_ptr<CNTK::DistributedLearner>>;
%template() std::vector<std::shared_ptr<CNTK::Trainer>>;
%template() std::vector<std::shared_ptr<CNTK::Evaluator>>;
%template() std::vector<std::shared_ptr<CNTK::ProgressWriter>>;
%template() std::pair<double, double>;
%template() std::pair<size_t, double>;
%template() std::pair<size_t, size_t>;
%template() std::pair<size_t, int>;
%template() std::vector<std::pair<size_t, double>>;
%template() std::vector<std::pair<size_t, size_t>>;
%template() std::vector<std::pair<CNTK::Variable, CNTK::Variable>>;
%template() std::vector<CNTK::Dictionary>;
%template() std::vector<std::wstring>;
%template() std::pair<std::vector<std::shared_ptr<CNTK::NDArrayView>>, std::vector<bool>>;


// They are defined twice under CNTK::Internal and under CNTK namespace
%ignore CNTK::Internal::Combine;
%ignore CNTK::Internal::Where;
%ignore CNTK::Internal::Gather;
%ignore CNTK::Internal::Scatter;
%ignore CNTK::Internal::Slice;
%ignore CNTK::Internal::MaxNumCPUThreadsSet;
%ignore CNTK::Internal::CosineDistanceWithNegativeSamples;

// These aren't exported from the CNTK C++ library
%ignore CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;
%ignore CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;
%ignore CNTK::Internal::IsRenamingFunctionsAllowed;
%ignore CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled;
%ignore CNTK::Internal::GetComputationNetworkTraceLevel;
%ignore CNTK::Internal::GetComputationNetworkTrackGapNans;
%ignore CNTK::Internal::TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir, const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize = nullptr);
%ignore CNTK::Internal::Convolution; 

%ignore CNTK::Function::Function(const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& name = L"", const std::wstring& uid = Internal::GenerateUid(L"UserDefinedFunction"));

%{
#define SWIG_FILE_WITH_INIT
%}

%init %{
    import_array();
%}

//
// Whenever a tuple of dynamic axes is returned we need to reverse it
//
%feature("shadow") CNTK::Variable::DynamicAxes %{
def dynamic_axes(self):
    return tuple(reversed($action(self)))
%}

%fragment("NDShapeToTuple", "header")
{
    PyObject *NDShapeToTuple(const CNTK::NDShape& shape)
    {
        int rank = static_cast<int>(shape.Rank());
        auto result = PyTuple_New(rank);
        for (size_t i = 0; i < rank; i++)
        {
            int dim = static_cast<int>((&shape)->operator[](i));
            PyTuple_SetItem(result, rank-i-1, PyInt_FromLong(dim));
        }
        return result;
    }
}

%fragment("NDArrayViewToNumPy", "header")
{
    PyObject* NDArrayViewToNumPy(const CNTK::NDArrayView* self) {
        if ((*self).GetStorageFormat() != StorageFormat::Dense)
            throw std::invalid_argument("only dense supported at the moment");

        // FIXME use not yet existing NDShape function that returns the dimensions at once
        std::vector<size_t> dimensions_cntk = (*self).Shape().Dimensions();
        std::vector<size_t> dimensions;

        // We have at least one element. In case the shape is empty (e.g.
        // '()'), we have a scalar, which we need to copy (e.g. a constant).
        size_t num_elements = 1;

        // CNTK uses column major, thus we reverse the shape
        for (int i = static_cast<int>(dimensions_cntk.size()) - 1; i >= 0; i--)
        {
            dimensions.push_back(dimensions_cntk[i]);
            num_elements *= dimensions_cntk[i];
        }

        npy_intp* shape = reinterpret_cast<npy_intp*>(&dimensions[0]);

        CNTK::DataType cntk_type = (*self).GetDataType();

        NDArrayView* cpuView;
        if ((*self).Device() != DeviceDescriptor::CPUDevice())
        {
            cpuView = new NDArrayView(cntk_type, (*self).Shape(), DeviceDescriptor::CPUDevice());
            cpuView->CopyFrom((*self));
        }
        else
        {
            cpuView = const_cast<NDArrayView*>(&(*self));
        }

        NPY_TYPES numpy_type;
        void* buffer;

        if (cntk_type == CNTK::DataType::Float)
        {
            numpy_type = NPY_FLOAT;
            buffer = (void*)cpuView->DataBuffer<float>();
        }
        else if (cntk_type == CNTK::DataType::Double)
        {
            numpy_type = NPY_DOUBLE;
            buffer = (void*)cpuView->DataBuffer<double>();
        }
        else
        {
            throw std::invalid_argument("unknown CNTK data type");
        }

        PyObject* ndarray = PyArray_SimpleNew(static_cast<int>(dimensions.size()), shape, numpy_type);
        void *arr_data = PyArray_DATA((PyArrayObject*)ndarray);

        memcpy(arr_data, buffer, PyArray_ITEMSIZE((PyArrayObject*) ndarray) * num_elements);

        if ((*self).Device() != DeviceDescriptor::CPUDevice())
        {
            delete cpuView;
        }
		
        return ndarray;
    }
}

%fragment("pydict_insert", "header")
{
     template<typename T> bool pydict_insert(PyObject* dictionary, const T& key, swig_type_info *swig_type, PyObject* item) {
        /* FIXME We would love to do the following, but the hashing does not
         * correctly work here, which is why we never find the keys. Instead,
         * we will for now loop over the dictionary and use C++ comparison.
         * Although not beautiful, there should not be a lot of overhead since
         * the dictionary usually contains only a handful of variables as keys.
        if (PyDict_Contains($input, returned_var))
        {
            SWIG_exception_fail(SWIG_ValueError, "returned output map contains unknown key");
        }
         */

        PyObject *py_key, *py_value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dictionary, &pos, &py_key, &py_value)) {
            void *cntk_key = 0 ;
            int flags = 0;
            if (swig_type == $descriptor(CNTK::Variable *))
                flags |= SWIG_POINTER_IMPLICIT_CONV;
            int res = SWIG_ConvertPtr(py_key, &cntk_key, swig_type, flags);
            if (!SWIG_IsOK(res)) {
                std::string s("cannot convert key of dictionary to ");
                s+=typeid(T).name();
                SWIG_exception_fail(SWIG_ArgError(res), s.c_str());
            }
            if (!cntk_key) {
                std::string s("invalid null reference when converting key of dictionary to ");
                s+=typeid(T).name();
                SWIG_exception_fail(SWIG_ValueError, s.c_str());
            }

            T* cntk_var = reinterpret_cast<T*>(cntk_key);
            if (*cntk_var == key)
            {
                PyDict_SetItem(dictionary, py_key, item);
                return true;
            }
        }
fail:   return false;
    }
}

%typecheck(1000) std::vector<CNTK::Variable> const& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::vector<CNTK::Variable> const& {
     //in std::vector<CNTK::Variable>
     if (PyList_Check($input)) {
        std::vector<CNTK::Variable>* vec = new std::vector<CNTK::Variable>();

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::Variable");
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, $descriptor(CNTK::Variable *),  SWIG_POINTER_IMPLICIT_CONV);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert list element to CNTK::Variable");
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a list element to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            vec->push_back(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::Variable");
        }

        $1 = vec;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}

%typemap(freearg) std::vector<CNTK::Variable> const& {
    //freearg std::vector<CNTK::Variable>
    delete $1;
}

//
// Converting Python list {DictionaryValue} to std::vector
//
%typecheck(1000) std::vector<CNTK::DictionaryValue>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::vector<CNTK::DictionaryValue>& {
     //in DictionaryValue
     if (PyList_Check($input)) {
        std::vector<CNTK::DictionaryValue>* vec = new std::vector<CNTK::DictionaryValue>();

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::DictionaryValue");
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, $descriptor(CNTK::DictionaryValue *),  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert list element to CNTK::DictionaryValue");
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a list element to CNTK::DictionaryValue");
            }

            CNTK::DictionaryValue* var = reinterpret_cast<CNTK::DictionaryValue*>(raw_var);

            vec->push_back(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::DictionaryValue");
        }

        $1 = vec;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}

%fragment("DictionaryValueToPy", "header", fragment="NDShapeToTuple", fragment="NDArrayViewToNumPy")
{
    PyObject *DictionaryValueToPy(const CNTK::DictionaryValue& dictVal)
    {
        PyObject *val = nullptr;
        switch (dictVal.ValueType())
        {
            case CNTK::DictionaryValue::Type::None:
                Py_INCREF(Py_None);
                val = Py_None;
                break;
            case CNTK::DictionaryValue::Type::Bool:
                val = PyBool_FromLong(static_cast<long>(dictVal.Value<bool>()));
                break;
            case CNTK::DictionaryValue::Type::Int:
                val = PyLong_FromLong(static_cast<long>(dictVal.Value<int>()));
                break;
            case CNTK::DictionaryValue::Type::SizeT:
                val = PyLong_FromSize_t(dictVal.Value<size_t>());
                break;
            case CNTK::DictionaryValue::Type::Float:
                val = PyFloat_FromDouble(static_cast<double>(dictVal.Value<float>()));
                break;
            case CNTK::DictionaryValue::Type::Double:
                val = PyFloat_FromDouble(dictVal.Value<double>());
                break;
            case CNTK::DictionaryValue::Type::String:
                val = PyUnicode_FromWideChar(dictVal.Value<std::wstring>().c_str(), dictVal.Value<std::wstring>().length());
                break;
            case CNTK::DictionaryValue::Type::NDShape:
                val = NDShapeToTuple(dictVal.Value<CNTK::NDShape>());
                break;
            case CNTK::DictionaryValue::Type::Axis:
                val = PyTuple_New(3);
                if (val == NULL)
                {
                    SWIG_exception(SWIG_RuntimeError, "error creating tuple for axis");
                }
                if (dictVal.Value<CNTK::Axis>().IsOrdered())
                    PyTuple_SetItem(val, 0, PyUnicode_FromWideChar(L"ordered", 7));
                else
                    PyTuple_SetItem(val, 0, PyUnicode_FromWideChar(L"unordered", 9));
                if (dictVal.Value<CNTK::Axis>().IsDynamicAxis())
                {
                    PyTuple_SetItem(val, 1, PyUnicode_FromWideChar(L"dynamic", 7));
                    PyTuple_SetItem(val, 2, PyUnicode_FromWideChar(
                        dictVal.Value<CNTK::Axis>().Name().c_str(),
                        dictVal.Value<CNTK::Axis>().Name().length()));
                }
                else
                {
                    PyTuple_SetItem(val, 1, PyUnicode_FromWideChar(L"static", 6));
                    PyTuple_SetItem(val, 2, PyLong_FromLong(
                        static_cast<long>(
                            dictVal.Value<CNTK::Axis>().StaticAxisIndex(true)
                        )));
                }
                break;
            case CNTK::DictionaryValue::Type::Vector:
                val = PyList_New(0);
                if (val == NULL)
                {
                    SWIG_exception(SWIG_RuntimeError, "error creating list");
                }
                for (auto it: dictVal.Value<std::vector<CNTK::DictionaryValue> >())
                {
                    PyObject* tmp = DictionaryValueToPy(it);
                    PyList_Append(val, tmp);
                    Py_DECREF(tmp);
                }
                break;
            case CNTK::DictionaryValue::Type::Dictionary:
                val = PyDict_New();
                if (val == NULL)
                {
                    SWIG_exception(SWIG_RuntimeError, "error creating dict");
                }
                for (auto it = dictVal.Value<CNTK::Dictionary>().begin(); it != dictVal.Value<CNTK::Dictionary>().end(); ++it)
                {
                    PyObject *key = PyUnicode_FromWideChar(it->first.c_str(), it->first.length());
                    PyObject *dvp = DictionaryValueToPy(it->second);
                    PyDict_SetItem(val, key, dvp);
                    Py_DECREF(key);
                    Py_DECREF(dvp);
                }
                break;
            case CNTK::DictionaryValue::Type::NDArrayView:
                val = NDArrayViewToNumPy(&(dictVal.Value<CNTK::NDArrayView>()));
                break;
            default:
                SWIG_exception(SWIG_RuntimeError, "unknown type for DictionaryValue object");
                break;
        }
        return val;
fail:
    return NULL;
    }
}

%typemap(out, fragment="DictionaryValueToPy") const CNTK::Dictionary& Attributes(){
    //out Dictionary& Attributes()
    PyObject* container = PyDict_New();
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing a dictionary to Python");
    }

    for (auto it = $1->begin(); it != $1->end(); ++it)
    {
        PyObject *key = PyUnicode_FromWideChar(it->first.c_str(), it->first.length());
        PyObject *val = DictionaryValueToPy(it->second);
        PyDict_SetItem(container, key, val);
        Py_DECREF(key);
        Py_DECREF(val);
    }
    $result = container;
}


%define %eq_for(DATA_TYPE, EQ)
%rename(EQ) operator==(const DATA_TYPE&, const DATA_TYPE&);
%enddef

%eq_for(Variable, Variable_eq)
%eq_for(Constant, Variable_eq)
%eq_for(Parameter, Variable_eq)
%eq_for(Axis, Axis_eq)
%eq_for(DeviceDescriptor, DeviceDescriptor_eq)

//
// size_t converter and extend DictionaryValue constructor
//

// declare python type
struct SizeTWrapper
{
public:
    size_t value;
    SizeTWrapper(int v) : value(static_cast<size_t>(v)) {}
    SizeTWrapper(size_t v) : value(v) {}
};

//inject to c++
%{
struct SizeTWrapper
{
public:
    size_t value;
    SizeTWrapper(int v) : value(static_cast<size_t>(v)) {}
    SizeTWrapper(size_t v) : value(v) {}
};
%}

// extend constructor
%extend CNTK::DictionaryValue {
    DictionaryValue(const SizeTWrapper& w)
    {
        return new DictionaryValue(w.value);
    }
}


%ignore CNTK::Dictionary::Keys;

%extend CNTK::Dictionary {
    PyObject* __getitem__(const wchar_t* key) {
    PyObject *DictionaryValueToPy(const CNTK::DictionaryValue&);
        return DictionaryValueToPy((*($self))[key]);
    }

    void __setitem__(const wchar_t* key, CNTK::DictionaryValue value) {
        (*($self))[key] = value;
    }

    PyObject* keys()
    {
        PyObject* container = PyList_New(0);
        for (auto var : (*($self)))
        {
            PyObject *item = PyUnicode_FromWideChar(var.first.c_str(), var.first.length());
            // No error handling here, because the error will be passed directly to Python
            PyList_Append(container, item);
            Py_DECREF(item);
        }
        return container;
    }
}

%extend CNTK::Axis {
    bool __eq__(const CNTK::Axis& other) const {
        return *$self == other;
    }
}

// Callback support
%feature("director") CNTK::Function;
%feature("nodirector") CNTK::Function::OnPlaceholdersReplaced;
%feature("nodirector") CNTK::Function::OpName;

%feature("director") CNTK::Learner;
%feature("nodirector") CNTK::Learner::Parameters;
%feature("nodirector") CNTK::Learner::ResetLearningRate;

%{
    #include "CNTKLibrary.h"
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include "numpy/ndarraytypes.h"
    #include "numpy/arrayobject.h"
    using namespace CNTK;
%}

//
// Exception handling
//
%include "CNTK_ExceptionHandling.i"

%feature("director:except") {
    if ($error != NULL) {
		PyErr_Print();
        throw Swig::DirectorMethodException();
    }
}

// Common directors
%feature("director") CNTK::TrainingSession;
%feature("nodirector") CNTK::TrainingSession::OnMinibatchStart;
%feature("nodirector") CNTK::TrainingSession::OnCheckpointStart;
%feature("nodirector") CNTK::TrainingSession::GetMinibatchSize;

%feature("director") CNTK::ProgressWriter;
%ignore CNTK::ProgressWriter::UpdateTraining;
%ignore CNTK::ProgressWriter::UpdateTest;
%ignore CNTK::ProgressWriter::WriteTrainingSummary;
%ignore CNTK::ProgressWriter::WriteTestSummary;

//
// NDShape
//
%typecheck(1000) CNTK::NDShape const &, CNTK::NDShape {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyTuple_Check($input) ? 1 : 0;
}

%typemap(in) CNTK::NDShape const & {
     if (PyTuple_Check($input)) {
        size_t rank = PyTuple_Size($input);
        std::vector<size_t> dimensions(rank);
        for (size_t i=0; i<rank; i++)
            dimensions[i] = PyLong_AsLong(PyTuple_GET_ITEM($input, rank-i-1));

        $1 = new CNTK::NDShape(dimensions);
     } else {
         SWIG_exception(SWIG_TypeError, "tuple expected");
     }
}

%typemap(freearg) CNTK::NDShape const & {
    delete (CNTK::NDShape*)$1;
}

%ignore CNTK::NDShape::operator[];
%ignore CNTK::NDShape::AppendShape;
%ignore CNTK::NDShape::Dimensions;

%typemap(out, fragment="NDShapeToTuple") CNTK::NDShape {
    $result = NDShapeToTuple($1);
}

%extend CNTK::NDShape {
    const size_t& __getitem__(int i) {
        // CNTK uses column major, thus we reverse the shape
        size_t rank = (*self).Rank();
        if (i<0)
        {
            return (*self)[-i-1];
        }
        return (*self)[rank-1-i];
    }

    PyObject* dimensions() {
        std::vector<size_t> dims = (*self).Dimensions();
        size_t rank = (*self).Rank();
        PyObject* result = PyTuple_New(rank);
        // CNTK uses column major, thus we reverse the shape
        for (size_t i=0; i<rank; i++)
        {
            int dim = static_cast<int>(dims[i]);
            PyTuple_SET_ITEM(result, rank-1-i, PyInt_FromLong(dim));
        }
        return result;
    }
}

%ignore CNTK::Dictionary::AppendShape;
%rename ("$ignore", fullname=1) CNTK::Variable(const NDShape&, CNTK::DataType, const wchar_t*);


// (size_t)-1 will result into an OverflowException
%ignore CNTK::NDShape::InferredDimension;
//%ignore CNTK::NDShape::Dimensions;
// FIXME: The following is not picked up yet, which is why we have to tag it to
// the module
//%constant long CNTK::NDShape::InferredDimension = -1;
%constant long InferredDimension = -1;

// end of NDShape

//
// Converting Python dictionary {Variable: ValuePtr} to std::unordered_map
//

%define %unordered_map_conversion(KEY_TYPE, VALUE_TYPE, SWIG_KEY_TYPE, SWIG_VALUE_TYPE) 
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    %typecheck(1000) std::unordered_map<KEY_TYPE, VALUE_TYPE> const&,
        const std::unordered_map<KEY_TYPE, VALUE_TYPE>&, 
        std::unordered_map<KEY_TYPE, VALUE_TYPE>&  
    { $1 = PyDict_Check($input) ? 1 : 0; }

    %typemap(in) std::unordered_map<KEY_TYPE, VALUE_TYPE>& (
            std::unordered_map<KEY_TYPE, VALUE_TYPE> args_map
    ) {
         //in unordered_map<K, V>& (unordered_map<K, V> args_map)
         if (PyDict_Check($input)) {

            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next($input, &pos, &key, &value)) {
                void *raw_var = 0 ;
                int key_flags = 0;
                if (SWIG_KEY_TYPE == $descriptor(CNTK::Variable *))
                    key_flags |= SWIG_POINTER_IMPLICIT_CONV;
                int res1 = SWIG_ConvertPtr(key, &raw_var, SWIG_KEY_TYPE,  key_flags );
                if (!SWIG_IsOK(res1)) {
                    SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary"); 
                }
                if (!raw_var) {
                    SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary");
                }

                KEY_TYPE* var = reinterpret_cast<KEY_TYPE*>(raw_var);

                int val_flags = 0; 
                if (SWIG_VALUE_TYPE == $descriptor(CNTK::Variable *))
                    val_flags |= SWIG_POINTER_IMPLICIT_CONV;
                void *raw_value = 0;
                int res2 = SWIG_ConvertPtr(value, &raw_value, SWIG_VALUE_TYPE,  val_flags );
                if (!SWIG_IsOK(res2)) {
                    SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert value of dictionary"); 
                }

                VALUE_TYPE* value;
                if (raw_value) {
                    value = reinterpret_cast<VALUE_TYPE*>(raw_value);
                    args_map.insert(std::make_pair(*var, *value));
                } else {
                    // We got an empty VALUE_TYPE, which carries a nullptr.
                    // This is only used for ValuePtr
                    args_map.insert(std::make_pair(*var, VALUE_TYPE()));
                }

            }

            $1 = &args_map;
         } else {
             SWIG_exception(SWIG_TypeError, "dictionary expected");
         }
    }
%enddef


// Implementing typemapping for InferOutputs virtual function that takes vector of variables by reference
// And can be implemented in Python.

%typemap(directorin) std::vector<CNTK::Variable>& outputs
{
    $input = PyList_New(0);
}

%typemap(directorargout) std::vector<CNTK::Variable>& outputs
{
    if (!PyList_Check($input))
        RuntimeError("List expected");

    auto iterator = std::shared_ptr<PyObject>(PyObject_GetIter($input), [](PyObject* p) { Py_DECREF(p); });
    if (!iterator)
        RuntimeError("Cannot get iterator to the list");

     PyObject *item = nullptr;
     while ((item = PyIter_Next(iterator.get())))
     {
         void *raw_var = nullptr;
         int res = SWIG_ConvertPtr(item, &raw_var, swig::type_info<CNTK::Variable>(),  0);
         if (!SWIG_IsOK(res))
             RuntimeError("Cannot convert list element to CNTK::Variable");

         if (!raw_var)
             RuntimeError("Invalid null reference when converting a list element to CNTK::Variable");

         auto var = reinterpret_cast<CNTK::Variable*>(raw_var);
         $1.push_back(*var);
         Py_DECREF(item);
     }

     if (PyErr_Occurred())
         RuntimeError("Cannot convert list element to CNTK::Variable");
}

// For the output dict (the non-const unordered_map) we need to get the
// modified values and put them back into the dictionary. This is used, when
// e.g. the user puts a variable into the dictionary, hoping that it will
// afterwards point to the proper value.
%typemap(argout, fragment="pydict_insert")
    // Swig would create this conversion for the 'const' variants as well, which
    // we do not want. Therefor, we have to explicitly tell it for which ones it should do it.
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputsToFetch,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputs,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& backPropagatedGradientValuesForInputs,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& gradients,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputsToEvaluate
{
    if (!PyDict_Check($input)) {
        SWIG_exception(SWIG_TypeError, "dictionary expected");
    }

    for (auto it: *$1)
    {
        // Push the ValuePtr onto the heap so that it survives
        std::shared_ptr<CNTK::Value> *smartresult = it.second ? new std::shared_ptr<CNTK::Value>(it.second) : 0;
        PyObject *returned_val = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), $descriptor(CNTK::ValuePtr *), SWIG_POINTER_OWN);

        // Find the corresponding Variable instance in the Python dictionary
        // and set its value to the new ValuePtr
        bool found = pydict_insert<CNTK::Variable>($input, it.first, $descriptor(CNTK::Variable *), returned_val);
        if (!found)
            RuntimeError("could not convert dictionary");
        Py_DECREF(returned_val);
    }
}

// For the output dict (the non-const unordered_map) we need to get the
// modified values and put them back into the dictionary. This is used, when
// e.g. the user puts a variable into the dictionary, hoping that it will
// afterwards point to the proper value.
%typemap(directorargout)
    // Swig would create this conversion for the 'const' variants as well, which
    // we do not want. Therefor, we have to explicitly tell it for which ones it should do it.
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputs,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& backPropagatedGradientValuesForInputs,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& gradients,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputsToEvaluate
{
    // $1 is the C++ input that needs to be filled with the data from the PyDict
    for (auto it: $1)
    {
        // Find the corresponding Variable instance in the Python dictionary
        // and set its value to the new ValuePtr

        /* FIXME We would love to do the following, but the hashing does not
         * correctly work here, which is why we never find the keys. Instead,
         * we will for now loop over the dictionary and use C++ comparison.
         * Although not beautiful, there should not be a lot of overhead since
         * the dictionary usually contains only a handful of variables as keys.
        if (PyDict_Contains($input, returned_var))
        {
            SWIG_exception_fail(SWIG_ValueError, "returned output map contains unknown key");
        }
         */

        PyObject *py_key, *py_value;
        Py_ssize_t pos = 0;
        bool found = false;

        while (PyDict_Next($input, &pos, &py_key, &py_value)) {
            void *cpp_key = 0;
            int key_res = SWIG_ConvertPtr(py_key, &cpp_key, $descriptor(CNTK::Variable *), 0 | SWIG_POINTER_IMPLICIT_CONV);
            if (!SWIG_IsOK(key_res)) {
                RuntimeError("cannot convert key of dictionary"); 
            }

            CNTK::Variable* cntk_var = reinterpret_cast<CNTK::Variable*>(cpp_key);
            if (*cntk_var == it.first)
            {
                found = true;

                void *cpp_val = 0;
                int val_res = SWIG_ConvertPtr(py_value, &cpp_val, $descriptor(CNTK::ValuePtr *), 0);
                if (!SWIG_IsOK(val_res)) {
                    RuntimeError("cannot convert value of dictionary"); 
                }

                CNTK::ValuePtr* cpp_value = reinterpret_cast<CNTK::ValuePtr*>(cpp_val);

                $1[it.first] = cpp_value ? *cpp_value : nullptr;
                break;
            }
        }
        if (!found)
        {
            RuntimeError("could not convert dictionary");
        }
    }
}

//
// Converting Python list [(Variable, Variable)] to std::vector<std::pair<CNTK::Variable, CNTK::Variable>>
//

%typecheck(1000) std::vector<std::pair<CNTK::Variable, CNTK::Variable>> const& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::vector<std::pair<CNTK::Variable, CNTK::Variable>> const& {
     //in std::vector<std::pair<CNTK::Variable, CNTK::Variable>>
     if (PyList_Check($input)) {
        std::vector<std::pair<CNTK::Variable, CNTK::Variable>>* vec = new std::vector<std::pair<CNTK::Variable, CNTK::Variable>>();

        PyObject *listIterator = PyObject_GetIter($input);
        if (listIterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to std::pair<CNTK::Variable, CNTK::Variable>");
        }

        PyObject *listItem;
        while ((listItem = PyIter_Next(listIterator))) {
			PyObject *iterator = PyObject_GetIter(listItem);
			if (iterator == NULL) {
				SWIG_exception_fail(SWIG_ValueError, "cannot convert tuple element to CNTK::Variable");
			}

			std::vector<CNTK::Variable> varPair;
			PyObject *item;
			while ((item = PyIter_Next(iterator))) {
				void *raw_var = 0 ;
				int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__Variable,  SWIG_POINTER_IMPLICIT_CONV);
				if (!SWIG_IsOK(res1)) {
					SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert tuple element to CNTK::Variable");
				}
				if (!raw_var) {
					SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a tuple element to CNTK::Variable");
				}

				CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);
				varPair.push_back(*var);

				Py_DECREF(item);
			}

			if (varPair.size() != 2) {
				SWIG_exception_fail(SWIG_ValueError, "tuple element has more than 2 elements");
			}

			vec->push_back({varPair[0], varPair[1]});

			Py_DECREF(iterator);
			Py_DECREF(listItem);
        }

        Py_DECREF(listIterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to std::pair<CNTK::Variable, CNTK::Variable>");
        }

        $1 = vec;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}

%typemap(freearg) std::vector<std::pair<CNTK::Variable, CNTK::Variable>> const& {
    //freearg std::vector<std::pair<CNTK::Variable, CNTK::Variable>>
    delete $1;
}

//
// Converting Python dictionary {StreamInformation: (mbsize, Value)} to std::unordered_map<CNTK::StreamInformation, std::pair<size_t, size_t>>&
//
%typecheck(1000)  std::unordered_map<CNTK::StreamInformation, std::pair<size_t, size_t>>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyDict_Check($input) ? 1 : 0;
}

%typemap(in)  std::unordered_map<CNTK::StreamInformation, std::pair<size_t, size_t>>& (
         std::unordered_map<CNTK::StreamInformation, std::pair<size_t, size_t>> args_map
) {
     if (PyDict_Check($input)) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next($input, &pos, &key, &value)) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(key, &raw_var, $descriptor(CNTK::StreamInformation *),  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::StreamInformation");
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::StreamInformation");
            }

            CNTK::StreamInformation* var = reinterpret_cast<CNTK::StreamInformation*>(raw_var);

            void *raw_value = 0;

            if (PyTuple_Check(value)) {
                PyObject* first = PyTuple_GET_ITEM(value, 0);
                size_t first_val = PyLong_AsSize_t(first);
                PyObject* second = PyTuple_GET_ITEM(value, 1);
                size_t second_val = PyLong_AsSize_t(second);
                args_map.insert(std::make_pair(*var, std::make_pair(first_val, second_val)));
            } else {
                SWIG_exception(SWIG_TypeError, "tuple expected");
            }
        }

        $1 = &args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

%typecheck(1000)  std::unordered_map<CNTK::StreamInformation, std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr>>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyDict_Check($input) ? 1 : 0;
}

%typemap(in)  std::unordered_map<CNTK::StreamInformation, std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr>>& (
         std::unordered_map<CNTK::StreamInformation, std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr>> args_map
){
     if (PyDict_Check($input)) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next($input, &pos, &key, &value)) {
            void *raw_var = 0 ;
            int res = SWIG_ConvertPtr(key, &raw_var, $descriptor(CNTK::StreamInformation *),  0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "cannot convert key of dictionary to CNTK::StreamInformation");
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::StreamInformation");
            }

            CNTK::StreamInformation* var = reinterpret_cast<CNTK::StreamInformation*>(raw_var);

            if (PyTuple_Check(value)) {
                PyObject* first = PyTuple_GET_ITEM(value, 0);
                void *raw_value1 = 0;
                int res1 = SWIG_ConvertPtr(first, &raw_value1, $descriptor(CNTK::NDArrayViewPtr *),  0);
                if (!SWIG_IsOK(res1)) {
                    SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert value of dictionary to CNTK::NDArrayViewPtr");
                }

                CNTK::NDArrayViewPtr* value1;
                if (raw_value1) {
                    value1 = reinterpret_cast<CNTK::NDArrayViewPtr*>(raw_value1);
                } else {
                    // We got an empty NDArrayViewPtr, which carries a nullptr.
                    value1 = new CNTK::NDArrayViewPtr();
                }

                PyObject* second = PyTuple_GET_ITEM(value, 1);
                void *raw_value2 = 0;
                int res2 = SWIG_ConvertPtr(second, &raw_value2, $descriptor(CNTK::NDArrayViewPtr *),  0);
                if (!SWIG_IsOK(res2)) {
                    SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert value of dictionary to CNTK::NDArrayViewPtr");
                }

                CNTK::NDArrayViewPtr* value2;
                if (raw_value2) {
                    value2 = reinterpret_cast<CNTK::NDArrayViewPtr*>(raw_value2);
                } else {
                    // We got an empty NDArrayViewPtr, which carries a nullptr.
                    value2 = new CNTK::NDArrayViewPtr();
                }

                args_map.insert(std::make_pair(*var, std::make_pair(*value1, *value2)));
            } else {
                SWIG_exception(SWIG_TypeError, "tuple expected");
            }
        }

        $1 = &args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

%typemap(argout, fragment="pydict_insert") 
std::unordered_map<CNTK::StreamInformation, std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr>>& 
{
    if (!PyDict_Check($input)) {
        SWIG_exception(SWIG_TypeError, "dictionary expected");
    }

    for (auto it: *$1)
    {
        // Push onto the heap so that it survives

        std::shared_ptr<CNTK::NDArrayView> *smartresult1 = it.second.first ? new std::shared_ptr<CNTK::NDArrayView>(it.second.first) : 0;
        PyObject *returned_val1 = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult1), $descriptor(CNTK::NDArrayViewPtr *), SWIG_POINTER_OWN);

        std::shared_ptr<CNTK::NDArrayView> *smartresult2 = it.second.second ? new std::shared_ptr<CNTK::NDArrayView>(it.second.second) : 0;
        PyObject *returned_val2 = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult2), $descriptor(CNTK::NDArrayViewPtr *), SWIG_POINTER_OWN);
        PyObject *item = PyTuple_Pack(2, returned_val1, returned_val2);

        // Find the corresponding Variable instance in the Python dictionary
        // and set its value to the new MinibatchData
        bool found = pydict_insert<CNTK::StreamInformation>($input, it.first, $descriptor(CNTK::StreamInformation *), item);
        if (!found)
            RuntimeError("could not convert dictionary");
        Py_DECREF(returned_val1);
        Py_DECREF(returned_val2);
    }
}


//
// Converting std::unordered_set to Python list.
// TOOD: figure out how to return a Python set instead of a list. For this,
// we need to define a hash function on SwigPyObject.
//

%define %unordered_set_ref_conversion_director(DATA_TYPE, _SWIG_TYPE)

%typemap(directorin) std::unordered_set<DATA_TYPE>& {
    PyObject* container = PyList_New(0);

    for (auto var : $1)
    {
        PyObject *item = SWIG_NewPointerObj(new DATA_TYPE(var), _SWIG_TYPE, SWIG_POINTER_OWN );
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
        Py_DECREF(item);
    }

    $input = container;
}

%enddef

%define %unordered_set_conversion(DATA_TYPE, _SWIG_TYPE)

%typemap(out) std::unordered_set<CNTK::DATA_TYPE> {
    PyObject* container = PyList_New(0);
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing a set to Python");
    }
 
    for (auto var : $1)
    {
        PyObject *item = SWIG_NewPointerObj(new CNTK::DATA_TYPE(var), _SWIG_TYPE, SWIG_POINTER_OWN );
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
        Py_DECREF(item);
    }

    $result = container;
}
%enddef
 
%define %unordered_set_ref_conversion(DATA_TYPE, _SWIG_TYPE)

%typecheck(1000) std::unordered_set<DATA_TYPE>&, std::unordered_set<DATA_TYPE>const & {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PySet_Check($input) || PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<DATA_TYPE>& (
        std::unordered_set<DATA_TYPE> args_set 
) {
     if (PySet_Check($input) || PyList_Check($input)) {

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert element"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int flags = 0; 
            if (_SWIG_TYPE == $descriptor(CNTK::Variable *))
                flags |=  SWIG_POINTER_IMPLICIT_CONV;
            int res1 = SWIG_ConvertPtr(item, &raw_var, _SWIG_TYPE,  flags);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert set element"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference");
            }

            DATA_TYPE* var = reinterpret_cast<DATA_TYPE*>(raw_var);

            args_set.insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert set element"); 
        }

        $1 = &args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "set expected");
     }
}

%typemap(out) std::unordered_set<DATA_TYPE>&  {
    PyObject* container = PyList_New(0);
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing a set to Python");
    }

    for (auto var : *$1)
    {
        PyObject *item = SWIG_NewPointerObj(new DATA_TYPE(var), _SWIG_TYPE, SWIG_POINTER_OWN );
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
        Py_DECREF(item);
    }

    $result = container;
}
%enddef

%unordered_set_conversion(CNTK::Variable, $descriptor(CNTK::Variable *))
%unordered_set_conversion(CNTK::Constant, $descriptor(CNTK::Constant *))
%unordered_set_conversion(CNTK::Parameter, $descriptor(CNTK::Parameter *))
%unordered_set_conversion(CNTK::StreamInformation, $descriptor(CNTK::StreamInformation *))
%unordered_set_conversion(CNTK::DistributedWorkerDescriptor, $descriptor(CNTK::DistributedWorkerDescriptor *))

%unordered_set_ref_conversion(CNTK::Variable, $descriptor(CNTK::Variable *))
%unordered_set_ref_conversion(CNTK::Parameter, $descriptor(CNTK::Parameter *))
%unordered_set_ref_conversion(CNTK::StreamInformation, $descriptor(CNTK::StreamInformation *))
%unordered_set_ref_conversion(CNTK::LearnerPtr, SWIGTYPE_p_std__shared_ptrT_CNTK__Learner_t)
%unordered_set_ref_conversion(CNTK::DistributedWorkerDescriptor, $descriptor(CNTK::DistributedWorkerDescriptor *))

// Unordered map conversion
%define %unordered_map_ref_conversion_director(DATA_TYPE1, _SWIG_TYPE1, DATA_TYPE2, _SWIG_TYPE2)

%typemap(directorin) std::unordered_map<DATA_TYPE1, DATA_TYPE2>& {
    PyObject* container = PyDict_New();
    for (auto it : $1)
    {
        PyObject *returned_var = SWIG_NewPointerObj(SWIG_as_voidptr(new DATA_TYPE1(it.first)), _SWIG_TYPE1, SWIG_POINTER_OWN);
        PyObject *returned_val;
        if (it.second == nullptr)
        {
            returned_val = Py_None;
            Py_INCREF(Py_None);
        }
        else {
            returned_val = SWIG_NewPointerObj(SWIG_as_voidptr(new DATA_TYPE2(it.second)), _SWIG_TYPE2, SWIG_POINTER_OWN);
        }

        PyDict_SetItem(container, returned_var, returned_val);

        Py_DECREF(returned_var);
        Py_DECREF(returned_val);
    }

    $input = container;
}

%enddef

%unordered_set_ref_conversion_director(CNTK::Variable,  $descriptor(CNTK::Variable *))
%unordered_set_ref_conversion_director(CNTK::ValuePtr,  $descriptor(CNTK::ValuePtr *))
%unordered_map_ref_conversion_director(CNTK::Variable,  $descriptor(CNTK::Variable *), CNTK::ValuePtr, $descriptor(CNTK::ValuePtr *))
%unordered_map_ref_conversion_director(CNTK::Parameter, $descriptor(CNTK::Parameter *), CNTK::NDArrayViewPtr, $descriptor(CNTK::NDArrayViewPtr *))

%define %unordered_map_ref_conversion(DATA_TYPE1, _SWIG_TYPE1, DATA_TYPE2, _SWIG_TYPE2)

%typemap(out) std::unordered_map<DATA_TYPE1, DATA_TYPE2>& {
    PyObject* container = PyDict_New();
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing a dictionary to Python");
    }

    for (auto it : *$1)
    {
        PyObject *returned_var = SWIG_NewPointerObj(SWIG_as_voidptr(new DATA_TYPE1(it.first)), _SWIG_TYPE1, SWIG_POINTER_OWN);
        PyObject *returned_val = SWIG_NewPointerObj(SWIG_as_voidptr(new DATA_TYPE2(it.second)), _SWIG_TYPE2, SWIG_POINTER_OWN);

        PyDict_SetItem(container, returned_var, returned_val);

        Py_DECREF(returned_var);
        Py_DECREF(returned_val);
    }

    $result = container;
}
%enddef

%unordered_map_conversion(CNTK::Variable,  const CNTK::ValuePtr,       $descriptor(CNTK::Variable *),  $descriptor(const CNTK::ValuePtr))
%unordered_map_conversion(CNTK::Variable,  CNTK::ValuePtr,             $descriptor(CNTK::Variable *),  $descriptor(CNTK::ValuePtr *))
%unordered_map_conversion(CNTK::Variable,  CNTK::Variable,             $descriptor(CNTK::Variable *),  $descriptor(CNTK::Variable *))
%unordered_map_conversion(CNTK::Parameter, const CNTK::NDArrayViewPtr, $descriptor(CNTK::Parameter *), $descriptor(const CNTK::NDArrayViewPtr))
%unordered_map_conversion(CNTK::Parameter, CNTK::NDArrayViewPtr,       $descriptor(CNTK::Parameter *), $descriptor(CNTK::NDArrayViewPtr *))
%unordered_map_conversion(CNTK::Variable,  CNTK::StreamInformation,    $descriptor(CNTK::Variable *),  $descriptor(CNTK::StreamInformation *))
%unordered_map_conversion(CNTK::Variable,  CNTK::MinibatchData,        $descriptor(CNTK::Variable *),  $descriptor(CNTK::MinibatchData *))

%unordered_map_ref_conversion(CNTK::StreamInformation, $descriptor(CNTK::StreamInformation *), CNTK::MinibatchData,  $descriptor(CNTK::MinibatchData *));
%unordered_map_ref_conversion(CNTK::Parameter,         $descriptor(CNTK::Parameter *),         CNTK::NDArrayViewPtr, $descriptor(CNTK::NDArrayViewPtr *));
%unordered_map_ref_conversion(CNTK::Variable,          $descriptor(CNTK::Variable *),          CNTK::Variable,       $descriptor(CNTK::Variable *));

%shared_ptr(CNTK::IDictionarySerializable)
%shared_ptr(CNTK::Evaluator)
%shared_ptr(CNTK::Trainer)
%shared_ptr(CNTK::TrainingSession)
%shared_ptr(CNTK::Function)
%shared_ptr(CNTK::NDArrayView)
%shared_ptr(CNTK::Value)
%shared_ptr(CNTK::NDMask)
%shared_ptr(CNTK::BackPropState)
%shared_ptr(CNTK::Learner)
%shared_ptr(CNTK::MinibatchSource)
%shared_ptr(CNTK::DistributedCommunicator)
%shared_ptr(CNTK::QuantizedDistributedCommunicator)
%shared_ptr(CNTK::DistributedLearner)
%shared_ptr(CNTK::Internal::TensorBoardFileWriter)
%shared_ptr(CNTK::ProgressWriter)

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%inline %{
   // Trainer initializers.
   // Because SWIG cannot properly handle smart pointers to derived classes (causes memory leak during the check for distributed learners),
   // we need to redefine CreateTrainer.
   // TODO: do progressWriters below also have a similar issue?
    CNTK::TrainerPtr TrainerImpl(const ::CNTK::FunctionPtr& model, const ::CNTK::FunctionPtr& lossFunction, const ::CNTK::FunctionPtr& evaluationFunction,
                                 const std::vector<CNTK::DistributedLearnerPtr>& parameterLearners, const std::vector<CNTK::ProgressWriterPtr>& progressWriters)
    {
        std::vector<LearnerPtr> learners;
        learners.reserve(parameterLearners.size());
        for(const auto& l : parameterLearners)
            learners.push_back(l);
        return CreateTrainer(model, lossFunction, evaluationFunction, learners, progressWriters);
    }

    CNTK::TrainerPtr TrainerImpl(const ::CNTK::FunctionPtr& model, const ::CNTK::FunctionPtr& lossFunction, const ::CNTK::FunctionPtr& evaluationFunction,
                                 const std::vector<CNTK::LearnerPtr>& parameterLearners, const std::vector<CNTK::ProgressWriterPtr>& progressWriters)
    {
        return CreateTrainer(model, lossFunction, evaluationFunction, parameterLearners, progressWriters);
    }

    // Global rank of current worker
    size_t WorkerGlobalRank()
    {
        return CNTK::MPICommunicator()->CurrentWorker().m_globalRank;
    }

    // Number of workers
    size_t NumberOfWorkers()
    {
        return CNTK::MPICommunicator()->Workers().size();
    }
%}


//
// NDMask
//
// FIXME ignore is ignored
%ignore CNTK::NDMask::DataBuffer();
%extend CNTK::NDMask {
    PyObject* to_ndarray() {
        std::vector<size_t> cntk_dims = (*self).Shape().Dimensions();
        static_assert(cntk_dims.size()==2, "mask requires exactly two dimensions");
        std::vector<size_t> dimensions = {cntk_dims[1], cntk_dims[0]};

        size_t num_elements = dimensions[0] * dimensions[1];

        npy_intp* shape = reinterpret_cast<npy_intp*>(&dimensions[0]);

        NDMask* cpuMask;
        if ((*self).Device() != DeviceDescriptor::CPUDevice())
        {
            cpuMask = new NDMask((*self).Shape(), DeviceDescriptor::CPUDevice());
            cpuMask->CopyFrom((*self));
        }
        else
        {
            cpuMask = &(*self);
        }

        void* buffer = const_cast<void*>(reinterpret_cast<const void*>(cpuMask->DataBuffer()));

        PyObject* ndarray = PyArray_SimpleNew(static_cast<int>(dimensions.size()), shape, NPY_BYTE);
        void *arr_data = PyArray_DATA((PyArrayObject*)ndarray);

        memcpy(arr_data, buffer, PyArray_ITEMSIZE((PyArrayObject*) ndarray) * num_elements);

        if ((*self).Device() != DeviceDescriptor::CPUDevice())
        {
            delete cpuMask;
        }

        return ndarray;
    }
}

// end NDMask

%include "CNTKValueExtend.i"

//
// NDArrayView
//
%extend CNTK::NDArrayView {

    NDArrayView(PyObject* numpyArrayObject, const CNTK::DeviceDescriptor& device, bool readOnly, bool borrow)
    {
        if (!PyArray_Check((PyArrayObject*)numpyArrayObject))
        {
            // Note that in contrast to numpy.i's implementation we demand NumPy arrays
            // and do not accept arbitrary sequences, which would needed to be copied around.
            throw std::logic_error("NumPy array expected");
        }

        // Borrowing the memory is only allowed on CPU for now
        borrow &= device == DeviceDescriptor::CPUDevice();

        PyArrayObject* array = (PyArrayObject*)numpyArrayObject;

        int rank = PyArray_NDIM(array);

        npy_intp* np_shape = PyArray_SHAPE(array);
        std::vector<size_t> shape(rank);

        npy_intp num_elements = 1;
        // CNTK uses column major, thus we reverse the shape
        for (int i=0; i<rank; i++)
        {
            shape[rank-i-1] = np_shape[i];
            num_elements *= np_shape[i];
        }

        int typecode = PyArray_TYPE(array);

        NDArrayView* view;
        if (typecode == NPY_FLOAT)
        {
            if (borrow)
            {
                 view = new NDArrayView(NDShape(shape), (float*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
            }
            else
            {
                 NDArrayView  tmp(NDShape(shape), (float*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
                 view = new NDArrayView(DataType::Float, tmp.Shape(), device);
                 view->CopyFrom(tmp);
            }
        }
        else if (typecode == NPY_DOUBLE)
        {
            if (borrow)
            {
                 view = new NDArrayView(NDShape(shape), (double*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
            }
            else
            {
                 NDArrayView  tmp(NDShape(shape), (double*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
                 view = new NDArrayView(DataType::Double, tmp.Shape(), device);
                 view->CopyFrom(tmp);
            }
        }
        else
        {
            throw std::logic_error("NumPy array of type float32 or float64 expected");
        }

        return view;
    }


    NDArrayView(const CNTK::NDShape& shape, PyObject* pyData, PyObject* pyColStarts, PyObject* pyRowIndices, const CNTK::DeviceDescriptor& device, bool readOnly, bool borrow)
    {
        //
        // pyData, pyColStarts, and pyRowIndices are fed by
        // scipy.sparse.csr_matrix's data, indptr, and indices
        //

        if (!PyArray_Check((PyArrayObject*)pyData))
        {
            throw std::logic_error("sparse data must be a NumPy array");
        }

        if (!PyArray_Check((PyArrayObject*)pyColStarts))
        {
            throw std::logic_error("indices must be a NumPy array");
        }

        if (!PyArray_Check((PyArrayObject*)pyRowIndices))
        {
            throw std::logic_error("index pointers must be a NumPy array");
        }

        // Borrowing the memory is only allowed on CPU for now
        borrow &= device == DeviceDescriptor::CPUDevice();

        PyArrayObject* data = (PyArrayObject*)pyData;
        PyArrayObject* indices = (PyArrayObject*)pyColStarts;
        PyArrayObject* indptr = (PyArrayObject*)pyRowIndices;

        int typecode = PyArray_TYPE(data);
        size_t numNonZeroValues = PyArray_SIZE(data);

        NDArrayView* view;
        if (typecode == NPY_FLOAT)
        {
            if (borrow)
            {
                view = new NDArrayView(shape,
                 (CNTK::SparseIndexType*)PyArray_DATA(indices),
                 (CNTK::SparseIndexType*)PyArray_DATA(indptr),
                 (float*)PyArray_DATA(data), numNonZeroValues,
                 DeviceDescriptor::CPUDevice(), readOnly);
            }
            else
            {
                NDArrayView tmp(shape,
                 (CNTK::SparseIndexType*)PyArray_DATA(indices),
                 (CNTK::SparseIndexType*)PyArray_DATA(indptr),
                 (float*)PyArray_DATA(data), numNonZeroValues,
                 DeviceDescriptor::CPUDevice(), readOnly);
                view = new NDArrayView(DataType::Float, StorageFormat::SparseCSC, tmp.Shape(), device);
                view->CopyFrom(tmp);
            }
        }
        else if (typecode == NPY_DOUBLE)
        {
            if (borrow)
            {
                view = new NDArrayView(shape,
                 (CNTK::SparseIndexType*)PyArray_DATA(indices),
                 (CNTK::SparseIndexType*)PyArray_DATA(indptr),
                 (double*)PyArray_DATA(data), numNonZeroValues,
                 DeviceDescriptor::CPUDevice(), readOnly);
            }
            else
            {
                NDArrayView tmp(shape,
                 (CNTK::SparseIndexType*)PyArray_DATA(indices),
                 (CNTK::SparseIndexType*)PyArray_DATA(indptr),
                 (double*)PyArray_DATA(data), numNonZeroValues,
                 DeviceDescriptor::CPUDevice(), readOnly);
                view = new NDArrayView(DataType::Double, StorageFormat::SparseCSC, tmp.Shape(), device);
                view->CopyFrom(tmp);
            }
        }
        else
        {
            throw std::logic_error("NumPy array of type float32 or float64 expected");
        }

        return view;
    }

    PyObject* to_ndarray() {
        PyObject *NDArrayViewToNumPy(const CNTK::NDArrayView*);
        return NDArrayViewToNumPy(self);
    }
}

// end of NDArrayView
%template(NDArrayViewFloat) CNTK::NDArrayView::NDArrayView<float>;
%template(NDArrayViewDouble) CNTK::NDArrayView::NDArrayView<double>;
%template(ConstantFloat) CNTK::Constant::Constant<float>;
%template(ConstantDouble) CNTK::Constant::Constant<double>;
%template(ParameterFloat) CNTK::Parameter::Parameter<float>;
%template(ParameterDouble) CNTK::Parameter::Parameter<double>;
%template(random_uniform_float) CNTK::NDArrayView::RandomUniform<float>;
%template(random_uniform_double) CNTK::NDArrayView::RandomUniform<double>;
%template(DictionaryValueFromDict) CNTK::DictionaryValue::DictionaryValue<CNTK::Dictionary>;

%template(training_parameter_per_sample_schedule) CNTK::TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Sample>;
%template(training_parameter_per_minibatch_schedule) CNTK::TrainingParameterPerUnitSchedule<double, CNTK::TrainingParameterSchedule<double>::UnitType::Minibatch>;

typedef CNTK::TrainingParameterPerUnitSchedule<size_t, CNTK::TrainingParameterSchedule<size_t>::UnitType::Sample> MinibatchSizeSchedule;
%template(minibatch_size_schedule) CNTK::TrainingParameterPerUnitSchedule<size_t, CNTK::TrainingParameterSchedule<size_t>::UnitType::Sample>;

//
// The following callback code is only for testing. Will have to be merged with
// the operator classes.
//
%shared_ptr(CNTK::UserBackPropState)

%inline %{

namespace CNTK {

    class UserBackPropState;
    typedef std::shared_ptr<UserBackPropState> UserBackPropStatePtr;

    class UserBackPropState : public BackPropState {
    public:
        UserBackPropState(const FunctionPtr& function, const DeviceDescriptor& computeDevice, PyObject* userData)
            : BackPropState(function, computeDevice), m_userData(userData)
        {
            Py_INCREF(m_userData);
        }

        const PyObject* Data() const
        {
            Py_INCREF(m_userData);
            return m_userData;
        }

        static const PyObject* Data(BackPropStatePtr state)
        {
            CNTK::UserBackPropStatePtr user_state = std::dynamic_pointer_cast<CNTK::UserBackPropState>(state);
            if (user_state == nullptr)
                InvalidArgument("Invalid backprop state specified");

            return user_state->Data();
        }

        virtual ~UserBackPropState()
        {
            Py_DECREF(m_userData);
        }

    private:
        const PyObject* m_userData;
    };
}

%}


//
// Release the GIL before calling into C++
//
%exception {
  Py_BEGIN_ALLOW_THREADS;
  $action
  Py_END_ALLOW_THREADS;
}


//
// Setting up hash calculation so that __hash__ on Swig objects
// are redirected to the std::hash computation of the C++ API
//
%define %py_hash_for(DATA_TYPE)
%extend CNTK::DATA_TYPE {
    const size_t __hash__() {
        return std::hash<CNTK::DATA_TYPE>()(*$self);
    }
}
%enddef

//
// Setting __str__ and __repr__ methods for frequently used Swig objects
//
%define %py_repr_for(TYPE)
%extend CNTK::TYPE {
    const std::wstring __str__() {
        return self->AsString();
    }

    const std::wstring __repr__() {
        return self->AsString();
    }
}
%enddef

%define %py_eq_for(DATA_TYPE, EQ)
%pythoncode %{
DATA_TYPE.__eq__ = lambda a,b: (a is not None and b is not None and EQ(a,b)) or (a is None and b is None)
%}
%enddef

%py_repr_for(Variable)
%py_repr_for(Parameter)
%py_repr_for(Constant)
%py_repr_for(Function)

%py_eq_for(Variable, Variable_eq)
%py_hash_for(Variable)

%py_eq_for(Constant, Variable_eq)
%py_hash_for(Constant)

%py_eq_for(Parameter, Variable_eq)
%py_hash_for(Parameter)

%py_eq_for(NDShape, NDShape_eq)
%py_hash_for(NDShape)

%py_eq_for(Axis, Axis_eq)
%py_hash_for(Axis)

%py_eq_for(DeviceDescriptor, DeviceDescriptor_eq)

%pythoncode %{
StreamInformation.__eq__ = lambda a,b: a.m_name==b.m_name and a.m_id==b.m_id and a.m_storage_format==b.m_storage_format and a.m_element_type==b.m_element_type and a.m_sample_layout.dimensions()==b.m_sample_layout.dimensions()
%}

%extend CNTK::StreamInformation {
    const size_t __hash__() {
        return std::hash<CNTK::StreamInformation>()(*$self);
    }
}

%pythoncode %{
from .tensor import _add_tensor_ops, _add_asarray
for klass in [Function, Variable]:
    _add_tensor_ops(klass)

for klass in [Constant, Parameter, Value, NDArrayView, NDMask, MinibatchData]:
    _add_asarray(klass)

enable_reversing_tensor_shapes_in_error_messages()
%}
