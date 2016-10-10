%module(directors="1") cntk_py

%include "stl.i"
%include "std_wstring.i" 
%include <std_vector.i>
%include <std_map.i>
%include <std_set.i>
%include <std_pair.i>
%include <windows.i>
%include <attribute.i>
%include <std_shared_ptr.i>

%rename(output_internal) CNTK::Function::Output;
%rename(replace_placeholders_internal) CNTK::Function::ReplacePlaceholders;
%rename(sgd_learner) CNTK::SGDLearner;
%rename(momentum_sgd_learner) CNTK::MomentumSGDLearner;
%rename(gpu_device) CNTK::DeviceDescriptor::GPUDevice;
%rename(cpu_device) CNTK::DeviceDescriptor::CPUDevice;

// if we don't except RandomUniform the corresponding template functions will not be generated
%rename("%(utitle)s", %$isfunction, notregexmatch$name="RandomUniform") "";
%rename("%(utitle)s", %$isvariable) "";

%template() std::vector<size_t>;
%template() std::vector<bool>;
%template() std::vector<double>;
%template() std::vector<std::vector<size_t>>;
%template() std::vector<std::vector<float>>;
%template() std::vector<std::vector<double>>;

%template() std::vector<CNTK::Variable>;
%template() std::vector<CNTK::Parameter>;
%template() std::vector<CNTK::Constant>;
%template() std::vector<CNTK::Axis>;
%template() std::vector<CNTK::DeviceDescriptor>;
%template() std::vector<CNTK::StreamConfiguration>;
//%template() std::vector<CNTK::DictionaryValue>;
%template() std::vector<std::shared_ptr<CNTK::Function>>;

// They are defined twice under CNTK::Internal and under CNTK namespace
%ignore CNTK::Internal::Combine;
%ignore CNTK::Internal::Where;
%ignore CNTK::Internal::Gather;
%ignore CNTK::Internal::Scatter;
%ignore CNTK::Internal::Slice;

// These aren't exported from the CNTK C++ library
%ignore CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;
%ignore CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;
%ignore CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled;

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

%define %eq_for(DATA_TYPE, EQ)
%rename(EQ) operator==(const DATA_TYPE&, const DATA_TYPE&);
%enddef

%eq_for(Variable, Variable_eq)
%eq_for(Constant, Variable_eq)
%eq_for(Parameter, Variable_eq)
%eq_for(NDShape, NDShape_eq)
%eq_for(DeviceDescriptor, DeviceDescriptor_eq)


%extend CNTK::Dictionary {
    CNTK::DictionaryValue __getitem__(const wchar_t* key) {
        return (*($self))[key];
    }

    void __setitem__(const wchar_t* key, CNTK::DictionaryValue value) {
        (*($self))[key] = value;
    }
}

%extend CNTK::TrainingParameterSchedule<double> {
    const double& __getitem__(size_t sampleCount) {
        return (*($self))[sampleCount];
    }
}


%{
    #include "CNTKLibrary.h"
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include "numpy/ndarraytypes.h"
    #include "numpy/arrayobject.h"
    using namespace CNTK;
%}


// Callback support
%feature("director") Callback;

//
// Exception handling
//
%exception {
    try { $action }
    catch (Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::invalid_argument &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::logic_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (...) { SWIG_exception(SWIG_RuntimeError,"Runtime exception"); }
}


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
        std::vector<size_t> dimensions;
        size_t rank = PyTuple_Size($input);
        for (size_t i=0; i<rank; i++)
            dimensions.push_back(PyLong_AsLong(PyTuple_GET_ITEM($input, i)));

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

%typemap(out) CNTK::NDShape {
    size_t rank = $1.Rank();
    $result = PyTuple_New(rank);
    for (size_t i=0; i<rank; i++)
    {
        size_t dim = (&$1)->operator[](i);
        PyTuple_SET_ITEM($result, i, PyInt_FromLong(dim));
    }
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
            size_t dim = dims[i];
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
%typecheck(1000) const std::unordered_map<CNTK::Variable, const CNTK::ValuePtr>&, std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyDict_Check($input) ? 1 : 0;
}

%typemap(in) const std::unordered_map<CNTK::Variable, const CNTK::ValuePtr>& (
        std::unordered_map<CNTK::Variable, const CNTK::ValuePtr> args_map
) {
     if (PyDict_Check($input)) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next($input, &pos, &key, &value)) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(key, &raw_var, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            void *raw_value = 0;
            int res2 = SWIG_ConvertPtr(value, &raw_value, SWIGTYPE_p_std__shared_ptrT_CNTK__Value_t,  0);
            if (!SWIG_IsOK(res2)) {
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert value of dictionary to CNTK::ValuePtr"); 
            }

            CNTK::ValuePtr* value;
            if (raw_value) {
                value = reinterpret_cast<CNTK::ValuePtr*>(raw_value);
            } else {
                // We got an empty ValuePtr, which carries a nullptr.
                value = new CNTK::ValuePtr();
            }

            args_map.insert(std::make_pair(*var, *value));
        }

        $1 = &args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

// supporting the non-const version
%typemap(in) std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& (
        std::unordered_map<CNTK::Variable, CNTK::ValuePtr> args_map
) {
     if (PyDict_Check($input)) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next($input, &pos, &key, &value)) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(key, &raw_var, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            void *raw_value = 0;
            int res2 = SWIG_ConvertPtr(value, &raw_value, SWIGTYPE_p_std__shared_ptrT_CNTK__Value_t,  0);
            if (!SWIG_IsOK(res2)) {
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert value of dictionary to CNTK::ValuePtr"); 
            }

            CNTK::ValuePtr* value;
            if (raw_value) {
                value = reinterpret_cast<CNTK::ValuePtr*>(raw_value);
            } else {
                // We got an empty ValuePtr, which carries a nullptr.
                value = new CNTK::ValuePtr();
            }

            args_map.insert(std::make_pair(*var, *value));
        }

        $1 = &args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

// For the output dict (the non-const unordered_map) we need to get the
// modified values and put them back into the dictionary. This is used, when
// e.g. the user puts a variable into the dictionary, hoping that it will
// afterwards point to the proper value.
%typemap(argout) 
    // Swig would create this conversion for the 'const' variants as well, which 
    // we do not want. Therefor, we have to explicitly tell it for which ones it should do it.
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputsToFetch, 
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& outputs,
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& backPropagatedGradientValuesForInputs
    {
     if (!PyDict_Check($input)) {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }

     for (auto it: *$1)
     {
        // Push the ValuePtr onto the heap so that it survives
        std::shared_ptr<CNTK::Value> *smartresult = it.second ? new std::shared_ptr<CNTK::Value>(it.second) : 0;
        PyObject *returned_val = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_CNTK__Value_t, SWIG_POINTER_OWN);

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

        while (PyDict_Next($input, &pos, &py_key, &py_value)) {
            void *cntk_key = 0 ;
            int res = SWIG_ConvertPtr(py_key, &cntk_key, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!cntk_key) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* cntk_var = reinterpret_cast<CNTK::Variable*>(cntk_key);
            if (*cntk_var == *&it.first)
            {
                PyDict_SetItem($input, py_key, returned_val);
            }
        }
    }
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
            int res1 = SWIG_ConvertPtr(key, &raw_var, SWIGTYPE_p_CNTK__StreamInformation,  0);
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
            int res = SWIG_ConvertPtr(key, &raw_var, SWIGTYPE_p_CNTK__StreamInformation,  0);
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
                int res1 = SWIG_ConvertPtr(first, &raw_value1, SWIGTYPE_p_std__shared_ptrT_CNTK__NDArrayView_t,  0);
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
                int res2 = SWIG_ConvertPtr(second, &raw_value2, SWIGTYPE_p_std__shared_ptrT_CNTK__NDArrayView_t,  0);
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

%typemap(argout) std::unordered_map<CNTK::StreamInformation, std::pair<CNTK::NDArrayViewPtr, CNTK::NDArrayViewPtr>>& {
     if (!PyDict_Check($input)) {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }

     for (auto it: *$1)
     {
        // Push onto the heap so that it survives

        std::shared_ptr<CNTK::NDArrayView> *smartresult1 = it.second.first ? new std::shared_ptr<CNTK::NDArrayView>(it.second.first) : 0;
        PyObject *returned_val1 = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult1), SWIGTYPE_p_std__shared_ptrT_CNTK__NDArrayView_t, SWIG_POINTER_OWN);

        std::shared_ptr<CNTK::NDArrayView> *smartresult2 = it.second.second ? new std::shared_ptr<CNTK::NDArrayView>(it.second.second) : 0;
        PyObject *returned_val2 = SWIG_NewPointerObj(SWIG_as_voidptr(smartresult2), SWIGTYPE_p_std__shared_ptrT_CNTK__NDArrayView_t, SWIG_POINTER_OWN);

        // Find the corresponding Variable instance in the Python dictionary
        // and set its value to the new MinibatchData

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

        while (PyDict_Next($input, &pos, &py_key, &py_value)) {
            void *cntk_key = 0 ;
            int res = SWIG_ConvertPtr(py_key, &cntk_key, SWIGTYPE_p_CNTK__StreamInformation,  0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "cannot convert key of dictionary to CNTK::StreamInformation"); 
            }
            if (!cntk_key) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::StreamInformation");
            }

            CNTK::StreamInformation* cntk_var = reinterpret_cast<CNTK::StreamInformation*>(cntk_key);
            if (*cntk_var == *&it.first)
            {
                PyDict_SetItem($input, py_key, PyTuple_Pack(2, returned_val1, returned_val2));
            }
        }
    }
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
     if (PyList_Check($input)) {
        std::vector<CNTK::DictionaryValue>* vec = new std::vector<CNTK::DictionaryValue>();

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::DictionaryValue"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__DictionaryValue,  0);
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
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::LearnerPtr"); 
        }

        $1 = vec;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}

// end of map conversion

// TODO: Parametrize the following four typemaps and unify set/list usage.

//
// Converting Python set {Variable} to std::unordered_set
//
%typecheck(1000) std::unordered_set<CNTK::Variable>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PySet_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<CNTK::Variable>& (
        std::unordered_set<CNTK::Variable> args_set 
) {
     if (PySet_Check($input)) {

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::Variable"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert set element to CNTK::Variable"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a list element to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            args_set.insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert set element to CNTK::Variable"); 
        }

        $1 = &args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "set expected");
     }
}

//
// Converting Python set {StreamInformation} to std::unordered_set
//
%typecheck(1000) std::unordered_set<CNTK::StreamInformation>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PySet_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<CNTK::StreamInformation>& (
        std::unordered_set<CNTK::StreamInformation> args_set 
) {
     if (PySet_Check($input)) {

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::StreamInformation"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__StreamInformation,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert set element to CNTK::StreamInformation"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a set element to CNTK::StreamInformation");
            }

            CNTK::StreamInformation* var = reinterpret_cast<CNTK::StreamInformation*>(raw_var);

            args_set.insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert set element to CNTK::StreamInformation"); 
        }

        $1 = &args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "set expected");
     }
}

//
// Converting Python list {Parameter} to std::unordered_set
//
%typecheck(1000) std::unordered_set<CNTK::Parameter>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<CNTK::Parameter>& (
        std::unordered_set<CNTK::Parameter> args_set 
) {
     if (PyList_Check($input)) {

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::Parameter"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__Parameter,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert set element to CNTK::Parameter"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a list element to CNTK::Parameter");
            }

            CNTK::Parameter* var = reinterpret_cast<CNTK::Parameter*>(raw_var);

            args_set.insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert set element to CNTK::Parameter"); 
        }

        $1 = &args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}


//
// Converting Python list {LearnerPtr} to std::unordered_set
//
%typecheck(1000) std::unordered_set<CNTK::LearnerPtr>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyList_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<CNTK::LearnerPtr>& (
        std::unordered_set<CNTK::LearnerPtr> args_set 
) {
     if (PyList_Check($input)) {

        PyObject *item;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::LearnerPtr"); 
        }

        while ((item = PyIter_Next(iterator))) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_std__shared_ptrT_CNTK__Learner_t,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert list element to CNTK::LearnerPtr"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting a list element to CNTK::LearnerPtr");
            }

            CNTK::LearnerPtr* var = reinterpret_cast<CNTK::LearnerPtr*>(raw_var);

            args_set.insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert list element to CNTK::LearnerPtr"); 
        }

        $1 = &args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "list expected");
     }
}

%typecheck(1000) const std::unordered_map<CNTK::Variable, CNTK::Variable>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PyDict_Check($input) ? 1 : 0;
}


%typemap(in) std::unordered_map<CNTK::Variable, CNTK::Variable>& (
        std::unordered_map<CNTK::Variable, CNTK::Variable> args_map 
) {
     if (PyDict_Check($input)) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next($input, &pos, &key, &value)) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(key, &raw_var, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            void *raw_value = 0;
            int res2 = SWIG_ConvertPtr(value, &raw_value, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res2)) {
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert value of dictionary to CNTK::Variable"); 
            }

            CNTK::Variable* value;
            if (raw_value) {
                value = reinterpret_cast<CNTK::Variable*>(raw_value);
            } else {
                // We got an empty Variable, which carries a nullptr.
                value = new CNTK::Variable();
            }

            args_map.insert(std::make_pair(*var, *value));
        }

        $1 = &args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}



//
// Converting std::unordered_set to Python list.
// TOOD: figure out how to return a Python set instead of a list. For this,
// we need to define a hash function on SwigPyObject.
//

%define %unordered_set_conversion(DATA_TYPE, _SWIG_TYPE)

%typemap(out) std::unordered_set<CNTK::DATA_TYPE> {
    PyObject* container = PyList_New(0);
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing set to Python");
    }
 
    // *&$1 -> $1 is the returned result being converted (unordered_set<...>*),
    // wrapped by SwigValueWrapper. So we need to unwrap it using '&', 
    // then access its value using '*'.
    for (auto var : *&$1)
    {
        PyObject *item = SWIG_NewPointerObj(new CNTK::DATA_TYPE(var), _SWIG_TYPE, SWIG_POINTER_OWN );
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
    }

    $result = container;
}
%enddef
 
%unordered_set_conversion(Variable, SWIGTYPE_p_CNTK__Variable)
%unordered_set_conversion(Constant, SWIGTYPE_p_CNTK__Constant)
%unordered_set_conversion(Parameter, SWIGTYPE_p_CNTK__Parameter)

%define %unordered_set_ref_conversion(DATA_TYPE, _SWIG_TYPE)

%typemap(out) std::unordered_set<CNTK::DATA_TYPE>& {
    PyObject* container = PyList_New(0);
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing set to Python");
    }
     
    for (auto var : *$1)
    {
        PyObject *item = SWIG_NewPointerObj(new CNTK::DATA_TYPE(var), _SWIG_TYPE, SWIG_POINTER_OWN );
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
    }

    $result = container;
}
%enddef

%unordered_set_ref_conversion(StreamInformation, SWIGTYPE_p_CNTK__StreamInformation)
%unordered_set_ref_conversion(LearnerPtr, SWIGTYPE_p_std__shared_ptrT_CNTK__Learner_t)
%unordered_set_ref_conversion(Parameter, SWIGTYPE_p_CNTK__Parameter)

// Unordered map conversion

%define %unordered_map_ref_conversion(DATA_TYPE1, _SWIG_TYPE1, DATA_TYPE2, _SWIG_TYPE2)

%typemap(out) std::unordered_map<CNTK::DATA_TYPE1, CNTK::DATA_TYPE2>& {
    PyObject* container = PyDict_New();
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing dictionary to Python");
    }
 
    // *&$1 -> $1 is the returned result being converted (unordered_map<...>*),
    // wrapped by SwigValueWrapper. So we need to unwrap it using '&', 
    // then access its value using '*'.
    for (auto it : *$1)
    {        
        PyObject *returned_var = SWIG_NewPointerObj(SWIG_as_voidptr(new CNTK::DATA_TYPE1(it.first)), _SWIG_TYPE1, SWIG_POINTER_OWN);
        PyObject *returned_val = SWIG_NewPointerObj(SWIG_as_voidptr(new CNTK::DATA_TYPE2(it.second)), _SWIG_TYPE2, SWIG_POINTER_OWN);
        
        PyDict_SetItem(container, returned_var, returned_val);        
    }

    $result = container;
}
%enddef

%unordered_map_ref_conversion(StreamInformation, SWIGTYPE_p_CNTK__StreamInformation, MinibatchData, SWIGTYPE_p_CNTK__MinibatchData);


%shared_ptr(CNTK::Function)
%shared_ptr(CNTK::NDArrayView)
%shared_ptr(CNTK::Value)
%shared_ptr(CNTK::NDMask)
%shared_ptr(CNTK::BackPropState)
%shared_ptr(CNTK::Learner)
%shared_ptr(CNTK::MinibatchSource)

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"


//
// NDMask
//
// FIXME ignore is ignored
%ignore CNTK::NDMask::DataBuffer();
%extend CNTK::NDMask {
    PyObject* to_numpy() {
        std::vector<size_t> cntk_dims = (*self).Shape().Dimensions();
        static_assert(dims.size()==2, "mask requires exactly two dimensions");
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
        
        PyObject* ndarray = PyArray_SimpleNew(dimensions.size(), shape, NPY_BYTE);
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

//
// NDArrayView
//
%extend CNTK::NDArrayView {

    NDArrayView(PyObject* pyobj, const CNTK::DeviceDescriptor& device, bool readOnly) 
    {
        if (!PyArray_Check((PyArrayObject*)pyobj))
        {
            // Note that in contrast to numpy.i's implementation we demand NumPy arrays 
            // and do not accept arbitrary sequences, which would needed to be copied around.
            throw std::logic_error("NumPy array expected");
        }

        PyArrayObject* array = (PyArrayObject*)pyobj;

        int rank = PyArray_NDIM(array); 
        
        npy_intp* np_shape = PyArray_SHAPE(array); 
        std::vector<size_t> shape;

        npy_intp num_elements = 1;
        // CNTK uses column major, thus we reverse the shape
        for (int i=rank-1; i>=0; i--)
        {
            shape.push_back(np_shape[i]);
            num_elements *= np_shape[i];            
        }

        int typecode = PyArray_TYPE(array);
        
        NDArrayView* view;
        if (typecode == NPY_FLOAT)
        {
            NDArrayView  tmp(NDShape(shape), (float*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
            view = new NDArrayView(DataType::Float, tmp.Shape(), device);
            view->CopyFrom(tmp);
        }
        else if (typecode == NPY_DOUBLE)
        {
            NDArrayView  tmp(NDShape(shape), (double*)PyArray_DATA(array), num_elements, DeviceDescriptor::CPUDevice(), readOnly);
            view = new NDArrayView(DataType::Double, tmp.Shape(), device);
            view->CopyFrom(tmp);
        }
        else
        {
            throw std::logic_error("NumPy array of type float32 or float64 expected");
        }

        return view;
    }

    PyObject* to_numpy() {
        if ((*self).GetStorageFormat() != StorageFormat::Dense)
            throw std::invalid_argument("only dense supported at the moment");

        // FIXME use not yet existing NDShape function that returns the dimensions at once
        std::vector<size_t> dimensions_cntk = (*self).Shape().Dimensions();
        std::vector<size_t> dimensions;

        // We have at least one element. In case the shape is empty (e.g.
        // '()'), we have a scalar, which we need to copy (e.g. a constant).
        size_t num_elements = 1;

        // CNTK uses column major, thus we reverse the shape
        for (int i=dimensions_cntk.size()-1; i>=0; i--)
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
            cpuView = &(*self);
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

        PyObject* ndarray = PyArray_SimpleNew(dimensions.size(), shape, numpy_type);
        void *arr_data = PyArray_DATA((PyArrayObject*)ndarray);

        memcpy(arr_data, buffer, PyArray_ITEMSIZE((PyArrayObject*) ndarray) * num_elements);

        if ((*self).Device() != DeviceDescriptor::CPUDevice())
        {
            delete cpuView;
        }

        return ndarray;
    }
}

%template(NDArrayViewFloat) CNTK::NDArrayView::NDArrayView<float>;
%template(NDArrayViewDouble) CNTK::NDArrayView::NDArrayView<double>;
%template(ConstantFloat) CNTK::Constant::Constant<float>;
%template(ConstantDouble) CNTK::Constant::Constant<double>;
%template(ParameterFloat) CNTK::Parameter::Parameter<float>;
%template(ParameterDouble) CNTK::Parameter::Parameter<double>;
%template(random_uniform_float) CNTK::NDArrayView::RandomUniform<float>;
%template(random_uniform_double) CNTK::NDArrayView::RandomUniform<double>;
%template(DictionaryValueFromDict) CNTK::DictionaryValue::DictionaryValue<CNTK::Dictionary>;

%template(training_param_schedule_double) CNTK::TrainingParameterSchedule<double>;

%pythoncode %{
learning_rates_per_sample = training_param_schedule_double
momentums_per_sample = training_param_schedule_double
%}
        
// end of NDArrayView

//
// The following callback code is only for testing. Will have to be merged with
// the operator classes.
//
%inline %{
class Callback {
public:
    virtual ~Callback() { std::cout << "Callback::~Callback()" << std:: endl; }
    virtual void forward() { std::cout << "Callback::forward()" << std::endl; }
    virtual void backward() { std::cout << "Callback::backward()" << std::endl; }
};

class FunctionInCNTK {
private:
    Callback *_callback;
public:
    FunctionInCNTK(): _callback(0) {}
    ~FunctionInCNTK() { delCallback(); }
    void delCallback() { delete _callback; _callback = 0; }
    void setCallback(Callback *cb) { delCallback(); _callback = cb; }
    void forward() { 
        if (_callback) 
            _callback->forward(); 
        else
            throw "Forward callback not defined!";
    }
    void backward() { if (_callback) _callback->backward(); }
};
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
%define %py_hash_for(DATA_TYPE, EQ)
%extend CNTK::DATA_TYPE {
    const size_t __hash__() {
        return std::hash<CNTK::DATA_TYPE>()(*$self);
    }
}
%enddef

%define %py_eq_for(DATA_TYPE, EQ)
%pythoncode %{
DATA_TYPE.__eq__ = lambda a,b: EQ(a,b)
%}
%enddef

%py_eq_for(Variable, Variable_eq)
%py_eq_for(Constant, Variable_eq)
%py_eq_for(Parameter, Variable_eq)
%py_eq_for(NDShape, NDShape_eq)

%py_hash_for(Variable, Variable_eq)
%py_hash_for(Constant, Variable_eq)
%py_hash_for(Parameter, Variable_eq)
%py_hash_for(NDShape, NDShape_eq)

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
# in case of multiple outputs return the function, not the variable
def get_output_and_keep_reference(self):
    variable = self.output_internal()    
    variable.owner = self
    return variable
Function.output = lambda self:get_output_and_keep_reference(self)
Function.replace_placeholders = lambda self, ph_map: self.replace_placeholders_internal(ph_map)

from .tensor import _add_tensor_ops, _add_eval
for klass in [Function, Variable]:
    _add_tensor_ops(klass)

_add_eval(Function)

enable_reversing_tensor_shapes_in_error_messages()
%}

