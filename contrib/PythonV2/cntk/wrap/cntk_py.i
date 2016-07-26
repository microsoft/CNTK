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

%template() std::vector<size_t>;
%template() std::vector<CNTK::Variable>;

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
    import_array();
%}

%apply (float* OUT_ARRAY1, int DIM1) {(float* py_data, int len)}

%define %eq_for(DATA_TYPE, EQ)
%rename(EQ) operator==(const DATA_TYPE&, const DATA_TYPE&);
%enddef

%extend CNTK::Variable {
    const size_t __hash__() {
        return std::hash<CNTK::Variable>()(*$self);
    }
}


%eq_for(Variable, Variable_eq)
%eq_for(Constant, Variable_eq)
%eq_for(Placeholder, Variable_eq)
%eq_for(Parameter, Variable_eq)


%{
    #include "CNTKLibrary.h"
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

%template() std::vector<CNTK::Variable>;

//%attribute2(CNTK::Variable, CNTK::NDShape, shape, Shape);

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
        size_t num_axes = PyTuple_Size($input);
        for (int i=0; i<num_axes; i++)
            dimensions.push_back(PyLong_AsLong(PyTuple_GET_ITEM($input, i)));

        // TODO cleans this up?
        $1 = new CNTK::NDShape(dimensions);
     } else {
         SWIG_exception(SWIG_TypeError, "tuple expected");
     }
}

%typemap(out) CNTK::NDShape {
    size_t num_axes = $1.NumAxes();
    $result = PyTuple_New(num_axes);
    for (int i=0; i<num_axes; i++)
    {
        size_t dim = (&$1)->operator[](i);
        PyTuple_SET_ITEM($result, i, PyInt_FromLong(dim));
    }
}

%extend CNTK::NDShape {
    const size_t& __getitem__(size_t i) {
        return (*self)[i];
    }
}

%ignore CNTK::NDShape::operator[];
%ignore CNTK::NDShape::AppendShape;
%ignore CNTK::Dictionary::AppendShape;

// (size_t)-1 will result into an OverflowException
%ignore CNTK::NDShape::InferredDimension;
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

%typemap(in) const std::unordered_map<CNTK::Variable, const CNTK::ValuePtr>& {
     if (PyDict_Check($input)) {
        std::unordered_map<CNTK::Variable, const CNTK::ValuePtr>* args_map = new std::unordered_map<CNTK::Variable, const CNTK::ValuePtr>();

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
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert key of dictionary to CNTK::ValuePtr"); 
            }

            CNTK::ValuePtr* value;
            if (raw_value) {
                value = reinterpret_cast<CNTK::ValuePtr*>(raw_value);
            } else {
                // We got an empty ValuePtr, which carries a nullptr.
                value = new CNTK::ValuePtr();
            }

            args_map->insert(std::make_pair(*var, *value));
        }

        $1 = args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

// supporting the non-const version
%typemap(in) std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& {
     if (PyDict_Check($input)) {
        std::unordered_map<CNTK::Variable, CNTK::ValuePtr>* args_map = new std::unordered_map<CNTK::Variable, CNTK::ValuePtr>();

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
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert key of dictionary to CNTK::ValuePtr"); 
            }

            CNTK::ValuePtr* value;
            if (raw_value) {
                value = reinterpret_cast<CNTK::ValuePtr*>(raw_value);
            } else {
                // We got an empty ValuePtr, which carries a nullptr.
                value = new CNTK::ValuePtr();
            }

            args_map->insert(std::make_pair(*var, *value));
        }

        $1 = args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }
}

// For the output dict (the non-const unordered_map) we need to get the
// modified values and put them back into the dictionary. This is used, when
// e.g. the user puts a variable into the dictionary, hoping that it will
// afterwards point to the proper value.
%typemap(argout) std::unordered_map<CNTK::Variable, CNTK::ValuePtr>& {
     if (!PyDict_Check($input)) {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
     }

     for (auto it: *$1)
     {
        // Convert ValuePtr to PyObject
        PyObject *returned_var = SWIG_NewPointerObj(SWIG_as_voidptr(&it.first), SWIGTYPE_p_CNTK__Variable, SWIG_POINTER_NOSHADOW);

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
                // FIXME is this necessary?
                Py_INCREF(returned_val);
            }
        }
    }
}

// end of map conversion

//
// Converting Python set {Variable} to std::unordered_set
//
%typecheck(1000) std::unordered_set<CNTK::Variable>& {
    // '1000' is the typecheck precedence code. It means: check after basic
    // types, but before arrays. See: http://www.swig.org/Doc1.3/Typemaps.html#Typemaps_overloading
    $1 = PySet_Check($input) ? 1 : 0;
}

%typemap(in) std::unordered_set<CNTK::Variable>& {
     if (PySet_Check($input)) {
        std::unordered_set<CNTK::Variable>* args_set = new std::unordered_set<CNTK::Variable>();

        PyObject *item;
        Py_ssize_t pos = 0;

        PyObject *iterator = PyObject_GetIter($input);
        if (iterator == NULL) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert key of dictionary to CNTK::Variable"); 
        }

        while (item = PyIter_Next(iterator)) {
            void *raw_var = 0 ;
            int res1 = SWIG_ConvertPtr(item, &raw_var, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!raw_var) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(raw_var);

            args_set->insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ValueError, "cannot convert key of dictionary to CNTK::Variable"); 
        }

        $1 = args_set;

     } else {
         SWIG_exception(SWIG_ValueError, "set expected");
     }
}

//
// Converting std::unordered_set to Python list.
// TOOD: figure out how to return a Python set instead of a list. For this,
// we need to define a hash function on SwigPyObject.
//

%define %unordered_set_conversion(DATA_TYPE, _SWIG_TYPE)

%typemap(out) std::unordered_set<CNTK::DATA_TYPE> {
    PyObject* container = PyList_New(NULL);
    if (container == NULL)
    {
        SWIG_exception(SWIG_RuntimeError, "error passing set to Python");
    }
 
    // *&$1 -> $1 is the returned result being converted (unordered_set<...>*),
    // wrapped by SwigValueWrapper. So we need to unwrap it using '&', 
    // then access its value using '*'.
    for (auto var : *&$1)
    {
        PyObject *item = SWIG_NewPointerObj(SWIG_as_voidptr(new CNTK::DATA_TYPE(var)), _SWIG_TYPE, SWIG_POINTER_NEW);
        // No error handling here, because the error will be passed directly to Python
        PyList_Append(container, item);
    }

    Py_INCREF(container);

    $result = container;
}
%enddef
 
%unordered_set_conversion(Variable, SWIGTYPE_p_CNTK__Variable)
%unordered_set_conversion(Constant, SWIGTYPE_p_CNTK__Constant)
%unordered_set_conversion(Placeholder, SWIGTYPE_p_CNTK__Placeholder)
%unordered_set_conversion(Parameter, SWIGTYPE_p_CNTK__Parameter)

%shared_ptr(CNTK::Function)
%shared_ptr(CNTK::NDArrayView)
%shared_ptr(CNTK::Value)
%shared_ptr(CNTK::NDMask)
%shared_ptr(CNTK::BackPropState)

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
%template() std::vector<CNTK::Variable>;

//
// NDMask
//
// FIXME ignore is ignored
%ignore CNTK::NDMask::DataBuffer();
%extend CNTK::NDMask {
    PyObject* ToNumPy() {
        std::vector<size_t> cntk_dims = (*self).Shape().Dimensions();
        static_assert(dims.size()==2, "mask requires exactly two dimensions");
        std::vector<size_t> dimensions = {cntk_dims[1], cntk_dims[0]};

        npy_intp* shape = reinterpret_cast<npy_intp*>(&dimensions[0]);

        void* buffer = const_cast<void*>(reinterpret_cast<const void*>((*self).DataBuffer()));
        
        PyObject* ndarray = PyArray_SimpleNewFromData(dimensions.size(), shape, NPY_UBYTE, buffer);

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

        int num_axes = PyArray_NDIM(array); 
        if (num_axes==0)
            throw std::logic_error("provided array is empty");
        
        npy_intp* np_shape = PyArray_SHAPE(array); 
        std::vector<size_t> shape;

        npy_intp num_elements = 1;
        for (int i=0; i<num_axes; i++)
        {
            shape.push_back(np_shape[i]);
            num_elements *= np_shape[i];
        }

        int typecode = PyArray_TYPE(array);
        
        void* buf;

        if (typecode == NPY_FLOAT)
        {
            size_t num_bytes = num_elements * sizeof(float);
            buf = malloc(num_bytes);
            memcpy(buf, PyArray_DATA(array), num_bytes);
            return new NDArrayView(NDShape(shape), (float*)buf, num_elements, device, readOnly);
        }
        else if (typecode == NPY_DOUBLE)
        {
            size_t num_bytes = num_elements * sizeof(double);
            buf = malloc(num_bytes);
            memcpy(buf, PyArray_DATA(array), num_bytes);
            return new NDArrayView(NDShape(shape), (double*)buf, num_elements, device, readOnly);
        }
        else
        {
            throw std::logic_error("NumPy array of type float32 or float64 expected");
        }
    }

    PyObject* ToNumPy() {
        // FIXME use not yet existing NDShape function that returns the dimensions at once
        std::vector<size_t> dimensions = (*self).Shape().Dimensions();
        npy_intp* shape = reinterpret_cast<npy_intp*>(&dimensions[0]);

        NPY_TYPES numpy_type;
        void* buffer;

        CNTK::DataType cntk_type = (*self).GetDataType();
        if (cntk_type == CNTK::DataType::Float)
        {
            numpy_type = NPY_FLOAT;
            buffer = (void*)(*self).DataBuffer<float>();
        }
        else if (cntk_type == CNTK::DataType::Double)
        {
            numpy_type = NPY_DOUBLE;
            buffer = (void*)(*self).DataBuffer<double>();
        }
        else
        {
            throw std::invalid_argument("unknown CNTK data type");
        }
        
        PyObject* ndarray = PyArray_SimpleNewFromData(dimensions.size(), shape, numpy_type, buffer);

        return ndarray;
    }
}

%template(NDArrayViewFloat) CNTK::NDArrayView::NDArrayView<float>;
%template(NDArrayViewDouble) CNTK::NDArrayView::NDArrayView<double>;
%template(ConstantFloat) CNTK::Constant::Constant<float>;
%template(ConstantDouble) CNTK::Constant::Constant<double>;
%template(RandomUniformFloat) CNTK::NDArrayView::RandomUniform<float>;
%template(RandomUniformDouble) CNTK::NDArrayView::RandomUniform<double>;

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

%pythoncode %{
DATA_TYPE.__eq__ = lambda a,b: EQ(a,b)
%}

%extend CNTK::DATA_TYPE {
    const size_t __hash__() {
        return std::hash<CNTK::DATA_TYPE>()(*$self);
    }
}
%enddef

%py_hash_for(Variable, Variable_eq)
%py_hash_for(Constant, Variable_eq)
%py_hash_for(Placeholder, Variable_eq)
%py_hash_for(Parameter, Variable_eq)

