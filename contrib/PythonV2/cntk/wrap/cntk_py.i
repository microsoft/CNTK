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

%rename(NDShape_eq) operator==(const NDShape&, const NDShape&);
%rename(Variable_eq) operator==(const Variable&, const Variable&);

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


// (size_t)-1 will result into an OverflowException
%ignore CNTK::NDShape::InferredDimension;
// BUG: The following is not picked up yet, which is why we have to tag it to
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
            void *argp1 = 0 ;
            int res1 = SWIG_ConvertPtr(key, &argp1, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!argp1) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(argp1);

            void *argp2 = 0;
            int res2 = SWIG_ConvertPtr(value, &argp2, SWIGTYPE_p_std__shared_ptrT_CNTK__Value_t,  0);
            if (!SWIG_IsOK(res2)) {
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert key of dictionary to CNTK::ValuePtr"); 
            }
            if (!argp2) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting value of dictionary to CNTK::ValuePtr");
            }

            CNTK::ValuePtr* value = reinterpret_cast<CNTK::ValuePtr*>(argp2);
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
            void *argp1 = 0 ;
            int res1 = SWIG_ConvertPtr(key, &argp1, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!argp1) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(argp1);

            void *argp2 = 0;
            int res2 = SWIG_ConvertPtr(value, &argp2, SWIGTYPE_p_std__shared_ptrT_CNTK__Value_t,  0);
            if (!SWIG_IsOK(res2)) {
                SWIG_exception_fail(SWIG_ArgError(res2), "cannot convert key of dictionary to CNTK::ValuePtr"); 
            }
            if (!argp2) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting value of dictionary to CNTK::ValuePtr");
            }

            CNTK::ValuePtr* value = reinterpret_cast<CNTK::ValuePtr*>(argp2);
            (*args_map)[*var] = *value;
        }

        $1 = args_map;
     } else {
         SWIG_exception(SWIG_TypeError, "dictionary expected");
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
            SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
        }

        while (item = PyIter_Next(iterator)) {
            void *argp1 = 0 ;
            int res1 = SWIG_ConvertPtr(item, &argp1, SWIGTYPE_p_CNTK__Variable,  0);
            if (!SWIG_IsOK(res1)) {
                SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
            }
            if (!argp1) {
                SWIG_exception_fail(SWIG_ValueError, "invalid null reference when converting key of dictionary to CNTK::Variable");
            }

            CNTK::Variable* var = reinterpret_cast<CNTK::Variable*>(argp1);

            args_set->insert(*var);

            Py_DECREF(item);
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) {
            SWIG_exception_fail(SWIG_ArgError(res1), "cannot convert key of dictionary to CNTK::Variable"); 
        }

        $1 = args_set;

     } else {
         SWIG_exception(SWIG_TypeError, "set expected");
     }
}


%shared_ptr(CNTK::Function)
%shared_ptr(CNTK::NDArrayView)
%shared_ptr(CNTK::Value)
%shared_ptr(CNTK::NDMask)
%shared_ptr(CNTK::BackPropState)

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
%template() std::vector<CNTK::Variable>;

/*
%template(FunctionPtr) std::shared_ptr<CNTK::Function>;
%template(NDArrayViewPtr) std::shared_ptr<CNTK::NDArrayView>;
%template(ValuePtr) std::shared_ptr<CNTK::Value>;
%template(NDMaskPtr) std::shared_ptr<CNTK::NDMask>;
%template(BackPropStatePtr) std::shared_ptr<CNTK::BackPropState>;
*/

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

        void* buf = PyArray_DATA(array);

        if (typecode == NPY_FLOAT)
        {
            return new NDArrayView(NDShape(shape), (float*)buf, num_elements, device, readOnly);
        }
        else if (typecode == NPY_DOUBLE)
        {
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

// end of NDArrayView

%inline %{
    std::shared_ptr<CNTK::NDArrayView> MakeNDArrayViewPtr(CNTK::NDArrayView* view) 
    {
        return std::shared_ptr<CNTK::NDArrayView>(view);
    }

    std::shared_ptr<CNTK::Value> MakeValuePtr(CNTK::Value* value) 
    {
        return std::shared_ptr<CNTK::Value>(value);
    }
%}


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

%pythoncode %{
Variable.__eq__ = lambda a,b: Variable_eq(a,b)
%}

%extend CNTK::Variable {
    const size_t __hash__() {
        return std::hash<CNTK::Variable>()(*$self);
    }
}
