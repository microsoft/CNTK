%module(directors="1") swig_cntk

%include "stl.i"
%include "std_wstring.i" 
%include <std_vector.i>
%include <std_map.i>
%include <std_set.i>
%include <std_pair.i>
%include <windows.i>

%template() std::vector<size_t>;

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

%exception {
    try { $action }
    catch (Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::invalid_argument &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::logic_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (...) { SWIG_exception(SWIG_RuntimeError,"Runtime exception"); }
}

// Reference counting
%feature("ref")   CNTK::_Internal::_ReferenceCounter "$this->AddReference();"
%feature("unref") CNTK::_Internal::_ReferenceCounter "$this->RemoveReference();"

%rename(NDShape_eq) operator==(const NDShape&, const NDShape&);
%rename(Variable_eq) operator==(const Variable&, const Variable&);
%rename(Variable_lt) operator<(const Variable&, const Variable&);

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%template(MapVarValuePtr) std::map<CNTK::Variable, CNTK::ValuePtr>;
%template(VarSet) std::set<CNTK::Variable>;


%{
// [@Amit] FIXME: can we move these behind function calls? 
namespace CNTK {
    /*static*/ Axis Axis::DefaultDynamicAxis = Axis(L"defaultDynamicAxis");
    /*static*/ Axis Axis::BatchAxis = Axis(L"batchAxis");
    /*static*/ Axis Axis::AllAxes = Axis(L"allAxes");
    }
%}

%extend CNTK::NDArrayView {

    NDArrayView(std::vector<size_t> shape, PyObject* pyobj, const CNTK::DeviceDescriptor& device, bool readOnly) 
    {
        if (!PyArray_Check((PyArrayObject*)pyobj))
        {
            // Note that in contrast to numpy.i's implementation we demand NumPy arrays 
            // and do not accept arbitrary sequences, which would needed to be copied around.
            throw std::logic_error("NumPy array expected");
        }

        PyArrayObject* array = (PyArrayObject*)pyobj;

        if (PyArray_NDIM(array) != 1)
        {
            throw std::logic_error("flat array expected");
        }

        int len = (int) PyArray_DIM(array, 0);

        int typecode = PyArray_TYPE(array);

        void* buf = PyArray_DATA(array);

        if (typecode == NPY_FLOAT)
        {
            return new NDArrayView(NDShape(shape), (float*)buf, len, device, readOnly);
        }
        else if (typecode == NPY_DOUBLE)
        {
            return new NDArrayView(NDShape(shape), (double*)buf, len, device, readOnly);
        }
        else
        {
            throw std::logic_error("NumPy array of type float32 or float64 expected");
        }
    }

    PyObject* ToNumPy() {
        // FIXME use not yet existing NDShape function that returns the dimensions at once
        int num_axes = (int)(*self).Shape().NumAxes();
        npy_intp* dims = new npy_intp[num_axes];
        for (int i=0; i<num_axes; i++)
            dims[i] = (*self).Shape()[i];

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
        
        PyObject* ndarray = PyArray_SimpleNewFromData(num_axes, dims, numpy_type, buffer);

        delete[] dims;

        return ndarray;
    }
}

%extend CNTK::NDShape {
    const size_t& __getitem__(size_t i) {
        return (*self)[i];
    }
}

%ignore CNTK::NDShape::operator[](size_t);

%template(FunctionPtr) CNTK::_Internal::_ReferenceCounterSharedPtr<Function>;
%template(NDArrayViewPtr) CNTK::_Internal::_ReferenceCounterSharedPtr<NDArrayView>;
%template(ValuePtr) CNTK::_Internal::_ReferenceCounterSharedPtr<Value>;
%template(NDMaskPtr) CNTK::_Internal::_ReferenceCounterSharedPtr<NDMask>;
%template(BackPropStatePtr) CNTK::_Internal::_ReferenceCounterSharedPtr<BackPropState>;

%template(NDArrayViewFloat) CNTK::NDArrayView::NDArrayView<float>;
%template(NDArrayViewDouble) CNTK::NDArrayView::NDArrayView<double>;

%inline %{

//
// The following callback code is only for testing. Will have to be merged with
// the operator classes.
//
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


// Release the GIL before calling into C++
%exception {
  Py_BEGIN_ALLOW_THREADS;
  $action
  Py_END_ALLOW_THREADS;
}
