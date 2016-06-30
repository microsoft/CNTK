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

// TODO [wilrich] Add support for unordered containers
/*%include "std_unordered_map_vc2013.i"*/
/*%include "std_unordered_set_vc2013.i"*/


%apply (float* ARGOUT_ARRAY1, int DIM1) {(float* data, int len)}

%{
    #include "CNTKLibrary.h"
    using namespace CNTK;
%}

%feature("director") Callback;
%exception {
    try { $action }
    catch (Swig::DirectorException &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::runtime_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::invalid_argument &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (std::logic_error &e) { SWIG_exception(SWIG_RuntimeError,e.what()); }
    catch (...) { SWIG_exception(SWIG_RuntimeError,"Runtime exception"); }
}

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
    %template(DataBufferFloat) DataBuffer<float>;
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

%inline %{

void data_from_value(float* cntk_data, float* data, int len) {
    for (int i=0; i<len; i++)
        data[i] = cntk_data[i];
}

void exception_tester() {
    throw "exc thrown in exception_tester()";
}

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
