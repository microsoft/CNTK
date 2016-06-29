%module swig_cntk

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

/*%include "std_unordered_map_vc2013.i"*/
/*%include "std_unordered_set_vc2013.i"*/


%apply (float* ARGOUT_ARRAY1, int DIM1) {(float* data, int len)}


%{
    #include "CNTKLibrary.h"
    using namespace CNTK;
%}

%feature("ref")   CNTK::_Internal::_ReferenceCounter "$this->AddReference();"
%feature("unref") CNTK::_Internal::_ReferenceCounter "$this->RemoveReference();"

%rename(NDShape_eq) operator==(const NDShape&, const NDShape&);
%rename(Variable_eq) operator==(const Variable&, const Variable&);
%rename(Variable_lt) operator<(const Variable&, const Variable&);

// the following are ignored for whatever reason
%rename(_ReferenceCounterSharedPtr_assign) CNTK::_Internal::_ReferenceCounterSharedPtr::operator=(const CNTK::_Internal::__ReferenceCounterSharedPtr&);
%rename(_ReferenceCounterSharedPtr_move) CNTK::_Internal::_ReferenceCounterSharedPtr::operator=(CNTK::_Internal::__ReferenceCounterSharedPtr&);


%rename(__call__) operator();
%rename(__dereference__) operator*;

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%template(MapVarValuePtr) std::map<CNTK::Variable, CNTK::ValuePtr>;
%template(VarSet) std::set<CNTK::Variable>;


%{
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

size_t bla;
extern void stuff(std::map<CNTK::Variable, CNTK::ValuePtr>& mymap)
{
	bla = 13;

    //return v->Data()->IsSparse();
    printf("size inside=%d", mymap.size());

    for (auto x : mymap)
    {
    printf("in stuff %ls\n", x.first.Name().c_str());
    }
}

extern void naivestuff(std::map<CNTK::Variable, int>& mymap)
{
    printf("naivestuff size inside=%\n", mymap.size());

    for (auto x : mymap)
    {
        printf("in stuff %ls (%d) ->%d\n", x.first.Name().c_str(), x.first.Shape().TotalSize(), x.second);
    }

    printf("naivestuff end\n");
}

void data_from_value(float* cntk_data, float* data, int len) {
    for (int i=0; i<len; i++)
        data[i] = cntk_data[i];
}

%}
