%module li_boost_shared_ptr_attribute

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGPYTHON) || defined(SWIGD) || defined(SWIGOCTAVE) || defined(SWIGRUBY)
#define SHARED_PTR_WRAPPERS_IMPLEMENTED
#endif

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%include "attribute.i"
%include "boost_shared_ptr.i"

%inline %{
#include <boost/shared_ptr.hpp>
using namespace boost;
%}
%shared_ptr(GetMe);
%shared_ptr(GetSetMe);
%attributestring(GetterOnly, shared_ptr<GetMe>, AddedAttrib, GetIt)
%attributestring(GetterSetter, shared_ptr<GetSetMe>, AddedAttrib, GetIt, SetIt)

%inline %{
struct GetMe {
    explicit GetMe(int n) : n(n) {}
    ~GetMe() {}
    int n;
};
struct GetSetMe {
    explicit GetSetMe(int n) : n(n) {}
    ~GetSetMe() {}
    int n;
};

struct GetterOnly {
    explicit GetterOnly(int n) : myval(new GetMe(n*n)) {}
    shared_ptr<GetMe> GetIt() const { return myval; }
    shared_ptr<GetMe> myval;
};
struct GetterSetter {
    explicit GetterSetter(int n) : myval(new GetSetMe(n*n)) {}
    shared_ptr<GetSetMe> GetIt() const { return myval; }
    void SetIt(shared_ptr<GetSetMe> newval) { myval = newval; }
    shared_ptr<GetSetMe> myval;
};
%}

#endif
