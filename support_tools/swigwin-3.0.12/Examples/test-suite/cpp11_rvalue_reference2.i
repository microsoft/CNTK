%module cpp11_rvalue_reference2

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK) globalrrval;

// This testcase tests lots of different places that rvalue reference syntax can be used

%typemap(in) Something && "/*in Something && typemap*/"
%rename(OperatorRValue) Thingy::operator int&&;
%rename(memberFnRenamed) memberFn(short &&i);
%feature("compactdefaultargs") Thingy::compactDefaultArgs(const bool &&b = (const bool &&)PublicGlobalTrue, const UserDef &&u  = (const UserDef &&)PublicUserDef);
%feature("exception") Thingy::privateDefaultArgs(const bool &&b = (const bool &&)PrivateTrue);
%ignore Thingy::operator=;

%inline %{
#include <utility>
struct UserDef {
  int a;
};
static const bool PublicGlobalTrue = true;
static const UserDef PublicUserDef = UserDef();
struct Thingy {
  typedef int Integer;
  int val;
  int &lvalref;
  int &&rvalref;
  Thingy(int v) : val(v), lvalref(val), rvalref(22) {}
  void refIn(long &i) {}
  void rvalueIn(long &&i) {}
  short && rvalueInOut(short &&i) { return std::move(i); }
  static short && staticRvalueInOut(short &&i) { return std::move(i); }
  // test both primitive and user defined rvalue reference default arguments and compactdefaultargs
  void compactDefaultArgs(const bool &&b = (const bool &&)PublicGlobalTrue, const UserDef &&u  = (const UserDef &&)PublicUserDef) {}
  void privateDefaultArgs(const bool &&b = (const bool &&)PrivateTrue) {}
  operator int &&() { return std::move(0); }
  Thingy(const Thingy& rhs) : val(rhs.val), lvalref(rhs.lvalref), rvalref(copy_int(rhs.rvalref)) {}
  Thingy& operator=(const Thingy& rhs) {
    val = rhs.val;
    lvalref = rhs.lvalref;
    rvalref = rhs.rvalref;
    return *this;
  }
private:
  static const bool PrivateTrue;
  int copy_int(int& i) { return i; }
  Thingy();
};
const bool Thingy::PrivateTrue = true;

short && globalRvalueInOut(short &&i) { return std::move(i); }

Thingy &&globalrrval = Thingy(55);

short && func(short &&i) { return std::move(i); }
Thingy getit() { return Thingy(22); }

void rvalrefFunction1(int &&v = (int &&)5) {}
void rvalrefFunctionBYVAL(short (Thingy::*memFunc)(short)) {}
void rvalrefFunctionLVALUE(short &(Thingy::*memFunc)(short &)) {}
void rvalrefFunction2(short && (Thingy::*memFunc)(short &&)) {}
void rvalrefFunction3(short && (*memFunc)(short &&)) {}

template <typename T> struct RemoveReference {
     typedef T type;
};
 
template <typename T> struct RemoveReference<T&> {
     typedef T type;
};
 
template <typename T> struct RemoveReference<T&&> {
     typedef T type;
};
 
template <> struct RemoveReference<short &&> {
     typedef short type;
};
 
// like std::move
template <typename T> typename RemoveReference<T>::type&& Move(T&& t) {
    return static_cast<typename RemoveReference<T>::type&&>(t);
}
%}

%template(RemoveReferenceDouble) RemoveReference<double &&>;
%template(RemoveReferenceFloat) RemoveReference<float &&>;
%template(RemoveReferenceShort) RemoveReference<short &&>;
%template(MoveFloat) Move<float>;


