// This tests intrusive_ptr is working okay. It also checks that there are no memory leaks in the
// class that intrusive_ptr is pointing via a counting mechanism in the constructors and destructor of Klass.
// In order to test that there are no leaks of the intrusive_ptr class itself (as it is created on the heap)
// the runtime tests can be run for a long time to monitor memory leaks using memory monitor tools 
// like 'top'. There is a wrapper for intrusive_ptr in intrusive_ptr_wrapper.h which enables one to
// count the instances of intrusive_ptr. Uncomment the INTRUSIVE_PTR_WRAPPER macro to turn this on.
//
// Also note the debug_shared flag  which can be set from the target language.
//
// Usage of intrusive_ptr_add_ref and intrusive_ptr_release based on boost testing:
// http://www.boost.org/doc/libs/1_36_0/libs/smart_ptr/test/intrusive_ptr_test.cpp

%module li_boost_intrusive_ptr

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);
%warnfilter(SWIGWARN_LANG_SMARTPTR_MISSING) KlassDerived;
%warnfilter(SWIGWARN_LANG_SMARTPTR_MISSING) KlassDerivedDerived;

%ignore intrusive_ptr_add_ref;
%ignore intrusive_ptr_release;

%{
#include <boost/shared_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

// Uncomment macro below to turn on intrusive_ptr memory leak checking as described above
//#define INTRUSIVE_PTR_WRAPPER

#ifdef INTRUSIVE_PTR_WRAPPER
# include "intrusive_ptr_wrapper.h"
# include "shared_ptr_wrapper.h"
#endif
%}

%{
#ifndef INTRUSIVE_PTR_WRAPPER
# define SwigBoost boost
#endif
%}

%include "std_string.i"
#ifndef INTRUSIVE_PTR_WRAPPER
# define SWIG_INTRUSIVE_PTR_NAMESPACE SwigBoost
# define SWIG_SHARED_PTR_NAMESPACE SwigBoost
#endif

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
#define INTRUSIVE_PTR_WRAPPERS_IMPLEMENTED
#endif

#if defined(INTRUSIVE_PTR_WRAPPERS_IMPLEMENTED)

%include <boost_intrusive_ptr.i>
%intrusive_ptr(Space::Klass)
%intrusive_ptr_no_wrap(Space::KlassWithoutRefCount)
%intrusive_ptr(Space::KlassDerived)
%intrusive_ptr(Space::KlassDerivedDerived)

//For the use_count shared_ptr functions
#if defined(SWIGJAVA)
%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::Klass > & ($*1_ltype tempnull) %{ 
  $1 = $input ? *($&1_ltype)&$input : &tempnull; 
%}
%typemap (jni) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::Klass > & "jlong"
%typemap (jtype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::Klass > & "long"
%typemap (jstype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::Klass > & "Klass"
%typemap(javain) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::Klass > & "Klass.getCPtr($javainput)"
  
%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerived > & ($*1_ltype tempnull) %{ 
  $1 = $input ? *($&1_ltype)&$input : &tempnull; 
%}
%typemap (jni) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerived > & "jlong"
%typemap (jtype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerived > & "long"
%typemap (jstype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerived > & "KlassDerived"
%typemap(javain) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerived > & "KlassDerived.getCPtr($javainput)"
  
%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerivedDerived > & ($*1_ltype tempnull) %{ 
  $1 = $input ? *($&1_ltype)&$input : &tempnull; 
%}
%typemap (jni) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerivedDerived > & "jlong"
%typemap (jtype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerivedDerived > & "long"
%typemap (jstype) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerivedDerived > & "KlassDerivedDerived"
%typemap(javain) SWIG_INTRUSIVE_PTR_QNAMESPACE::shared_ptr< Space::KlassDerivedDerived > & "KlassDerivedDerived.getCPtr($javainput)"

#elif defined(SWIGCSHARP)
// TODO!
#endif

#endif

// TODO:
// const intrusive_ptr
// std::vector
// Add in generic %extend for the Upcast function for derived classes
// Remove proxy upcast method - implement %feature("shadow") ??? which replaces the proxy method

%exception {
  if (debug_shared) {
    cout << "++++++" << endl << flush;
    cout << "calling $name" << endl << flush;
  }
  $action
  if (debug_shared) {
    cout << "------" << endl << flush;
  }
}

%ignore IgnoredRefCountingBase;
%ignore *::operator=;
%newobject pointerownertest();
%newobject smartpointerpointerownertest();
  
%inline %{
#include <iostream>
using namespace std;

static bool debug_shared = false;

namespace Space {

struct Klass {
  Klass() : value("EMPTY"), count(0) { if (debug_shared) cout << "Klass() [" << value << "]" << endl << flush; increment(); }

  Klass(const std::string &val) : value(val), count(0) { if (debug_shared) cout << "Klass(string) [" << value << "]" << endl << flush; increment(); }

  virtual ~Klass() { if (debug_shared) cout << "~Klass() [" << value << "]" << endl << flush; decrement(); }
  virtual std::string getValue() const { return value; }
  void append(const std::string &s) { value += s; }
  Klass(const Klass &other) : value(other.value), count(0) { if (debug_shared) cout << "Klass(const Klass&) [" << value << "]" << endl << flush; increment(); }

  Klass &operator=(const Klass &other) { value = other.value; return *this; }

  void addref(void) const { ++count; }
  void release(void) const { if (--count == 0) delete this; }
  int use_count(void) const { return count; }
  static long getTotal_count() { return total_count; }
  friend void intrusive_ptr_add_ref(const Klass* r) { r->addref(); }
  friend void intrusive_ptr_release(const Klass* r) { r->release(); }

private:
  static void increment() { ++total_count; if (debug_shared) cout << "      ++xxxxx Klass::increment tot: " << total_count << endl;}
  static void decrement() { --total_count; if (debug_shared) cout << "      --xxxxx Klass::decrement tot: " << total_count << endl;}
  static boost::detail::atomic_count total_count;
  std::string value;
  int array[1024];
  mutable boost::detail::atomic_count count;
};

struct KlassWithoutRefCount {
  KlassWithoutRefCount() : value("EMPTY") { if (debug_shared) cout << "KlassWithoutRefCount() [" << value << "]" << endl << flush; increment(); }

  KlassWithoutRefCount(const std::string &val) : value(val) { if (debug_shared) cout << "KlassWithoutRefCount(string) [" << value << "]" << endl << flush; increment(); }

  virtual ~KlassWithoutRefCount() { if (debug_shared) cout << "~KlassWithoutRefCount() [" << value << "]" << endl << flush; decrement(); }
  virtual std::string getValue() const { return value; }
  void append(const std::string &s) { value += s; }
  KlassWithoutRefCount(const KlassWithoutRefCount &other) : value(other.value) { if (debug_shared) cout << "KlassWithoutRefCount(const KlassWithoutRefCount&) [" << value << "]" << endl << flush; increment(); }
  std::string getSpecialValueFromUnwrappableClass() { return "this class cannot be wrapped by intrusive_ptrs but we can still use it"; }
  KlassWithoutRefCount &operator=(const KlassWithoutRefCount &other) { value = other.value; return *this; }
  static long getTotal_count() { return total_count; }

private:
  static void increment() { ++total_count; if (debug_shared) cout << "      ++xxxxx KlassWithoutRefCount::increment tot: " << total_count << endl;}
  static void decrement() { --total_count; if (debug_shared) cout << "      --xxxxx KlassWithoutRefCount::decrement tot: " << total_count << endl;}
  static boost::detail::atomic_count total_count;
  std::string value;
  int array[1024];
};

struct IgnoredRefCountingBase { 
  IgnoredRefCountingBase() : count(0) { if (debug_shared) cout << "IgnoredRefCountingBase()" << endl << flush; increment(); }

  IgnoredRefCountingBase(const IgnoredRefCountingBase &other) : count(0) { if (debug_shared) cout << "IgnoredRefCountingBase(const IgnoredRefCountingBase&)" << endl << flush; increment(); }

  IgnoredRefCountingBase &operator=(const IgnoredRefCountingBase& other) {
    return *this;
  }
 
  virtual ~IgnoredRefCountingBase() { if (debug_shared) cout << "~IgnoredRefCountingBase()" << endl << flush; decrement(); }
  
  void addref(void) const { ++count; }
  void release(void) const { if (--count == 0) delete this; }
  int use_count(void) const { return count; }
  inline friend void intrusive_ptr_add_ref(const IgnoredRefCountingBase* r) { r->addref(); }
  inline friend void intrusive_ptr_release(const IgnoredRefCountingBase* r) { r->release(); }
  static long getTotal_count() { return total_count; }
  
 private:
  static void increment() { ++total_count; if (debug_shared) cout << "      ++xxxxx IgnoredRefCountingBase::increment tot: " << total_count << endl;}
  static void decrement() { --total_count; if (debug_shared) cout << "      --xxxxx IgnoredRefCountingBase::decrement tot: " << total_count << endl;}
  static boost::detail::atomic_count total_count;
  double d; 
  double e;
  mutable boost::detail::atomic_count count;
};

long getTotal_IgnoredRefCountingBase_count() {
  return IgnoredRefCountingBase::getTotal_count();
}

// For most compilers, this use of multiple inheritance results in different derived and base class 
// pointer values ... for some more challenging tests :)
struct KlassDerived : IgnoredRefCountingBase, KlassWithoutRefCount {
  KlassDerived() : KlassWithoutRefCount() { if (debug_shared) cout << "KlassDerived()" << endl << flush; increment(); }
  KlassDerived(const std::string &val) : KlassWithoutRefCount(val) { if (debug_shared) cout << "KlassDerived(string) [" << val << "]" << endl << flush; increment(); }
  KlassDerived(const KlassDerived &other) : KlassWithoutRefCount(other) { if (debug_shared) cout << "KlassDerived(const KlassDerived&))" << endl << flush; increment(); }
  virtual ~KlassDerived() { if (debug_shared) cout << "~KlassDerived()" << endl << flush; decrement(); }
  virtual std::string getValue() const { return KlassWithoutRefCount::getValue() + "-Derived"; }
  int use_count(void) const { return IgnoredRefCountingBase::use_count(); }
  static long getTotal_count() { return total_count; }
  
 private:
  static void increment() { ++total_count; if (debug_shared) cout << "      ++xxxxx KlassDerived::increment tot: " << total_count << endl;}
  static void decrement() { --total_count; if (debug_shared) cout << "      --xxxxx KlassDerived::decrement tot: " << total_count << endl;}
  static boost::detail::atomic_count total_count;
};
struct KlassDerivedDerived : KlassDerived {
  KlassDerivedDerived() : KlassDerived() { if (debug_shared) cout << "KlassDerivedDerived()" << endl << flush; increment(); }
  KlassDerivedDerived(const std::string &val) : KlassDerived(val) { if (debug_shared) cout << "KlassDerivedDerived(string) [" << val << "]" << endl << flush; increment(); }
  KlassDerivedDerived(const KlassDerived &other) : KlassDerived(other) { if (debug_shared) cout << "KlassDerivedDerived(const KlassDerivedDerived&))" << endl << flush; increment(); }
  virtual ~KlassDerivedDerived() { if (debug_shared) cout << "~KlassDerivedDerived()" << endl << flush; decrement(); }
  virtual std::string getValue() const { return KlassWithoutRefCount::getValue() + "-DerivedDerived"; }
  static long getTotal_count() { return total_count; }
  
 private:
  static void increment() { ++total_count; if (debug_shared) cout << "      ++xxxxx KlassDerivedDerived::increment tot: " << total_count << endl;}
  static void decrement() { --total_count; if (debug_shared) cout << "      --xxxxx KlassDerivedDerived::decrement tot: " << total_count << endl;}
  static boost::detail::atomic_count total_count;
};
KlassDerived* derivedpointertest(KlassDerived* kd) {
  if (kd)
    kd->append(" derivedpointertest");
  return kd;
}
KlassDerived derivedvaluetest(KlassDerived kd) {
  kd.append(" derivedvaluetest");
  return kd;
}
KlassDerived& derivedreftest(KlassDerived& kd) {
  kd.append(" derivedreftest");
  return kd;
}
SwigBoost::intrusive_ptr<KlassDerived> derivedsmartptrtest(SwigBoost::intrusive_ptr<KlassDerived> kd) {
  if (kd)
    kd->append(" derivedsmartptrtest");
  return kd;
}
SwigBoost::intrusive_ptr<KlassDerived>* derivedsmartptrpointertest(SwigBoost::intrusive_ptr<KlassDerived>* kd) {
  if (kd && *kd)
    (*kd)->append(" derivedsmartptrpointertest");
  return kd;
}
SwigBoost::intrusive_ptr<KlassDerived>* derivedsmartptrreftest(SwigBoost::intrusive_ptr<KlassDerived>* kd) {
  if (kd && *kd)
    (*kd)->append(" derivedsmartptrreftest");
  return kd;
}
SwigBoost::intrusive_ptr<KlassDerived>*& derivedsmartptrpointerreftest(SwigBoost::intrusive_ptr<KlassDerived>*& kd) {
  if (kd && *kd)
    (*kd)->append(" derivedsmartptrpointerreftest");
  return kd;
}

SwigBoost::intrusive_ptr<Klass> factorycreate() {
  return SwigBoost::intrusive_ptr<Klass>(new Klass("factorycreate"));
}
// smart pointer
SwigBoost::intrusive_ptr<Klass> smartpointertest(SwigBoost::intrusive_ptr<Klass> k) {
  if (k)
    k->append(" smartpointertest");
  return SwigBoost::intrusive_ptr<Klass>(k);
}
SwigBoost::intrusive_ptr<Klass>* smartpointerpointertest(SwigBoost::intrusive_ptr<Klass>* k) {
  if (k && *k)
    (*k)->append(" smartpointerpointertest");
  return k;
}
SwigBoost::intrusive_ptr<Klass>& smartpointerreftest(SwigBoost::intrusive_ptr<Klass>& k) {
  if (k)
    k->append(" smartpointerreftest");
  return k;
}
SwigBoost::intrusive_ptr<Klass>*& smartpointerpointerreftest(SwigBoost::intrusive_ptr<Klass>*& k) {
  if (k && *k)
    (*k)->append(" smartpointerpointerreftest");
  return k;
}
// const
SwigBoost::intrusive_ptr<const Klass> constsmartpointertest(SwigBoost::intrusive_ptr<const Klass> k) {
  return SwigBoost::intrusive_ptr<const Klass>(k);
}
SwigBoost::intrusive_ptr<const Klass>* constsmartpointerpointertest(SwigBoost::intrusive_ptr<const Klass>* k) {
  return k;
}
SwigBoost::intrusive_ptr<const Klass>& constsmartpointerreftest(SwigBoost::intrusive_ptr<const Klass>& k) {
  return k;
}
// plain pointer
Klass valuetest(Klass k) {
  k.append(" valuetest");
  return k;
}
Klass *pointertest(Klass *k) {
  if (k)
    k->append(" pointertest");
  return k;
}
Klass& reftest(Klass& k) {
  k.append(" reftest");
  return k;
}
Klass *const& pointerreftest(Klass *const& k) {
  k->append(" pointerreftest");
  return k;
}
// null
std::string nullsmartpointerpointertest(SwigBoost::intrusive_ptr<Klass>* k) {
  if (k && *k)
    return "not null";
  else if (!k)
    return "null smartpointer pointer";
  else if (!*k)
    return "null pointer";
  else
    return "also not null";
}
// $owner
Klass *pointerownertest() {
  return new Klass("pointerownertest");
}
SwigBoost::intrusive_ptr<Klass>* smartpointerpointerownertest() {
  return new SwigBoost::intrusive_ptr<Klass>(new Klass("smartpointerpointerownertest"));
}

const SwigBoost::intrusive_ptr<Klass>& ref_1() { 
  static SwigBoost::intrusive_ptr<Klass> sptr;
  return sptr;
}

// overloading tests
std::string overload_rawbyval(int i) { return "int"; }
std::string overload_rawbyval(Klass k) { return "rawbyval"; }

std::string overload_rawbyref(int i) { return "int"; }
std::string overload_rawbyref(Klass &k) { return "rawbyref"; }

std::string overload_rawbyptr(int i) { return "int"; }
std::string overload_rawbyptr(Klass *k) { return "rawbyptr"; }

std::string overload_rawbyptrref(int i) { return "int"; }
std::string overload_rawbyptrref(Klass *const&k) { return "rawbyptrref"; }



std::string overload_smartbyval(int i) { return "int"; }
std::string overload_smartbyval(SwigBoost::intrusive_ptr<Klass> k) { return "smartbyval"; }

std::string overload_smartbyref(int i) { return "int"; }
std::string overload_smartbyref(SwigBoost::intrusive_ptr<Klass> &k) { return "smartbyref"; }

std::string overload_smartbyptr(int i) { return "int"; }
std::string overload_smartbyptr(SwigBoost::intrusive_ptr<Klass> *k) { return "smartbyptr"; }

std::string overload_smartbyptrref(int i) { return "int"; }
std::string overload_smartbyptrref(SwigBoost::intrusive_ptr<Klass> *&k) { return "smartbyptrref"; }

} // namespace Space

%}
%{
  boost::detail::atomic_count Space::Klass::total_count(0);
  boost::detail::atomic_count Space::KlassWithoutRefCount::total_count(0);
  boost::detail::atomic_count Space::IgnoredRefCountingBase::total_count(0);
  boost::detail::atomic_count Space::KlassDerived::total_count(0);
  boost::detail::atomic_count Space::KlassDerivedDerived::total_count(0);
%}

// Member variables

%inline %{
struct MemberVariables {
  MemberVariables() : SmartMemberPointer(new SwigBoost::intrusive_ptr<Space::Klass>()), SmartMemberReference(*(new SwigBoost::intrusive_ptr<Space::Klass>())), MemberPointer(0), MemberReference(MemberValue) {}
  virtual ~MemberVariables() {
    delete SmartMemberPointer;
    delete &SmartMemberReference;
  }
  SwigBoost::intrusive_ptr<Space::Klass> SmartMemberValue;
  SwigBoost::intrusive_ptr<Space::Klass> * SmartMemberPointer;
  SwigBoost::intrusive_ptr<Space::Klass> & SmartMemberReference;
  Space::Klass MemberValue;
  Space::Klass * MemberPointer;
  Space::Klass & MemberReference;
};

// Global variables
SwigBoost::intrusive_ptr<Space::Klass> GlobalSmartValue;
Space::Klass GlobalValue;
Space::Klass * GlobalPointer = 0;
Space::Klass & GlobalReference = GlobalValue;

%}

#if defined(INTRUSIVE_PTR_WRAPPERS_IMPLEMENTED)

// Note: %template after the intrusive_ptr typemaps
%intrusive_ptr(Base<int, double>)
%intrusive_ptr(Pair<int, double>)

#endif

// Templates
%inline %{
template <class T1, class T2> struct Base {
  Space::Klass klassBase;
  T1 baseVal1;
  T2 baseVal2;
  Base(T1 t1, T2 t2) : baseVal1(t1*2), baseVal2(t2*2) {}
  virtual std::string getValue() const { return "Base<>"; };
  mutable int count;
  void addref(void) const { count++; }
  void release(void) const { if (--count == 0) delete this; }
  int use_count(void) const { return count; }
  inline friend void intrusive_ptr_add_ref(const Base<T1, T2>* r) { r->addref(); }
  inline friend void intrusive_ptr_release(const Base<T1, T2>* r) { r->release(); }
};
%}

%template(BaseIntDouble) Base<int, double>;

%inline %{
template <class T1, class T2> struct Pair : Base<T1, T2> {
  Space::Klass klassPair;
  T1 val1;
  T2 val2;
  Pair(T1 t1, T2 t2) : Base<T1, T2>(t1, t2), val1(t1), val2(t2) {}
  virtual std::string getValue() const { return "Pair<>"; };
};

Pair<int, double> pair_id2(Pair<int, double> p) { return p; }
SwigBoost::intrusive_ptr< Pair<int, double> > pair_id1(SwigBoost::intrusive_ptr< Pair<int, double> > p) { return p; }

long use_count(const SwigBoost::shared_ptr<Space::Klass>& sptr) {
  return sptr.use_count();
}
long use_count(const SwigBoost::shared_ptr<Space::KlassDerived>& sptr) {
  return sptr.use_count();
}
long use_count(const SwigBoost::shared_ptr<Space::KlassDerivedDerived>& sptr) {
  return sptr.use_count();
}
%}

%template(PairIntDouble) Pair<int, double>;

// For counting the instances of intrusive_ptr (all of which are created on the heap)
// intrusive_ptr_wrapper_count() gives overall count
%inline %{
namespace SwigBoost {
  const int NOT_COUNTING = -123456;
  int intrusive_ptr_wrapper_count() { 
  #ifdef INTRUSIVE_PTR_WRAPPER
    return SwigBoost::IntrusivePtrWrapper::getTotalCount(); 
  #else
    return NOT_COUNTING;
  #endif
  }
  #ifdef INTRUSIVE_PTR_WRAPPER
  template<> std::string show_message(boost::intrusive_ptr<Space::Klass >*t) {
    if (!t)
      return "null intrusive_ptr!!!";
    if (*t)
      return "Klass: " + (*t)->getValue();
    else
      return "Klass: NULL";
  }
  template<> std::string show_message(boost::intrusive_ptr<const Space::Klass >*t) {
    if (!t)
      return "null intrusive_ptr!!!";
    if (*t)
      return "Klass: " + (*t)->getValue();
    else
      return "Klass: NULL";
  }
  template<> std::string show_message(boost::intrusive_ptr<Space::KlassDerived >*t) {
    if (!t)
      return "null intrusive_ptr!!!";
    if (*t)
      return "KlassDerived: " + (*t)->getValue();
    else
      return "KlassDerived: NULL";
  }
  template<> std::string show_message(boost::intrusive_ptr<const Space::KlassDerived >*t) {
    if (!t)
      return "null intrusive_ptr!!!";
    if (*t)
      return "KlassDerived: " + (*t)->getValue();
    else
      return "KlassDerived: NULL";
  }
  #endif
}
%}

