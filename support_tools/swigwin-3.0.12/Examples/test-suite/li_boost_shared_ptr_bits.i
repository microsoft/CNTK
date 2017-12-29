%module li_boost_shared_ptr_bits

#if defined(SWIGJAVA) || defined(SWIGCSHARP) || defined(SWIGPYTHON) || defined(SWIGD) || defined(SWIGOCTAVE) || defined(SWIGRUBY)
#define SHARED_PTR_WRAPPERS_IMPLEMENTED
#endif

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%include <boost_shared_ptr.i>
%shared_ptr(NonDynamic)

#endif

#if defined(SWIGPYTHON)
%pythonnondynamic NonDynamic;
#endif

%inline %{
#include <boost/shared_ptr.hpp>
struct NonDynamic {
  int i;
};
boost::shared_ptr<NonDynamic> boing(boost::shared_ptr<NonDynamic> b) { return b; }
%}

// vector of shared_ptr
%include "std_vector.i"

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%shared_ptr(IntHolder);

#endif

%inline %{
#include "boost/shared_ptr.hpp"
struct IntHolder {
  int val;
  IntHolder(int a) : val(a) {}
};
int sum(std::vector< boost::shared_ptr<IntHolder> > v) {
  int sum = 0;
  for (size_t i=0; i<v.size(); ++i)
    sum += v[i]->val;
  return sum;
}
%}

%template(VectorIntHolder) std::vector< boost::shared_ptr<IntHolder> >;


/////////////////////////////////////////////////
// Test non public destructor - was leading to memory leaks as the destructor was not wrapped
// Bug 3024875
/////////////////////////////////////////////////

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%shared_ptr(HiddenDestructor)

#endif

%inline %{
class HiddenDestructor;
typedef boost::shared_ptr< HiddenDestructor > FooPtr;

class HiddenDestructor {
public:
   static FooPtr create();
   virtual void doit();

protected:
   HiddenDestructor();
   static void Foo_body( FooPtr self );
   virtual ~HiddenDestructor();
private:
   HiddenDestructor( const HiddenDestructor& );
   class Impl;
   Impl* impl_;

   class FooDeleter {
   public:
     void operator()(HiddenDestructor* hidden) {
       delete hidden;
     }
   };
};
%}

%{
#include <iostream>
using namespace std;

/* Impl would generally hold a weak_ptr to HiddenDestructor a.s.o, but this stripped down example should suffice */
class HiddenDestructor::Impl {
public:
    int mymember;
};

FooPtr HiddenDestructor::create()
{
    FooPtr hidden( new HiddenDestructor(), HiddenDestructor::FooDeleter() );
    Foo_body( hidden );
    return hidden;
}

void HiddenDestructor::doit()
{
    // whatever
}

HiddenDestructor::HiddenDestructor()
{
//  cout << "HiddenDestructor::HiddenDestructor()" << endl;
    // always empty
}

void HiddenDestructor::Foo_body( FooPtr self )
{
    // init self as you would do in ctor
    self->impl_ = new Impl();
}

HiddenDestructor::~HiddenDestructor()
{
//  cout << "HiddenDestructor::~HiddenDestructor()" << endl;
    // destruct (e.g. delete Pimpl object)
    delete impl_;
}
%}

////////////////////////////
// As above but private instead of protected destructor
////////////////////////////

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)

%shared_ptr(HiddenPrivateDestructor)

#endif


%inline %{
class HiddenPrivateDestructor {
private:
  HiddenPrivateDestructor() {}
  virtual ~HiddenPrivateDestructor() {
    DeleteCount++;
  }

  class Deleter {
  public:
    void operator()(HiddenPrivateDestructor *hidden) {
      delete hidden;
    }
  };

public:
  static boost::shared_ptr<HiddenPrivateDestructor> create() {
    boost::shared_ptr<HiddenPrivateDestructor> hidden( new HiddenPrivateDestructor(), HiddenPrivateDestructor::Deleter() );
    return hidden;
  }
  static int DeleteCount;
};

int HiddenPrivateDestructor::DeleteCount = 0;
%}

/////////////////////////////////////////////////
// Non-public inheritance and shared_ptr
/////////////////////////////////////////////////

#if defined(SHARED_PTR_WRAPPERS_IMPLEMENTED)
%shared_ptr(Base)
// No %shared_ptr(DerivedPrivate1) to check Warning 520 does not appear
// No %shared_ptr(DerivedProtected1) to check Warning 520 does not appear
%shared_ptr(DerivedPrivate2)
%shared_ptr(DerivedProtected2)

%ignore Base2;
%shared_ptr(DerivedPublic)
#endif

%inline %{
class Base {
public:
  virtual int b() = 0;
  virtual ~Base() {}
};

class DerivedProtected1 : protected Base {
public:
  virtual int b() { return 20; }
};
class DerivedPrivate1 : private Base {
public:
  virtual int b() { return 20; }
};

class DerivedProtected2 : protected Base {
public:
  virtual int b() { return 20; }
};
class DerivedPrivate2 : private Base {
public:
  virtual int b() { return 20; }
};

class Base2 {
public:
  virtual int b2() = 0;
  virtual ~Base2() {}
};
class DerivedPublic : public Base2 {
public:
  virtual int b2() { return 20; }
};
%}
