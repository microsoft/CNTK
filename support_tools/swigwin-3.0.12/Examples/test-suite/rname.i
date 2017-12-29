// This module tests various facets of the %rename directive

%module rname

/* Applied everywhere */
%rename(foo_i) foo(int);
%rename(foo_d) foo(double);

/* Applied only to global scope */

%rename(foo_s) ::foo(short);

/* Applied only to class scope */

%rename(foo_u) *::foo(unsigned);

/* Rename classes in a class hierarchy */
%rename (RenamedBase) Space::Base;
%rename (RenamedDerived) Space::Derived;

/* Rename base class method applies to derived classes too */#
%rename (newname) Space::Base::oldname(double d) const;

/* Rename derived class method only */
%rename (Xfunc) Space::Derived::fn(Base baseValue, Base* basePtr, Base& baseRef);

%inline %{
class Bar {
public:
   char *foo(int)      { return (char *) "Bar::foo-int"; }
   char *foo(double)   { return (char *) "Bar::foo-double"; }
   char *foo(short)    { return (char *) "Bar::foo-short"; }
   char *foo(unsigned) { return (char *) "Bar::foo-unsigned"; }
};

char *foo(int)      { return (char *) "foo-int"; }
char *foo(double)   { return (char *) "foo-double"; }
char *foo(short)    { return (char *) "foo-short"; }
char *foo(unsigned) { return (char *) "foo-unsigned"; }

namespace Space {
class Base {
public: 
  Base(){}; 
  virtual ~Base(){};
  void fn(Base baseValue, Base* basePtr, Base& baseRef){}
  virtual const char * oldname(double d) const { return "Base"; }
};
class Derived : public Base {
public:
  Derived(){}
  ~Derived(){}
  void fn(Base baseValue, Base* basePtr, Base& baseRef){}
  virtual const char * oldname(double d) const { return "Derived"; }
};
}

%}

