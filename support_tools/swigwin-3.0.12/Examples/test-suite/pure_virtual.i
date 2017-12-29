/* SF Bug #445219, submitted by Krzysztof Kozminski
   <kozminski@users.sf.net>. 

   Swig 1.3.6 gets confused by pure virtual destructors,
   as in this file:
*/

%module(ruby_minherit="1") pure_virtual

%warnfilter(SWIGWARN_JAVA_MULTIPLE_INHERITANCE,
	    SWIGWARN_CSHARP_MULTIPLE_INHERITANCE,
	    SWIGWARN_D_MULTIPLE_INHERITANCE,
	    SWIGWARN_PHP_MULTIPLE_INHERITANCE) E; /* C#, D, Java, PHP multiple inheritance */

%nodefaultctor C;
%nodefaultdtor C;
%nodefaultctor E;
%nodefaultdtor E;

%inline %{

class A {
 public:
  A() { };
  virtual ~A() = 0;
  virtual void something() = 0;
  virtual void method() = 0;
};

class B : public A {
public:
  B() {};
  virtual ~B() { };
  virtual void something() { };
  virtual void method() { };
};

/* class C is abstract because it doesn't define all methods in A */
class C : public A {
 public:
  virtual ~C() { };
  virtual void method() { };
}
;

/* class D is not abstract, it defines everything */
class D : public C {
 public:
  virtual ~D() { };
  virtual void something() { };
}
;

/* Another abstract class */
class AA {
  public:
     virtual ~AA() { }
     virtual void method2() = 0;
};

/* Multiple inheritance between two abstract classes */
class E : public C, public AA {
public:
   virtual void something() { };
};
%}

/* Fill in method from AA.  This class should be constructable */
#if defined(SWIGCSHARP) || defined(SWIGD)
%ignore F::method2(); // Work around for lack of multiple inheritance support - base AA is ignored.
#endif

%inline %{
class F : public E {
   public:
     virtual void method2() { }
};
%}

%{
A::~A() {}
%}

