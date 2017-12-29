// Test using a target language specified base class, primarily for Java/C#/D and possibly other single inheritance languages

// Note the multiple inheritance warnings don't appear because of the two techniques used in here: typemaps and %ignore

%module inherit_target_language

#if defined(SWIGJAVA)
# define csbase javabase
#elif defined(SWIGD)
# define csbase dbase
#endif

%pragma(csharp) moduleimports=%{
using System;
using System.Runtime.InteropServices;
public class TargetLanguageBase { public virtual void targetLanguageBaseMethod() {} };
public class TargetLanguageBase2 { public virtual void targetLanguageBase2Method() {} };
%}

%pragma(java) moduleimports=%{
class TargetLanguageBase { public void targetLanguageBaseMethod() {} };
class TargetLanguageBase2 { public void targetLanguageBase2Method() {} };
%}

%pragma(d) globalproxyimports=%{
private class TargetLanguageBase { public void targetLanguageBaseMethod() {} };
private class TargetLanguageBase2 { public void targetLanguageBase2Method() {} };
%}

%typemap(csbase) SWIGTYPE "TargetLanguageBase"

// Two ways to replace a C++ base with a completely different target language base
%ignore Base1; // another way to use the target language base
%typemap(csbase, replace="1") Derived2 "TargetLanguageBase"

%inline %{
struct Base1 { virtual ~Base1() {} };
struct Base2 { virtual ~Base2() {} };
struct Derived1 : Base1 {};
struct Derived2 : Base2 {};
%}

// Multiple inheritance
%ignore MBase1a;
%ignore MBase1b;
%typemap(csbase, replace="1") MultipleDerived2 "TargetLanguageBase"

%inline %{
struct MBase1a { virtual ~MBase1a() {} virtual void a() {} };
struct MBase1b { virtual ~MBase1b() {} virtual void b() {} };
struct MBase2a { virtual ~MBase2a() {} virtual void c() {} };
struct MBase2b { virtual ~MBase2b() {} virtual void d() {} };
struct MultipleDerived1 : MBase1a, MBase1b {};
struct MultipleDerived2 : MBase1a, MBase2b {};
%}


%ignore MBase3a;
%ignore MBase4b;
%typemap(csbase) MultipleDerived3 ""
%typemap(csbase) MultipleDerived4 ""

%inline %{
struct MBase3a { virtual ~MBase3a() {} virtual void e() {} };
struct MBase3b { virtual ~MBase3b() {} virtual void f() {} };
struct MBase4a { virtual ~MBase4a() {} virtual void g() {} };
struct MBase4b { virtual ~MBase4b() {} virtual void h() {} };
struct MultipleDerived3 : MBase3a, MBase3b {};
struct MultipleDerived4 : MBase4a, MBase4b {};
%}

// Replace a C++ base, but only classes that do not have a C++ base
%typemap(csbase, notderived="1") SWIGTYPE "TargetLanguageBase2"

%inline %{
struct BaseX            { virtual ~BaseX() {}; void basex() {} };
struct DerivedX : BaseX { void derivedx() {} };
%}

