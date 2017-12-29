%module(directors="1") director_ignore

%warnfilter(SWIGWARN_LANG_DIRECTOR_ABSTRACT) DIgnoreOnlyConstructor;

%include "std_string.i"

%feature("director");

%ignore OverloadedMethod(int n, int xoffset = 0, int yoffset = 0);
%ignore OverloadedProtectedMethod(int n, int xoffset = 0, int yoffset = 0);
%ignore DIgnoreConstructor(bool b);
%ignore DIgnoreOnlyConstructor(bool b);
%ignore DIgnoreDestructor::~DIgnoreDestructor;
%ignore Pointers;
%ignore References;
%ignore PublicMethod1;
%ignore PublicMethod2;
%ignore PublicPureVirtualMethod1;
%ignore PublicPureVirtualMethod2;
%ignore ProtectedMethod1;
%ignore ProtectedMethod2;
%ignore ProtectedPureVirtualMethod1;
%ignore ProtectedPureVirtualMethod2;

%typemap(imtype,
  inattributes="[inattributes should not be used]",
  outattributes="[outattributes should not be used]",
  directorinattributes="[directorinattributes should not be used]",
  directoroutattributes="[directoroutattributes should not be used]"
 ) int& "imtype should not be used"

%inline %{

#include <string>
class DIgnores
{
  public:
    virtual ~DIgnores() {}
    virtual void OverloadedMethod(int n, int xoffset = 0, int yoffset = 0) {}
    virtual void OverloadedMethod(bool b) {}
    virtual int Triple(int n) { return n*3; }
    virtual int& References(int& n) { static int nn; nn=n; return nn; }
    virtual int* Pointers(int* n) { static int nn; nn=*n; return &nn; }
    virtual double PublicMethod1() { return 0.0; }
    virtual double PublicPureVirtualMethod1() = 0;
    virtual void PublicMethod2() {}
    virtual void PublicPureVirtualMethod2() = 0;
    virtual void TempMethod() = 0;
  protected:
    virtual void OverloadedProtectedMethod(int n, int xoffset = 0, int yoffset = 0) {}
    virtual void OverloadedProtectedMethod() {}
    virtual double ProtectedMethod1() { return 0.0; }
    virtual double ProtectedPureVirtualMethod1() = 0;
    virtual void ProtectedMethod2() {}
    virtual void ProtectedPureVirtualMethod2() = 0;
};

class DAbstractIgnores
{
  public:
    virtual ~DAbstractIgnores() {}
    virtual double OverloadedMethod(int n, int xoffset = 0, int yoffset = 0) = 0;
    virtual double OverloadedMethod(bool b) = 0;
    virtual int Quadruple(int n) { return n*4; }
    virtual int& References(int& n) { static int nn; nn=n; return nn; }
    virtual int* Pointers(int* n) { static int nn; nn=*n; return &nn; }
  protected:
    virtual double OverloadedProtectedMethod(int n, int xoffset = 0, int yoffset = 0) = 0;
    virtual double OverloadedProtectedMethod() = 0;
};

template <typename T> class DTemplateAbstractIgnores
{
  T t;
  public:
    virtual ~DTemplateAbstractIgnores() {}
    virtual double OverloadedMethod(int n, int xoffset = 0, int yoffset = 0) = 0;
    virtual double OverloadedMethod(bool b) = 0;
    virtual int Quadruple(int n) { return n*4; }
    virtual int& References(int& n) { static int nn; nn=n; return nn; }
    virtual int* Pointers(int* n) { static int nn; nn=*n; return &nn; }
  protected:
    virtual double OverloadedProtectedMethod(int n, int xoffset = 0, int yoffset = 0) = 0;
    virtual double OverloadedProtectedMethod() = 0;
};
%}

%template(DTemplateAbstractIgnoresInt) DTemplateAbstractIgnores<int>;

class DIgnoreConstructor
{
  public:
    virtual ~DIgnoreConstructor() {}
    DIgnoreConstructor(std::string s, int i) {}
    DIgnoreConstructor(bool b) {}
};

class DIgnoreOnlyConstructor
{
  public:
    virtual ~DIgnoreOnlyConstructor() {}
    DIgnoreOnlyConstructor(bool b) {}
};

class DIgnoreDestructor
{
 public:
  DIgnoreDestructor() {}
  virtual ~DIgnoreDestructor() {}
};

%{
class DIgnoreConstructor
{
  public:
    virtual ~DIgnoreConstructor() {}
    DIgnoreConstructor(std::string s, int i) {}
  private: // Hide constructor
    DIgnoreConstructor(bool b) {}
};

class DIgnoreOnlyConstructor
{
  public:
    virtual ~DIgnoreOnlyConstructor() {}
  private: // Hide constructor
    DIgnoreOnlyConstructor(bool b) {}
};

class DIgnoreDestructor
{
 public:
  DIgnoreDestructor() {}
  virtual ~DIgnoreDestructor() {}
};

%}
