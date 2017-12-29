/* Test %apply for char */

%module(directors="1") apply_signed_char

%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) DirectorTest;

#if defined(SWIGSCILAB)
%rename(DirTest) DirectorTest;
#endif

%apply signed char {char, const char};
%apply const signed char & {const char &};

%inline %{
  char CharValFunction(char number) { return number; }
  const char CCharValFunction(const char number) { return number; }
  const char & CCharRefFunction(const char & number) { return number; }
  char globalchar = -109;
  const char globalconstchar = -110;
%}

// Director test
%feature("director");

%inline %{
  struct DirectorTest {

    DirectorTest() : memberchar(-111), memberconstchar(-112) {}

    virtual char CharValFunction(char number) { return number; }
    virtual const char CCharValFunction(const char number) { return number; }
    virtual const char & CCharRefFunction(const char & number) { return number; }

    char memberchar;
    const char memberconstchar;

    virtual ~DirectorTest() {}
  private:
    DirectorTest& operator=(const DirectorTest &);
  };
%}
