%module(directors="1") director_string;
%include <stl.i>

#ifndef SWIG_STL_UNIMPL

%include std_vector.i
%include std_string.i

// Using thread unsafe wrapping
%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) A;

%{
#include <vector>
#include <string>
%}

%feature("director") A;
%inline %{

struct A
{
  A(const std::string& first)
    : m_strings(1, first)
  {}
  
  virtual ~A() {}
  
  virtual const std::string& get_first() const
  { return get(0); }
  
  virtual const std::string& get(int n) const
  { return m_strings[n]; }

  virtual const std::string& call_get_first() const
  { return get_first(); }

  virtual const std::string& call_get(int n) const
  { return get(n); }

  virtual int string_length(const std::string & s) const
  { return (int)s.size(); }

  std::vector<std::string> m_strings;


  virtual void process_text(const char *text) 
  {
  }

  void call_process_func() { process_text("hello"); }
 };
 
 %}

%template(StringVector) std::vector<std::string>;


#endif
