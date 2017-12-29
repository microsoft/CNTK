%module(directors="1") director_wstring;
%include <stl.i>

#ifndef SWIG_STL_UNIMPL

%include std_vector.i
%include std_wstring.i

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
  A(const std::wstring& first)
    : m_strings(1, first)
  {}
  
  virtual ~A() {}
  
  virtual const std::wstring& get_first() const
  { return get(0); }
  
  virtual const std::wstring& get(int n) const
  { return m_strings[n]; }

  virtual const std::wstring& call_get_first() const
  { return get_first(); }

  virtual const std::wstring& call_get(int n) const
  { return get(n); }

  std::vector<std::wstring> m_strings;


  virtual void process_text(const wchar_t *text) 
  {
  }

  virtual std::wstring multiple_params_val(const std::wstring& p1, const std::wstring& p2, std::wstring p3, std::wstring p4) const
  { return get_first(); }
  
  virtual const std::wstring& multiple_params_ref(const std::wstring& p1, const std::wstring& p2, std::wstring p3, std::wstring p4) const
  { return get_first(); }
  
  void call_process_func() { process_text(L"hello"); }
 };
 
 %}

%template(StringVector) std::vector<std::wstring>;

#endif
