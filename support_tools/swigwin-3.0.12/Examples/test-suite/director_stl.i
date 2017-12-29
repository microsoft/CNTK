%module(directors="1") director_stl
#pragma SWIG nowarn=SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR

%include "std_string.i"
%include "std_pair.i"
%include "std_vector.i"

#ifndef SWIG_STD_DEFAULT_INSTANTIATION
%template() std::vector<double>;
%template() std::vector<int>;
%template() std::vector<std::string>;
%template() std::pair<std::string, int>;
%template() std::pair<int,double>;
%template() std::pair<double,int>;
#endif

%feature("director") Foo;

%feature("director:except") {
#ifndef SWIGPHP
  if ($error != NULL) {
#else
  if ($error == FAILURE) {
#endif
    throw Swig::DirectorMethodException();
  }
}

%exception {
  try { $action }
  catch (...) { SWIG_fail; }
}

%inline 
{
class Foo {
public:
  virtual ~Foo() {}

  virtual std::string& bar(std::string& s) 
  {
    return s;
  }
  

  virtual std::string ping(std::string s) = 0;
  virtual std::string pong(const std::string& s) 
  { return std::string("Foo::pong:") + s + ":" + ping(s); }

  std::string tping(std::string s) { return ping(s); }
  std::string tpong(const std::string& s) { return pong(s); }
  
  virtual std::pair<double, int>
  pident(const std::pair<double, int>& p) { return p; }

  virtual std::vector<int>
  vident(const std::vector<int>& p) { return p; }

  virtual std::vector<int>
  vsecond(const std::vector<int>& p, const std::vector<int>& s) { return s; }    

  std::pair<double, int>
  tpident(const std::pair<double, int>& p) { return pident(p); }

  std::vector<int>
  tvident(const std::vector<int>& p) { return vident(p); }

  virtual std::vector<int>
  tvsecond(const std::vector<int>& p, const std::vector<int>& s) { return vsecond(p,s); }


  virtual std::vector<std::string>
  vidents(const std::vector<std::string>& p) { return p; }

  std::vector<std::string>
  tvidents(const std::vector<std::string>& p) { return vidents(p); }
  
};

}
