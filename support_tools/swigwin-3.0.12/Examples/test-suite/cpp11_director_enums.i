%module(directors="1") cpp11_director_enums

%warnfilter(SWIGWARN_TYPEMAP_THREAD_UNSAFE,SWIGWARN_TYPEMAP_DIRECTOROUT_PTR) Cpp11DirectorEnumsCallback::g;

%director Cpp11DirectorEnumsCallback;

%inline %{
enum class Color { Red, Green, Blue=10 };
struct Cpp11DirectorEnumsCallback {
  virtual Color f(Color c) = 0;
  virtual const Color & g(const Color &c) = 0;
  virtual ~Cpp11DirectorEnumsCallback() {}
};
%}
