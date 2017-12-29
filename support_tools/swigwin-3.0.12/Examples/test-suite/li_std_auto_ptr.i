%module li_std_auto_ptr

%{
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // auto_ptr deprecation
#endif
%}

#if defined(SWIGCSHARP) || defined(SWIGJAVA) || defined(SWIGPYTHON)

%include "std_auto_ptr.i"

%auto_ptr(Klass)

%inline %{

#include <memory>
#include <string>
#include "swig_examples_lock.h"

class Klass {
public:
  explicit Klass(const char* label) :
    m_label(label)
  {
    SwigExamples::Lock lock(critical_section);
    total_count++;
  }

  const char* getLabel() const { return m_label.c_str(); }

  ~Klass()
  {
    SwigExamples::Lock lock(critical_section);
    total_count--;
  }

  static int getTotal_count() { return total_count; }

private:
  static SwigExamples::CriticalSection critical_section;
  static int total_count;

  std::string m_label;
};

SwigExamples::CriticalSection Klass::critical_section;
int Klass::total_count = 0;

%}

%template(KlassAutoPtr) std::auto_ptr<Klass>;

%inline %{

std::auto_ptr<Klass> makeKlassAutoPtr(const char* label) {
  return std::auto_ptr<Klass>(new Klass(label));
}

%}

#endif
