%module(directors="1") java_director_assumeoverride

%{
class OverrideMe {
public:
  virtual ~OverrideMe() {}
  virtual void func() {};
};

#include "java_director_assumeoverride_wrap.h"
bool isFuncOverridden(OverrideMe* f) {
  SwigDirector_OverrideMe* director = dynamic_cast<SwigDirector_OverrideMe*>(f);
  if (!director) {
    return false;
  }
  return director->swig_overrides(0);
}

%}

%feature("director", assumeoverride=1) OverrideMe;

class OverrideMe {
public:
  virtual ~OverrideMe();
  virtual void func();
};

bool isFuncOverridden(OverrideMe* f);
