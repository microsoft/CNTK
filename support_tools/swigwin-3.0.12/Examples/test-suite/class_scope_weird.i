%module class_scope_weird

// Use this version with extra qualifiers to test SWIG as some compilers accept this
class Foo {
public:
  Foo::Foo(void) {}
  Foo::Foo(int) {}
  int Foo::bar(int x) {
    return x;
  }
};

// Remove extra qualifiers for the compiler as some compilers won't compile the extra qaulification (eg gcc-4.1 onwards) 
%{
class Foo {
public:
  Foo(void) {}
  Foo(int) {}
  int bar(int x) {
    return x;
  }
};
%}

%inline %{
class Quat;
class matrix4;
class tacka3;
%}

// Use this version with extra qualifiers to test SWIG as some compilers accept this
class Quat {
public:
  Quat::Quat(void){}  
  Quat::Quat(float in_w, float x, float y, float z){}
  Quat::Quat(const tacka3& axis, float angle){}
  Quat::Quat(const matrix4& m){}
};

// Remove extra qualifiers for the compiler as some compilers won't compile the extra qaulification (eg gcc-4.1 onwards) 
%{
class Quat {
public:
  Quat(void){}  
  Quat(float in_w, float x, float y, float z){}
  Quat(const tacka3& axis, float angle){}
  Quat(const matrix4& m){}
};
%}

