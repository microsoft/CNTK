%module default_arg_values

%{
struct Display {
  // Some compilers warn about 'float v = NULL', so only SWIG sees this peculiarity
  // Bad Python wrappers were being generated when NULL used for primitive type
  float draw1(float v = 0) { return v; }
  float draw2(float *v = 0) { return v ? *v : 0; }
  bool bool0(bool x = 0) { return x; }
  bool bool1(bool x = 1) { return x; }

  typedef bool mybool;
  bool mybool0(mybool x = 0) { return x; }
  bool mybool1(mybool x = 1) { return x; }
};
float* createPtr(float v) { static float val; val = v; return &val; }
%}

struct Display {
  // Bad Python wrappers were being generated when NULL used for primitive type
  float draw1(float v = NULL) { return v; }
  float draw2(float *v = NULL) { return v ? *v : 0; }
  bool bool0(bool x = 0) { return x; }
  bool bool1(bool x = 1) { return x; }

  typedef bool mybool;
  bool mybool0(mybool x = 0) { return x; }
  bool mybool1(mybool x = 1) { return x; }
};
float* createPtr(float v) { static float val; val = v; return &val; }
