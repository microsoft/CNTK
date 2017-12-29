%module typemap_qualifier_strip

%typemap(freearg) int *ptr ""
%typemap(freearg) int *const ptrConst ""
%typemap(freearg) int const* constPtr ""

%typemap(in) int *ptr {
  int temp = 1234;
  $1 = &temp;
}

%typemap(in) int *const ptrConst {
  int temp = 5678;
  $1 = &temp;
}

%typemap(in) int const* constPtr {
  int temp = 3456;
  $1 = &temp;
}

%inline %{
int *create_int(int newval) {
  static int val = 0;
  val = newval;
  return &val;
}
int testA1(int const*const ptr) {
  return *ptr;
}
int testA2(int const* ptr) {
  return *ptr;
}
int testA3(int *const ptr) {
  return *ptr;
}
int testA4(int * ptr) {
  return *ptr;
}

int testB1(int const*const p) {
  return *p;
}
int testB2(int const* p) {
  return *p;
}
int testB3(int *const p) {
  return *p;
}
int testB4(int * p) {
  return *p;
}

int testC1(int const*const ptrConst) {
  return *ptrConst;
}
int testC2(int const* ptrConst) {
  return *ptrConst;
}
int testC3(int *const ptrConst) {
  return *ptrConst;
}
int testC4(int * ptrConst) {
  return *ptrConst;
}

int testD1(int const*const constPtr) {
  return *constPtr;
}
int testD2(int const* constPtr) {
  return *constPtr;
}
int testD3(int *const constPtr) {
  return *constPtr;
}
int testD4(int * constPtr) {
  return *constPtr;
}
%}

