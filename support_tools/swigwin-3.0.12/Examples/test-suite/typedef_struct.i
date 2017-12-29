%module typedef_struct

%inline %{

#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* for anonymous enums */
/* dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing] */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

  typedef struct {
    int numpoints;
  } LineObj;
  
  typedef LineObj MultipointObj;
  
  typedef struct {
#ifdef SWIG
    %immutable;
#endif
    char *filename; 
    int numfonts;
#ifdef SWIG
    %mutable;
#endif
    int* fonts;
#ifndef SWIG
    void* map;
#endif
  } FontSetObj;

#define MS_NOOVERRIDE -1111

%}


%inline %{
typedef struct {
    int a;
    int b;
} A_t, *B_t;

A_t* make_a() {
    return (A_t*)malloc(sizeof(A_t));
} 

B_t make_b() {
    return make_a();
}
%} 


%inline %{

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _Foo {
  enum { NONAME1, NONAME2 } enumvar;
  int foovar;
  void (*fptr)(int);
} Foo;

#ifdef __cplusplus
}
#endif

%}
