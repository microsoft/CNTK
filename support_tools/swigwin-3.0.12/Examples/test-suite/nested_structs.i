%module nested_structs

#if defined(SWIG_JAVASCRIPT_V8)

%inline %{
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* for nested C class wrappers compiled as C++ code */
/* dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing] */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
%}

#endif

// bug #491476
%inline %{
struct Outer {
  struct {
    int val;
  } inner1, inner2, *inner3, inner4[1], **inner5;
  struct Named {
    int val;
  } inside1, inside2, *inside3, inside4[1], **inside5;
} outer;

void setValues(struct Outer *outer, int val) {
  outer->inner1.val = val;
  outer->inner2.val = val * 2;
  outer->inner3 = &outer->inner2;
  outer->inner4[0].val = val * 4;
  outer->inner5 = &outer->inner3;

  val = val * 10;
  outer->inside1.val = val;
  outer->inside2.val = val * 2;
  outer->inside3 = &outer->inside2;
  outer->inside4[0].val = val * 4;
  outer->inside5 = &outer->inside3;
}

int getInside1Val(struct Outer *n) { return n->inside1.val; }
%}

/* 
Below was causing problems in Octave as wrappers were compiled as C++.
Solution requires regenerating the inner struct into
the global C++ namespace (which is where it is intended to be in C).
See cparse_cplusplusout / Swig_cparse_cplusplusout in the Source.
*/
%inline %{
int nestedByVal(struct Named s);
int nestedByPtr(struct Named *s);
%}
%{
int nestedByVal(struct Named s) { return s.val; }
int nestedByPtr(struct Named *s) { return s->val; }
%}

