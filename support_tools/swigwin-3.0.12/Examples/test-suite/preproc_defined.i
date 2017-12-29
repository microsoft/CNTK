%module preproc_defined

// Check 'defined' passes through the preprocessor without being processed like '#if defined(ABC)' would be (SF bug #1940536)

%define DEFINED_MACRO
%{
  int defined(int b) { 
    return b > 10;
  }
  int vvv = -1;
  void fn(int val) {
    if (defined(val))
      vvv = 1;
    else
      vvv = 0; 
  }
%}
%enddef

DEFINED_MACRO

%{
int checking(void) {
  int okay;
  fn(11);
  okay = (vvv == 1);
  fn(9);
  okay = okay && (vvv == 0);
  return okay; /* should be 1 */
}
%}

%inline %{
int call_checking(void) {
  return checking();
}
%}

/*****************************************************************************/
/* Check #if/#elif defined() macro expansions
   Also checks #if/#elif defined() works correctly within macros... this is not
   standard C, but is now relied on in the SWIG library. */
/*****************************************************************************/

#define AAA
#define BBB
#define CCC

#if defined(AAA)\
&& defined(BBB) \
&& defined(CCC)
%{
void thing(int i) {}
void stuff(int i) {}
struct Defined {
  int defined;
};
void bumpf(int i) {}
%}
#else
#endif

%define ANOTHER_MACRO(TYPE)

#if defined(AAA) && defined(BBB) && defined(CCC)
void thing(TYPE) {}
#else
void thing_not(TYPE) {}
#endif

#if defined(AAA) &&\
 defined(BBB) \\
&& defined(CCC)
void stuff(TYPE) {}
#else
void stuff_not(TYPE);
#endif

#if defined(0)
void defined_not(TYPE);
#elif defined(AAA) && defined( BBB ) && defined(CCC)
struct Defined {
  int defined;
};
#else
void defined_not(TYPE);
#endif

#if !( defined(AAA) \
 defined(BBB) \\
&& defined(CCC) )
void bumpf_not(TYPE);
#else
void bumpf(TYPE) {}
#endif

%enddef

ANOTHER_MACRO(int)

%{
void another_macro_checking(void) {
  struct Defined d;
  d.defined = 10;
  (void)d;
  thing(10);
  stuff(10);
  bumpf(10);
}
%}

/* Check that unknown preprocessor directives are ignored inside an inactive
 * conditional (github issue #394).
 */
#ifdef APPLE_OPENGL
# import <OpenGLES/ES1/gl.h>
#endif
#ifdef AAA
# define B
#else
# wibble wibble
#endif
#if 0
# wobble wobble
#endif
