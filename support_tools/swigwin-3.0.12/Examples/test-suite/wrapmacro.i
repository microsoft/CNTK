%module wrapmacro

#ifdef SWIGLUA	// lua only has one numeric type, so some overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) SWIGMACRO_maximum;
#endif

/* Testing technique for wrapping macros */

%{
#ifdef max
#undef max
#endif
%}

/* Here, some macros to wrap */
%inline %{

typedef unsigned short guint16;

#define GUINT16_SWAP_LE_BE_CONSTANT(val) ((guint16) ( \
    (guint16) ((guint16) (val) >> 8) |  \
    (guint16) ((guint16) (val) << 8)))

/* Don't use max(), it's a builtin function for PHP. */
#define maximum(a,b) ((a) > (b) ? (a) : (b))
  
%}


/* Here, the auxiliary macro to wrap a macro */
%define %wrapmacro(type, name, lparams, lnames)
%rename(name) SWIGMACRO_##name;
%inline %{
type SWIGMACRO_##name(lparams) {
  return name(lnames);
}
%}
%enddef
#define PLIST(...) __VA_ARGS__



/* Here, wrapping the macros */
%wrapmacro(guint16, GUINT16_SWAP_LE_BE_CONSTANT, guint16 val, val);
%wrapmacro(size_t, maximum, PLIST(size_t a, const size_t& b), PLIST(a, b));
%wrapmacro(double, maximum, PLIST(double a, double b), PLIST(a, b));


/* Maybe in the future, a swig directive will make this easier:

#define max(a,b) ((a) > (b) ? (a) : (b))

%wrapmacro double max(long a, double b); // target name is 'max'
%wrapmacro(max_i) int max(int a, int b); // changes target name to 'max_i'.

*/

%{
#ifdef max
#undef max
#endif
%}
