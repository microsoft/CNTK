/* This interface file tests for type-related typemap substitutions.
 */

%module typemap_subst

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) xyzzy; /* Ruby, wrong class name */

%inline %{
  struct xyzzy {
    int member;
  };
%}

%typemap(in) const struct xyzzy **TEST
  ($type temp, $*type startemp, $&type amptemp, $basetype basetemp)
{
  { /* Test C type name substitutions */
    $ltype a = (struct xyzzy **) NULL;
    const struct xyzzy **b = ($type) NULL;
    $&ltype c = (struct xyzzy ***) NULL;
    const struct xyzzy ***d = ($&type) NULL;
    $*ltype e = *a;
    $basetype f;
    f.member = 42;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)f;
  }
  { /* Test locals */
    basetemp.member = 0;
    startemp = &basetemp;
    temp = &startemp;
    amptemp = &temp;
    (void)amptemp;
  }
  { /* Test descriptors */
    void *desc = $descriptor;
    void *stardesc = $*descriptor;
    void *ampdesc = $&descriptor;
    (void)desc;
    (void)stardesc;
    (void)ampdesc;
  }
  { /* Test mangled names */
    void *desc = SWIGTYPE$mangle;
    void *stardesc = SWIGTYPE$*mangle;
    void *ampdesc = SWIGTYPE$&mangle;
    (void)desc;
    (void)stardesc;
    (void)ampdesc;
  }
  { /* Test descriptor macro */
    void *desc = $descriptor(const struct xyzzy **);
    void *stardesc = $descriptor(const struct xyzzy *);
    void *ampdesc = $descriptor(const struct xyzzy ***);
    (void)desc;
    (void)stardesc;
    (void)ampdesc;
  }
  $1 = ($ltype) temp;  
}

/* Java, C#, Go and D modules don't use SWIG's runtime type system */
#if !defined(SWIGJAVA) && !defined(SWIGCSHARP) && !defined(SWIGGO) && !defined(SWIGD)
%inline %{
  void foo(const struct xyzzy **TEST) {}
%}
#endif




