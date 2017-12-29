template_typedef

d = make_Identity_float();
c = make_Identity_reald();


try
  a = swig_this(d);
  a = swig_this(c);
catch
  error
end_try_catch

try
  e = make_Multiplies_float_float_float_float(d, d);
  a = swig_this(e);
catch
  e
  error("is not an instance")
end_try_catch

try
  f = make_Multiplies_reald_reald_reald_reald(c, c);
  a = swig_this(f);
catch
  f
  error("is not an instance");
end_try_catch

try
  g = make_Multiplies_float_float_reald_reald(d, c);
  a = swig_this(g);
catch
  error(g, "is not an instance")
end_try_catch


# the old large format
if (strcmp("<unknown>",swig_typequery("vfncs::ArithUnaryFunction<vfncs::arith_traits<float,double>::argument_type,vfncs::arith_traits<float,double >::result_type > *")))
  error
endif

# the reduced format
if (strcmp("<unknown>",swig_typequery("vfncs::ArithUnaryFunction<double,double> *")))
  error
endif

# this is a bad name
if (!strcmp("<unknown>",swig_typequery("vfncs::ArithUnaryFunction<double,doublex> *")))
  error
endif
