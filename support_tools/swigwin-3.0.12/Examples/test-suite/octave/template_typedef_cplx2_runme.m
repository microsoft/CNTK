template_typedef_cplx2

#
# double case
#

try
  d = make_Identity_double();
  a = swig_this(d);
catch
  d
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(d)) != 1)
  d
  error("is not an ArithUnaryFunction")
  error
endif

try
  e = make_Multiplies_double_double_double_double(d, d);
  a = swig_this(e);
catch
  e
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(e)) != 1)
  e
  error("is not an ArithUnaryFunction")
endif


#
# complex case
#

try
  c = make_Identity_complex();
  a = swig_this(c);
catch
  c
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(c)) != 1)
  c
  error("is not an ArithUnaryFunction")
endif

try
  f = make_Multiplies_complex_complex_complex_complex(c, c);
  a = swig_this(f);
catch
  f
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(f)) != 1)
  f
  error("is not an ArithUnaryFunction")
endif

#
# Mix case
#

try
  g = make_Multiplies_double_double_complex_complex(d, c);
  a = swig_this(g);
catch
  g
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(g)) != 1)
  g
  error("is not an ArithUnaryFunction")
  error
endif


try
  h = make_Multiplies_complex_complex_double_double(c, d);
  a = swig_this(h);
catch
  h
  error("is not an instance")
end_try_catch

if (findstr('ArithUnaryFunction',swig_type(h)) == -1)
  h
  error("is not an ArithUnaryFunction")
endif

try
  a = g.get_value();
catch
  error(g, "has not get_value() method")
  error
end_try_catch

