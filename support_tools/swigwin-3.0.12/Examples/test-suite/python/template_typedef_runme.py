from template_typedef import *

d = make_Identity_float()
c = make_Identity_reald()


try:
    a = d.this
    a = c.this
except:
    raise RuntimeError

try:
    e = make_Multiplies_float_float_float_float(d, d)
    a = e.this
except:
    print e, "is not an instance"
    raise RuntimeError

try:
    f = make_Multiplies_reald_reald_reald_reald(c, c)
    a = f.this
except:
    print f, "is not an instance"
    raise RuntimeError

try:
    g = make_Multiplies_float_float_reald_reald(d, c)
    a = g.this
except:
    print g, "is not an instance"
    raise RuntimeError


# the old large format
if not SWIG_TypeQuery("vfncs::ArithUnaryFunction<vfncs::arith_traits<float,double>::argument_type,vfncs::arith_traits<float,double >::result_type > *"):
    raise RuntimeError


# the reduced format
if not SWIG_TypeQuery("vfncs::ArithUnaryFunction<double,double> *"):
    raise RuntimeError

# this is a bad name
if SWIG_TypeQuery("vfncs::ArithUnaryFunction<double,doublex> *"):
    raise RuntimeError
