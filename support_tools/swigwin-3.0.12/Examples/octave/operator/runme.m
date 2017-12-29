# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

# Operator overloading example
swigexample

a = swigexample.ComplexVal(2,3);
b = swigexample.ComplexVal(-5,10);

printf("a   = %s\n",disp(a));
printf("b   = %s\n",disp(b));

c = a + b;
printf("c   = %s\n",disp(c));
printf("a*b = %s\n",disp(a*b));
printf("a-c = %s\n",disp(a-c));

e = swigexample.ComplexVal(a-c);
printf("e   = %s\n",disp(e));

# Big expression
f = ((a+b)*(c+b*e)) + (-a);
printf("f   = %s\n",disp(f));

# paren overloading
printf("a(3)= %s\n",disp(a(3)));

# friend operator
printf("2*a = %s\n",disp(2*a));

# conversions
printf("single(a) = %g\n", single(a));
printf("double(a) = %g\n", double(a));

# unary functions
if swig_octave_prereq(3,8,0)
  printf("real(a) = %g\n", real(a));
  printf("imag(a) = %g\n", imag(a));
  printf("abs(a) = %g\n", abs(a));
  printf("conj(a) = %s\n", disp(conj(a)));
  printf("exp(a) = %s\n", disp(exp(a)));
endif
