overload_template
f = foo();

a = maximum(3,4);
b = maximum(3.4,5.2);

# mix 1
if (mix1("hi") != 101)
  error("mix1(const char*)")
endif

if (mix1(1.0, 1.0) != 102)
  error("mix1(double, const double &)")
endif

if (mix1(1.0) != 103)
  error("mix1(double)")
endif

# mix 2
if (mix2("hi") != 101)
  error("mix2(const char*)")
endif

if (mix2(1.0, 1.0) != 102)
  error("mix2(double, const double &)")
endif

if (mix2(1.0) != 103)
  error("mix2(double)")
endif

# mix 3
if (mix3("hi") != 101)
  error("mix3(const char*)")
endif

if (mix3(1.0, 1.0) != 102)
  error("mix3(double, const double &)")
endif

if (mix3(1.0) != 103)
  error("mix3(double)")
endif

# Combination 1
if (overtparams1(100) != 10)
  error("overtparams1(int)")
endif

if (overtparams1(100.0, 100) != 20)
  error("overtparams1(double, int)")
endif

# Combination 2
if (overtparams2(100.0, 100) != 40)
  error("overtparams2(double, int)")
endif

# Combination 3
if (overloaded() != 60)
  error("overloaded()")
endif

if (overloaded(100.0, 100) != 70)
  error("overloaded(double, int)")
endif

# Combination 4
if (overloadedagain("hello") != 80)
  error("overloadedagain(const char *)")
endif

if (overloadedagain() != 90)
  error("overloadedagain(double)")
endif

# specializations
if (specialization(10) != 202)
  error("specialization(int)")
endif

if (specialization(10.1) != 203)
  error("specialization(double)")
endif

if (specialization(10, 10) != 204)
  error("specialization(int, int)")
endif

if (specialization(10.0, 10.1) != 205)
  error("specialization(double, double)")
endif

if (specialization("hi", "hi") != 201)
  error("specialization(const char *, const char *)")
endif


# simple specialization
xyz();
xyz_int();
xyz_double();

# a bit of everything
if (overload("hi") != 0)
  error("overload()")
endif

if (overload(1) != 10)
  error("overload(int t)")
endif

if (overload(1, 1) != 20)
  error("overload(int t, const int &)")
endif

if (overload(1, "hello") != 30)
  error("overload(int t, const char *)")
endif

k = Klass();
if (overload(k) != 10)
  error("overload(Klass t)")
endif

if (overload(k, k) != 20)
  error("overload(Klass t, const Klass &)")
endif

if (overload(k, "hello") != 30)
  error("overload(Klass t, const char *)")
endif

if (overload(10.1, "hi") != 40)
  error("overload(double t, const char *)")
endif

if (overload() != 50)
  error("overload(const char *)")
endif


# everything put in a namespace
if (nsoverload("hi") != 1000)
  error("nsoverload()")
endif

if (nsoverload(1) != 1010)
  error("nsoverload(int t)")
endif

if (nsoverload(1, 1) != 1020)
  error("nsoverload(int t, const int &)")
endif

if (nsoverload(1, "hello") != 1030)
  error("nsoverload(int t, const char *)")
endif

if (nsoverload(k) != 1010)
  error("nsoverload(Klass t)")
endif

if (nsoverload(k, k) != 1020)
  error("nsoverload(Klass t, const Klass &)")
endif

if (nsoverload(k, "hello") != 1030)
  error("nsoverload(Klass t, const char *)")
endif

if (nsoverload(10.1, "hi") != 1040)
  error("nsoverload(double t, const char *)")
endif

if (nsoverload() != 1050)
  error("nsoverload(const char *)")
endif


A_foo(1);
b = B();
b.foo(1);
