from overload_template import *
f = foo()

a = maximum(3, 4)
b = maximum(3.4, 5.2)

# mix 1
if (mix1("hi") != 101):
    raise RuntimeError, ("mix1(const char*)")

if (mix1(1.0, 1.0) != 102):
    raise RuntimeError, ("mix1(double, const double &)")

if (mix1(1.0) != 103):
    raise RuntimeError, ("mix1(double)")

# mix 2
if (mix2("hi") != 101):
    raise RuntimeError, ("mix2(const char*)")

if (mix2(1.0, 1.0) != 102):
    raise RuntimeError, ("mix2(double, const double &)")

if (mix2(1.0) != 103):
    raise RuntimeError, ("mix2(double)")

# mix 3
if (mix3("hi") != 101):
    raise RuntimeError, ("mix3(const char*)")

if (mix3(1.0, 1.0) != 102):
    raise RuntimeError, ("mix3(double, const double &)")

if (mix3(1.0) != 103):
    raise RuntimeError, ("mix3(double)")

# Combination 1
if (overtparams1(100) != 10):
    raise RuntimeError, ("overtparams1(int)")

if (overtparams1(100.0, 100) != 20):
    raise RuntimeError, ("overtparams1(double, int)")

# Combination 2
if (overtparams2(100.0, 100) != 40):
    raise RuntimeError, ("overtparams2(double, int)")

# Combination 3
if (overloaded() != 60):
    raise RuntimeError, ("overloaded()")

if (overloaded(100.0, 100) != 70):
    raise RuntimeError, ("overloaded(double, int)")

# Combination 4
if (overloadedagain("hello") != 80):
    raise RuntimeError, ("overloadedagain(const char *)")

if (overloadedagain() != 90):
    raise RuntimeError, ("overloadedagain(double)")

# specializations
if (specialization(10) != 202):
    raise RuntimeError, ("specialization(int)")

if (specialization(10.0) != 203):
    raise RuntimeError, ("specialization(double)")

if (specialization(10, 10) != 204):
    raise RuntimeError, ("specialization(int, int)")

if (specialization(10.0, 10.0) != 205):
    raise RuntimeError, ("specialization(double, double)")

if (specialization("hi", "hi") != 201):
    raise RuntimeError, ("specialization(const char *, const char *)")


# simple specialization
xyz()
xyz_int()
xyz_double()

# a bit of everything
if (overload("hi") != 0):
    raise RuntimeError, ("overload()")

if (overload(1) != 10):
    raise RuntimeError, ("overload(int t)")

if (overload(1, 1) != 20):
    raise RuntimeError, ("overload(int t, const int &)")

if (overload(1, "hello") != 30):
    raise RuntimeError, ("overload(int t, const char *)")

k = Klass()
if (overload(k) != 10):
    raise RuntimeError, ("overload(Klass t)")

if (overload(k, k) != 20):
    raise RuntimeError, ("overload(Klass t, const Klass &)")

if (overload(k, "hello") != 30):
    raise RuntimeError, ("overload(Klass t, const char *)")

if (overload(10.0, "hi") != 40):
    raise RuntimeError, ("overload(double t, const char *)")

if (overload() != 50):
    raise RuntimeError, ("overload(const char *)")


# everything put in a namespace
if (nsoverload("hi") != 1000):
    raise RuntimeError, ("nsoverload()")

if (nsoverload(1) != 1010):
    raise RuntimeError, ("nsoverload(int t)")

if (nsoverload(1, 1) != 1020):
    raise RuntimeError, ("nsoverload(int t, const int &)")

if (nsoverload(1, "hello") != 1030):
    raise RuntimeError, ("nsoverload(int t, const char *)")

if (nsoverload(k) != 1010):
    raise RuntimeError, ("nsoverload(Klass t)")

if (nsoverload(k, k) != 1020):
    raise RuntimeError, ("nsoverload(Klass t, const Klass &)")

if (nsoverload(k, "hello") != 1030):
    raise RuntimeError, ("nsoverload(Klass t, const char *)")

if (nsoverload(10.0, "hi") != 1040):
    raise RuntimeError, ("nsoverload(double t, const char *)")

if (nsoverload() != 1050):
    raise RuntimeError, ("nsoverload(const char *)")


A_foo(1)
b = B()
b.foo(1)
