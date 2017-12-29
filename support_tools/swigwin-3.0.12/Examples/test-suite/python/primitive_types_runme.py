import sys
from primitive_types import *

var_init()

# assigning globals calls
cvar.var_bool = sct_bool
cvar.var_schar = sct_schar
cvar.var_uchar = sct_uchar
cvar.var_int = sct_int
cvar.var_uint = sct_uint
cvar.var_short = sct_short
cvar.var_ushort = sct_ushort
cvar.var_long = sct_long
cvar.var_ulong = sct_ulong
cvar.var_llong = sct_llong
cvar.var_ullong = sct_ullong
cvar.var_char = sct_char
cvar.var_pchar = sct_pchar
cvar.var_pcharc = sct_pcharc
cvar.var_pint = sct_pint
cvar.var_sizet = sct_sizet
cvar.var_hello = sct_hello
cvar.var_myint = sct_myint
cvar.var_namet = def_namet
cvar.var_parami = sct_parami
cvar.var_paramd = sct_paramd
cvar.var_paramc = sct_paramc

v_check()


def pyerror(name, val, cte):
    print "bad val/cte", name, val, cte
    raise RuntimeError
    pass

if cvar.var_bool != cct_bool:
    pyerror("bool", cvar.var_bool, cct_bool)
if cvar.var_schar != cct_schar:
    pyerror("schar", cvar.var_schar, cct_schar)
if cvar.var_uchar != cct_uchar:
    pyerror("uchar", cvar.var_uchar, cct_uchar)
if cvar.var_int != cct_int:
    pyerror("int", cvar.var_int, cct_int)
if cvar.var_uint != cct_uint:
    pyerror("uint", cvar.var_uint, cct_uint)
if cvar.var_short != cct_short:
    pyerror("short", cvar.var_short, cct_short)
if cvar.var_ushort != cct_ushort:
    pyerror("ushort", cvar.var_ushort, cct_ushort)
if cvar.var_long != cct_long:
    pyerror("long", cvar.var_long, cct_long)
if cvar.var_ulong != cct_ulong:
    pyerror("ulong", cvar.var_ulong, cct_ulong)
if cvar.var_llong != cct_llong:
    pyerror("llong", cvar.var_llong, cct_llong)
if cvar.var_ullong != cct_ullong:
    pyerror("ullong", cvar.var_ullong, cct_ullong)
if cvar.var_char != cct_char:
    pyerror("char", cvar.var_char, cct_char)
if cvar.var_pchar != cct_pchar:
    pyerror("pchar", cvar.var_pchar, cct_pchar)
if cvar.var_pcharc != cct_pcharc:
    pyerror("pchar", cvar.var_pcharc, cct_pcharc)
if cvar.var_pint != cct_pint:
    pyerror("pint", cvar.var_pint, cct_pint)
if cvar.var_sizet != cct_sizet:
    pyerror("sizet", cvar.var_sizet, cct_sizet)
if cvar.var_hello != cct_hello:
    pyerror("hello", cvar.var_hello, cct_hello)
if cvar.var_myint != cct_myint:
    pyerror("myint", cvar.var_myint, cct_myint)
if cvar.var_namet != def_namet:
    pyerror("name", cvar.var_namet, def_namet)


class PyTest (TestDirector):

    def __init__(self):
        TestDirector.__init__(self)
        pass

    def ident(self, x):
        return x

    def vval_bool(self, x): return self.ident(x)

    def vval_schar(self, x): return self.ident(x)

    def vval_uchar(self, x): return self.ident(x)

    def vval_int(self, x): return self.ident(x)

    def vval_uint(self, x): return self.ident(x)

    def vval_short(self, x): return self.ident(x)

    def vval_ushort(self, x): return self.ident(x)

    def vval_long(self, x): return self.ident(x)

    def vval_ulong(self, x): return self.ident(x)

    def vval_llong(self, x): return self.ident(x)

    def vval_ullong(self, x): return self.ident(x)

    def vval_float(self, x): return self.ident(x)

    def vval_double(self, x): return self.ident(x)

    def vval_char(self, x): return self.ident(x)

    def vval_pchar(self, x): return self.ident(x)

    def vval_pcharc(self, x): return self.ident(x)

    def vval_pint(self, x): return self.ident(x)

    def vval_sizet(self, x): return self.ident(x)

    def vval_hello(self, x): return self.ident(x)

    def vval_myint(self, x): return self.ident(x)

    def vref_bool(self, x): return self.ident(x)

    def vref_schar(self, x): return self.ident(x)

    def vref_uchar(self, x): return self.ident(x)

    def vref_int(self, x): return self.ident(x)

    def vref_uint(self, x): return self.ident(x)

    def vref_short(self, x): return self.ident(x)

    def vref_ushort(self, x): return self.ident(x)

    def vref_long(self, x): return self.ident(x)

    def vref_ulong(self, x): return self.ident(x)

    def vref_llong(self, x): return self.ident(x)

    def vref_ullong(self, x): return self.ident(x)

    def vref_float(self, x): return self.ident(x)

    def vref_double(self, x): return self.ident(x)

    def vref_char(self, x): return self.ident(x)

    def vref_pchar(self, x): return self.ident(x)

    def vref_pcharc(self, x): return self.ident(x)

    def vref_pint(self, x): return self.ident(x)

    def vref_sizet(self, x): return self.ident(x)

    def vref_hello(self, x): return self.ident(x)

    def vref_myint(self, x): return self.ident(x)

    pass


t = Test()
p = PyTest()


# internal call check
if t.c_check() != p.c_check():
    raise RuntimeError, "bad director"

p.var_bool = p.stc_bool
p.var_schar = p.stc_schar
p.var_uchar = p.stc_uchar
p.var_int = p.stc_int
p.var_uint = p.stc_uint
p.var_short = p.stc_short
p.var_ushort = p.stc_ushort
p.var_long = p.stc_long
p.var_ulong = p.stc_ulong
p.var_llong = p.stc_llong
p.var_ullong = p.stc_ullong
p.var_char = p.stc_char
p.var_pchar = sct_pchar
p.var_pcharc = sct_pcharc
p.var_pint = sct_pint
p.var_sizet = sct_sizet
p.var_hello = sct_hello
p.var_myint = sct_myint
p.var_namet = def_namet
p.var_parami = sct_parami
p.var_paramd = sct_paramd
p.var_paramc = sct_paramc

p.v_check()


t.var_bool = t.stc_bool
t.var_schar = t.stc_schar
t.var_uchar = t.stc_uchar
t.var_int = t.stc_int
t.var_uint = t.stc_uint
t.var_short = t.stc_short
t.var_ushort = t.stc_ushort
t.var_long = t.stc_long
t.var_ulong = t.stc_ulong
t.var_llong = t.stc_llong
t.var_ullong = t.stc_ullong
t.var_char = t.stc_char
t.var_pchar = sct_pchar
t.var_pcharc = sct_pcharc
t.var_pint = sct_pint
t.var_sizet = sct_sizet
t.var_hello = sct_hello
t.var_myint = sct_myint
t.var_namet = def_namet
t.var_parami = sct_parami
t.var_paramd = sct_paramd
t.var_paramc = sct_paramc

t.v_check()

# this value contains a '0' char!
if def_namet != 'hola':
    print "bad namet", def_namet
    raise RuntimeError

t.var_namet = def_namet
if t.var_namet != def_namet:
    print "bad namet", t.var_namet, def_namet
    raise RuntimeError

t.var_namet = 'hola'

if t.var_namet != 'hola':
    print "bad namet", t.var_namet
    raise RuntimeError

t.var_namet = 'hol'

if t.var_namet != 'hol':
    # if t.var_namet != 'hol\0\0':
    print "bad namet", t.var_namet
    raise RuntimeError


cvar.var_char = '\0'
if cvar.var_char != '\0':
    raise RuntimeError, "bad char '0' case"

cvar.var_char = 0
if cvar.var_char != '\0':
    raise RuntimeError, "bad char '0' case"

cvar.var_namet = '\0'
# if cvar.var_namet != '\0\0\0\0\0':
if cvar.var_namet != '':
    print 'hola', '', cvar.var_namet
    raise RuntimeError, "bad char '\0' case"

cvar.var_namet = ''
# if cvar.var_namet != '\0\0\0\0\0':
if cvar.var_namet != '':
    raise RuntimeError, "bad char empty case"

cvar.var_pchar = None
if cvar.var_pchar != None:
    raise RuntimeError, "bad None case"

cvar.var_pchar = ''
if cvar.var_pchar != '':
    print '%c' % (cvar.var_pchar[0],)
    raise RuntimeError, "bad char empty case"

cvar.var_pcharc = None
if cvar.var_pcharc != None:
    raise RuntimeError, "bad None case"

cvar.var_pcharc = ''
if cvar.var_pcharc != '':
    raise RuntimeError, "bad char empty case"


#
# creating a raw char*
#
pc = new_pchar(5)
pchar_setitem(pc, 0, 'h')
pchar_setitem(pc, 1, 'o')
pchar_setitem(pc, 2, 'l')
pchar_setitem(pc, 3, 'a')
pchar_setitem(pc, 4, 0)


cvar.var_pchar = pc
if cvar.var_pchar != "hola":
    print cvar.var_pchar
    raise RuntimeError, "bad pointer case"

cvar.var_namet = pc
# if cvar.var_namet != "hola\0":
if cvar.var_namet != "hola":
    raise RuntimeError, "bad pointer case"

delete_pchar(pc)

#
# Now when things should fail
#


try:
    error = 0
    a = t.var_uchar
    t.var_uchar = 10000
    error = 1
except OverflowError:
    if a != t.var_uchar:
        error = 1
    pass
if error:
    raise RuntimeError, "bad uchar typemap"


try:
    error = 0
    a = t.var_char
    t.var_char = '23'
    error = 1
except TypeError:
    if a != t.var_char:
        error = 1
    pass
if error:
    raise RuntimeError, "bad char typemap"

try:
    error = 0
    a = t.var_ushort
    t.var_ushort = -1
    error = 1
except OverflowError:
    if a != t.var_ushort:
        error = 1
    pass
if error:
    raise RuntimeError, "bad ushort typemap"

try:
    error = 0
    a = t.var_uint
    t.var_uint = -1
    error = 1
except OverflowError:
    if a != t.var_uint:
        error = 1
    pass
if error:
    raise RuntimeError, "bad uint typemap"

try:
    error = 0
    a = t.var_sizet
    t.var_sizet = -1
    error = 1
except OverflowError:
    if a != t.var_sizet:
        error = 1
    pass
if error:
    raise RuntimeError, "bad sizet typemap"

try:
    error = 0
    a = t.var_ulong
    t.var_ulong = -1
    error = 1
except OverflowError:
    if a != t.var_ulong:
        error = 1
    pass
if error:
    raise RuntimeError, "bad ulong typemap"

#
#
try:
    error = 0
    a = t.var_namet
    t.var_namet = '123456'
    error = 1
except TypeError:
    if a != t.var_namet:
        error = 1
    pass
if error:
    raise RuntimeError, "bad namet typemap"

#
#
#
t2 = p.vtest(t)
if t.var_namet != t2.var_namet:
    raise RuntimeError, "bad SWIGTYPE* typemap"


if cvar.fixsize != 'ho\0la\0\0\0':
    raise RuntimeError, "bad FIXSIZE typemap"

cvar.fixsize = 'ho'
if cvar.fixsize != 'ho\0\0\0\0\0\0':
    raise RuntimeError, "bad FIXSIZE typemap"


f = Foo(3)
f1 = fptr_val(f)
f2 = fptr_ref(f)
if f1._a != f2._a:
    raise RuntimeError, "bad const ptr& typemap"


v = char_foo(1, 3)
if v != 3:
    raise RuntimeError, "bad int typemap"

s = char_foo(1, "hello")
if s != "hello":
    raise RuntimeError, "bad char* typemap"


v = SetPos(1, 3)
if v != 4:
    raise RuntimeError, "bad int typemap"

#
# Check the bounds for converting various types
#

# ctypes not available until 2.5
if sys.version_info[0:2] <= (2, 4):
    sys.exit(0)
import ctypes

# Get the minimum and maximum values that fit in signed char, short, int, long, and long long
overchar = 2 ** 7
while ctypes.c_byte(overchar).value > 0:
    overchar *= 2
minchar = -overchar
maxchar = overchar - 1
maxuchar = 2 * maxchar + 1
overshort = overchar
while ctypes.c_short(overshort).value > 0:
    overshort *= 2
minshort = -overshort
maxshort = overshort - 1
maxushort = 2 * maxshort + 1
overint = overshort
while ctypes.c_int(overint).value > 0:
    overint *= 2
minint = -overint
maxint = overint - 1
maxuint = 2 * maxint + 1
overlong = overint
while ctypes.c_long(overlong).value > 0:
    overlong *= 2
minlong = -overlong
maxlong = overlong - 1
maxulong = 2 * maxlong + 1
overllong = overlong
while ctypes.c_longlong(overllong).value > 0:
    overllong *= 2
minllong = -overllong
maxllong = overllong - 1
maxullong = 2 * maxllong + 1

# Make sure Python 2's sys.maxint is the same as the maxlong we calculated
if sys.version_info[0] <= 2 and maxlong != sys.maxint:
    raise RuntimeError, "sys.maxint is not the maximum value of a signed long"

def checkType(t, e, val, delta):
    """t = Test object, e = type name (e.g. ulong), val = max or min allowed value, delta = +1 for max, -1 for min"""
    error = 0
    # Set the extreme valid value for var_*
    setattr(t, 'var_' + e, val)
    # Make sure it was set properly and works properly in the val_* and ref_* methods
    if getattr(t, 'var_' + e) != val or getattr(t, 'val_' + e)(val) != val or getattr(t, 'ref_' + e)(val) != val:
        error = 1
    # Make sure setting a more extreme value fails without changing the value
    try:
        a = getattr(t, 'var_' + e)
        setattr(t, 'var_' + e, val + delta)
        error = 1
    except OverflowError:
        if a != getattr(t, 'var_' + e):
            error = 1
    # Make sure the val_* and ref_* methods fail with a more extreme value
    try:
        getattr(t, 'val_' + e)(val + delta)
        error = 1
    except OverflowError:
        pass
    try:
        getattr(t, 'ref_' + e)(val + delta)
        error = 1
    except OverflowError:
        pass
    if error:
        raise RuntimeError, "bad " + e + " typemap"

def checkFull(t, e, maxval, minval):
    """Check the maximum and minimum bounds for the type given by e"""
    checkType(t, e, maxval,  1)
    checkType(t, e, minval, -1)

checkFull(t, 'llong',  maxllong,  minllong)
checkFull(t, 'long',   maxlong,   minlong)
checkFull(t, 'int',    maxint,    minint)
checkFull(t, 'short',  maxshort,  minshort)
checkFull(t, 'schar',  maxchar,   minchar)
checkFull(t, 'ullong', maxullong, 0)
checkFull(t, 'ulong',  maxulong,  0)
checkFull(t, 'uint',   maxuint,   0)
checkFull(t, 'ushort', maxushort, 0)
checkFull(t, 'uchar',  maxuchar,  0)

def checkOverload(t, name, val, delta, prevval, limit):
    """
    Check that overloading works
        t = Test object
        name = type name (e.g. ulong)
        val = max or min allowed value
        delta = +1 for max, -1 for min
        prevval = corresponding value for one smaller type
        limit = most extreme value for any type
    """
    # If val == prevval, then the smaller typemap will win
    if val != prevval:
        # Make sure the most extreme value of this type gives the name of this type
        if t.ovr_str(val) != name:
            raise RuntimeError, "bad " + name + " typemap"
        # Make sure a more extreme value doesn't give the name of this type
        try:
            if t.ovr_str(val + delta) == name:
                raise RuntimeError, "bad " + name + " typemap"
            if val == limit:
                # Should raise NotImplementedError here since this is the largest integral type
                raise RuntimeError, "bad " + name + " typemap"
        except NotImplementedError:
            # NotImplementedError is expected only if this is the most extreme type
            if val != limit:
                raise RuntimeError, "bad " + name + " typemap"
        except TypeError:
            # TypeError is raised instead if swig is run with -O or -fastdispatch
            if val != limit:
                raise RuntimeError, "bad " + name + " typemap"

# Check that overloading works: uchar > schar > ushort > short > uint > int > ulong > long > ullong > llong
checkOverload(t, 'uchar',  maxuchar,  +1, 0,         maxullong)
checkOverload(t, 'ushort', maxushort, +1, maxuchar,  maxullong)
checkOverload(t, 'uint',   maxuint,   +1, maxushort, maxullong)
checkOverload(t, 'ulong',  maxulong,  +1, maxuint,   maxullong)
checkOverload(t, 'ullong', maxullong, +1, maxulong,  maxullong)
checkOverload(t, 'schar',  minchar,   -1, 0,         minllong)
checkOverload(t, 'short',  minshort,  -1, minchar,   minllong)
checkOverload(t, 'int',    minint,    -1, minshort,  minllong)
checkOverload(t, 'long',   minlong,   -1, minint,    minllong)
checkOverload(t, 'llong',  minllong,  -1, minlong,   minllong)

# Make sure that large ints can be converted to doubles properly
if val_double(sys.maxint + 1) != float(sys.maxint + 1):
    raise RuntimeError, "bad double typemap"
if val_double(-sys.maxint - 2) != float(-sys.maxint - 2):
    raise RuntimeError, "bad double typemap"


# Check the minimum and maximum values that fit in ptrdiff_t and size_t
def checkType(name, maxfunc, maxval, minfunc, minval, echofunc):
    if maxfunc() != maxval:
        raise RuntimeError, "bad " + name + " typemap"
    if minfunc() != minval:
        raise RuntimeError, "bad " + name + " typemap"
    if echofunc(maxval) != maxval:
        raise RuntimeError, "bad " + name + " typemap"
    if echofunc(minval) != minval:
        raise RuntimeError, "bad " + name + " typemap"
    error = 0
    try:
        echofunc(maxval + 1)
        error = 1
    except OverflowError:
        pass
    if error == 1:
        raise RuntimeError, "bad " + name + " typemap"
    try:
        echofunc(minval - 1)
        error = 1
    except OverflowError:
        pass
    if error == 1:
        raise RuntimeError, "bad " + name + " typemap"

# sys.maxsize is the largest value supported by Py_ssize_t, which should be the same as ptrdiff_t
if sys.version_info[0:2] >= (2, 6):
    checkType("ptrdiff_t", get_ptrdiff_max, sys.maxsize,           get_ptrdiff_min, -(sys.maxsize + 1), ptrdiff_echo)
    checkType("size_t",    get_size_max,    (2 * sys.maxsize) + 1, get_size_min,    0,                  size_echo)
