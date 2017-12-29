# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

overload_simple_cast

Ai=@(x) subclass('x',x,'__int',@(self) self.x);
Ad=@(x) subclass('x',x,'__float',@(self) self.x);

ai = Ai(4);

ad = Ad(5.0);
add = Ad(5.5);

try
    fint(add);
    good = 0;
catch
    good = 1;
end_try_catch

if (!good)
    error("fint(int)")
endif


if (!strcmp(fint(ad),"fint:int"))
    error("fint(int)")
endif

if (!strcmp(fdouble(ad),"fdouble:double"))
    error("fdouble(double)")
endif

if (!strcmp(fint(ai),"fint:int"))
    error("fint(int)")
endif

if (!strcmp(fint(5.0),"fint:int"))
    error("fint(int)")
endif
    
if (!strcmp(fint(3),"fint:int"))
    error("fint(int)")
endif
if (!strcmp(fint(3.0),"fint:int"))
    error("fint(int)")
endif

if (!strcmp(fdouble(ad),"fdouble:double"))
    error("fdouble(double)")
endif
if (!strcmp(fdouble(3),f"fdouble:double"))
    error("fdouble(double)")
endif
if (!strcmp(fdouble(3.0),"fdouble:double"))
    error("fdouble(double)")
endif

if (!strcmp(fid(3,3.0),"fid:intdouble"))
    error("fid:intdouble")
endif

if (!strcmp(fid(3.0,3),"fid:doubleint"))
    error("fid:doubleint")
endif

if (!strcmp(fid(ad,ai),"fid:doubleint"))
    error("fid:doubleint")
endif

if (!strcmp(fid(ai,ad),"fid:intdouble"))
    error("fid:intdouble")
endif



if (!strcmp(foo(3),"foo:int"))
    error("foo(int)")
endif

if (!strcmp(foo(3.0),"foo:double"))
    error("foo(double)")
endif

if (!strcmp(foo("hello"),"foo:char *"))
    error("foo(char *)")
endif

f = Foo();
b = Bar();

if (!strcmp(foo(f),"foo:Foo *"))
    error("foo(Foo *)")
endif

if (!strcmp(foo(b),"foo:Bar *"))
    error("foo(Bar *)")
endif

v = malloc_void(32);

if (!strcmp(foo(v),"foo:void *"))
    error("foo(void *)")
endif

s = Spam();

if (!strcmp(s.foo(3),"foo:int"))
    error("Spam::foo(int)")
endif

if (!strcmp(s.foo(3.0),"foo:double"))
    error("Spam::foo(double)")
endif

if (!strcmp(s.foo("hello"),"foo:char *"))
    error("Spam::foo(char *)")
endif

if (!strcmp(s.foo(f),"foo:Foo *"))
    error("Spam::foo(Foo *)")
endif

if (!strcmp(s.foo(b),"foo:Bar *"))
    error("Spam::foo(Bar *)")
endif

if (!strcmp(s.foo(v),"foo:void *"))
    error("Spam::foo(void *)")
endif

if (!strcmp(Spam_bar(3),"bar:int"))
    error("Spam::bar(int)")
endif

if (!strcmp(Spam_bar(3.0),"bar:double"))
    error("Spam::bar(double)")
endif

if (!strcmp(Spam_bar("hello"),"bar:char *"))
    error("Spam::bar(char *)")
endif

if (!strcmp(Spam_bar(f),"bar:Foo *"))
    error("Spam::bar(Foo *)")
endif

if (!strcmp(Spam_bar(b),"bar:Bar *"))
    error("Spam::bar(Bar *)")
endif

if (!strcmp(Spam_bar(v),"bar:void *"))
    error("Spam::bar(void *)")
endif

# Test constructors

s = Spam();
if (!strcmp(s.type,"none"))
    error("Spam()")
endif

s = Spam(3);
if (!strcmp(s.type,"int"))
    error("Spam(int)")
endif
    
s = Spam(3.4);
if (!strcmp(s.type,"double"))
    error("Spam(double)")
endif

s = Spam("hello");
if (!strcmp(s.type,"char *"))
    error("Spam(char *)")
endif

s = Spam(f);
if (!strcmp(s.type,"Foo *"))
    error("Spam(Foo *)")
endif

s = Spam(b);
if (!strcmp(s.type,"Bar *"))
    error("Spam(Bar *)")
endif

s = Spam(v);
if (!strcmp(s.type,"void *"))
    error("Spam(void *)")
endif


# unsigned long long
ullmax = 9223372036854775807; #0xffffffffffffffff
ullmaxd = 9007199254740992.0;
ullmin = 0;
ullmind = 0.0;
if (ull(ullmin) != ullmin)
    error("ull(ullmin)")
endif
if (ull(ullmax) != ullmax)
    error("ull(ullmax)")
endif
if (ull(ullmind) != ullmind)
    error("ull(ullmind)")
endif
if (ull(ullmaxd) != ullmaxd)
    error("ull(ullmaxd)")
endif

# long long
llmax = 9223372036854775807; #0x7fffffffffffffff
llmin = -9223372036854775808;
# these are near the largest  floats we can still convert into long long
llmaxd = 9007199254740992.0;
llmind = -9007199254740992.0;
if (ll(llmin) != llmin)
    error("ll(llmin)")
endif
if (ll(llmax) != llmax)
    error("ll(llmax)")
endif
if (ll(llmind) != llmind)
    error("ll(llmind)")
endif
if (ll(llmaxd) != llmaxd)
    error("ll(llmaxd)")
endif

free_void(v);

