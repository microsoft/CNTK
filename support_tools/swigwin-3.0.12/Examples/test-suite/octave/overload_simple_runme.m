overload_simple

# unless explicitly casted via {{u}int{8,16,32,64},double,single},
# octave will take numeric literals as doubles.

if (!strcmp(foo(3),"foo:int"))
    error("foo(int)")
endif

if (!strcmp(foo(3.1),"foo:double"))
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

if (!strcmp(s.foo(3.1),"foo:double"))
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

if (!strcmp(Spam_bar(3.1),"bar:double"))
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

free_void(v);

a = ClassA();
b = a.method1(1);
