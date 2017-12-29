# Note: This example assumes that namespaces are flattened
cpp_namespace

n = cpp_namespace.fact(4);
if (n != 24)
    error("Bad return value!")
endif

if (cpp_namespace.cvar.Foo != 42)
    error("Bad variable value!")
endif

t = cpp_namespace.Test();
if (!strcmp(t.method(),"Test::method"))
    error("Bad method return value!")
endif

if (!strcmp(cpp_namespace.do_method(t),"Test::method"))
    error("Bad return value!")
endif

if (!strcmp(cpp_namespace.do_method2(t),"Test::method"))
    error("Bad return value!")
endif
    
cpp_namespace.weird("hello", 4);

clear t;

t2 = cpp_namespace.Test2();
t3 = cpp_namespace.Test3();
t4 = cpp_namespace.Test4();
t5 = cpp_namespace.Test5();

if (cpp_namespace.foo3(42) != 42)
    error("Bad return value!")
endif

if (!strcmp(cpp_namespace.do_method3(t2,40),"Test2::method"))
    error("Bad return value!")
endif

if (!strcmp(cpp_namespace.do_method3(t3,40),"Test3::method"))
    error("Bad return value!")
endif

if (!strcmp(cpp_namespace.do_method3(t4,40),"Test4::method"))
    error("Bad return value!")
endif

if (!strcmp(cpp_namespace.do_method3(t5,40),"Test5::method"))
    error("Bad return value!")
endif

    
