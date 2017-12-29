varargs

if (!strcmp(varargs.test("Hello"),"Hello"))
    error("Failed")
endif

f = varargs.Foo("Greetings");
if (!strcmp(f.str,"Greetings"))
    error("Failed")
endif

if (!strcmp(f.test("Hello"),"Hello"))
    error("Failed")
endif


if (!strcmp(varargs.test_def("Hello",1),"Hello"))
    error("Failed")
endif

if (!strcmp(varargs.test_def("Hello"),"Hello"))
    error("Failed")
endif
