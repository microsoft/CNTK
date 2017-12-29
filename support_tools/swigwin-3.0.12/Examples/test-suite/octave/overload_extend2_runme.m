overload_extend2

f = overload_extend2.Foo();
if (f.test(3) != 1)
    error
endif
if (f.test("hello") != 2)
    error
endif
if (f.test(3.5,2.5) != 3)
    error
endif
if (f.test("hello",20) != 1020)
    error
endif
if (f.test("hello",20,100) != 120)
    error
endif

# C default args
if (f.test(f) != 30)
    error
endif
if (f.test(f,100) != 120)
    error
endif
if (f.test(f,100,200) != 300)
    error
endif

