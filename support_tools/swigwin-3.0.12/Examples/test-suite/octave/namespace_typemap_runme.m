namespace_typemap

if (!strcmp(stest1("hello"),"hello"))
    error
endif

if (!strcmp(stest2("hello"),"hello"))
    error
endif

if (!strcmp(stest3("hello"),"hello"))
    error
endif

if (!strcmp(stest4("hello"),"hello"))
    error
endif

if (!strcmp(stest5("hello"),"hello"))
    error
endif

if (!strcmp(stest6("hello"),"hello"))
    error
endif

if (!strcmp(stest7("hello"),"hello"))
    error
endif

if (!strcmp(stest8("hello"),"hello"))
    error
endif

if (!strcmp(stest9("hello"),"hello"))
    error
endif

if (!strcmp(stest10("hello"),"hello"))
    error
endif

if (!strcmp(stest11("hello"),"hello"))
    error
endif

if (!strcmp(stest12("hello"),"hello"))
    error
endif

c = complex(2,3);
r = real(c);

if (ctest1(c) != r)
    error
endif

if (ctest2(c) != r)
    error
endif

if (ctest3(c) != r)
    error
endif

if (ctest4(c) != r)
    error
endif

if (ctest5(c) != r)
    error
endif

if (ctest6(c) != r)
    error
endif

if (ctest7(c) != r)
    error
endif

if (ctest8(c) != r)
    error
endif

if (ctest9(c) != r)
    error
endif

if (ctest10(c) != r)
    error
endif

if (ctest11(c) != r)
    error
endif

if (ctest12(c) != r)
    error
endif

try
    ttest1(-14)
    error
catch
end_try_catch
