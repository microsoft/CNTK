using_private

f = FooBar();
f.x = 3;

if (f.blah(4) != 4)
    error, "blah(int)"
endif

if (f.defaulted() != -1)
    error, "defaulted()"
endif

if (f.defaulted(222) != 222)
    error, "defaulted(222)"
endif
