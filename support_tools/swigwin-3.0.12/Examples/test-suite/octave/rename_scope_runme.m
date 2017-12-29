rename_scope

a = Natural_UP();
b = Natural_BP();

if (a.rtest() != 1)
    error
endif

if (b.rtest() != 1)
    error
endif

f = @equals;

