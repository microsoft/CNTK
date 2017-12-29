exec("swigtest.start", -1);

if CHAR_CONSTANT_get() <> "x" then swigtesterror(); end
if STRING_CONSTANT_get() <> "xyzzy" then swigtesterror(); end
if ESC_CONST_get() <> ascii(1) then swigtesterror(); end
if ia_get() <> ascii('a') then swigtesterror(); end
if ib_get() <> ascii('b') then swigtesterror(); end

exec("swigtest.quit", -1);
