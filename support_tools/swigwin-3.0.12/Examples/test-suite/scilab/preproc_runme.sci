exec("swigtest.start", -1);

if endif_get() <> 1 then swigtesterror(); end
if define_get() <> 1 then swigtesterror(); end
if defined_get() <> 1 then swigtesterror(); end
if 2 * one_get() <> two_get() then swigtesterror(); end

exec("swigtest.quit", -1);
