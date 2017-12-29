exec("swigtest.start", -1);

try
    foo_2();
catch
    swigtesterror();
end
if bar_2_get() <> 17 then swigtesterror(); end
if Baz_2_get() <> 47 then swigtesterror(); end

exec("swigtest.quit", -1);
