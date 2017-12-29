exec("swigtest.start", -1);

if typeof(CSP_ITERATION_FWD_get()) <> "constant" then swigtesterror(); end
if typeof(CSP_ITERATION_BWD_get()) <> "constant" then swigtesterror(); end
if typeof(ABCDE_get()) <> "constant" then swigtesterror(); end
if typeof(FGHJI_get()) <> "constant" then swigtesterror(); end

try
    bar1(CSP_ITERATION_FWD_get())
    bar1(CSP_ITERATION_BWD_get())
    bar1(1)
    bar1(int32(1))

    bar2(ABCDE_get())
    bar2(FGHJI_get())
    bar2(1)
    bar2(int32(1))

    bar3(ABCDE_get())
    bar3(FGHJI_get())
    bar3(1)
    bar3(int32(1))
catch
    swigtesterror()
end

if typeof(enumInstance_get()) <> "constant" then swigtesterror(); end
if enumInstance_get() <> 2 then swigtesterror(); end

if typeof(Slap_get()) <> "constant" then swigtesterror(); end
if Slap_get() <> 10 then swigtesterror(); end

if typeof(Mine_get()) <> "constant" then swigtesterror(); end
if Mine_get() <> 11 then swigtesterror(); end

if typeof(Thigh_get()) <> "constant" then swigtesterror(); end
if Thigh_get() <> 12 then swigtesterror(); end

exec("swigtest.quit", -1);
